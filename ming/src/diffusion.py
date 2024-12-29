import pdb
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import Model
from torch.optim import lr_scheduler
from src.utils import get_beta_schedule, mask_minibatch

class Diffuser(pl.LightningModule):
    """
    Arguments:
        input_dim: int, size of eigvec
        output_dim_node: int, size of node dim
        output_dim_edge: int, size of edge dim
        hidden_dim: int, number of neurons in hidden layers
        n_layers: int, number of layers (total, including first and last)
        lr_in: float, expectation opt lr
        lr_out: float, maximisation opt lr
        num_inner_steps: int, number of inner opt steps
        lr_patience: int,
        latent_dim: int, z dim
        beta_schedule: str,
        beta_start: float,
        beta_end: float,
        num_diffusion_timesteps: int,
        metric: Metric,
        reg_z: bool, if regularize z
        loss_weight: float,
        lr_patience: int, learning rate patience (in number of epochs)
        latent_dim: int, size of the latents
    """

    def __init__(
        self,
        input_dim: int,
        output_dim_node: int,
        output_dim_edge: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
        lr_in: float=None,
        lr_out: float=None,
        num_inner_steps: int=None,
        lr_patience: int = 500,
        latent_dim: int = 256,
        beta_schedule: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_diffusion_timesteps: int = 1000,
        metric: any=None,
        reg_z: bool=None,
        loss_weight: float=0.001,
        dataset: str=None
    ):
        super().__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.output_dim_node = output_dim_node
        self.output_dim_edge = output_dim_edge
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr_in = lr_in
        self.lr_out = lr_out
        self.num_inner_steps = num_inner_steps
        self.lr_patience = lr_patience
        self.metric = metric
        self.reg_z = reg_z
        self.loss_weight=loss_weight
        
        self.sample_outputs = []
        self.sync_dist = torch.cuda.device_count() > 1

        # modules
        inp_dim_s1 = input_dim + 1
        inp_dim_s2 = input_dim + 1 # time TODO test also cosine embedding
        out_dim = output_dim_edge + output_dim_node - 1 # -1 as having two vitr indices
        inp_dim_m = latent_dim

        self.model = Model(
            inp_dim_s1,
            out_dim,
            hidden_dim,
            inp_dim_m,
            n_layers)

        self.denoiser = Model(
            inp_dim_s2,
            out_dim,
            hidden_dim,
            inp_dim_m,
            n_layers)

        # diffusion
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        # betas
        betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)
        self.betas = torch.from_numpy(betas)
        self.alphas = 1 - self.betas
        self.one_minus_alphas = 1 - self.alphas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.one_minus_alphas_cumprod = 1. - self.alphas_cumprod

        # loss
        self.loss_ce = nn.CrossEntropyLoss(reduction='mean')
        self.loss_mse = nn.MSELoss(reduction='mean')
        self.loss_mse_inner = nn.MSELoss(reduction='sum') # mean or sum
        self.save_hyperparameters(ignore=['metric'])
    
    def scale_eigvec(self, eigvec, mask):
        '''
        '''
        num_node = mask.sum(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        eigvec = eigvec * torch.sqrt(num_node)
        return eigvec
    
    def get_edge_mask(self, mask, return_diag=False):
        '''
        @params
            mask: (bs, n)
                node mask
        '''
        bs, n = mask.shape[0], mask.shape[1]
        mask_row = mask.unsqueeze(2).unsqueeze(-1)
        mask_col = mask.unsqueeze(1).unsqueeze(-1)
        edge_mask = (mask_row*mask_col).gt(0)
        diag_mask = (~torch.eye(n, device=mask.device).gt(0)).unsqueeze(0).unsqueeze(-1).repeat(bs,1,1,1)
        if not return_diag:
            return edge_mask # (bs, n, n, 1)
        else: 
            return edge_mask, diag_mask # (bs, n, n, 1)    
    
    def get_model_data(self, eigvec, node_gt, edge_gt, device):
        '''
        @params:
            eigvec: (bs, n, f)
            node_gt: (bs, n, num_max_atom +1)
            edge_gt: (bs, n, n, num_max_bond +1)
        '''
        bs, n = edge_gt.shape[0], edge_gt.shape[1]
        
        _eigvec1= eigvec.unsqueeze(1)
        _eigvec2= eigvec.unsqueeze(2)
        inp = _eigvec1*_eigvec2 # (bs, n, n, f) 
    
        _node_gt = node_gt.argmax(-1, keepdim=True)
        _eye = torch.eye(n,n, device=device).unsqueeze(0).repeat(bs,1,1)
        _node_gt = _eye * _node_gt
        _node_gt = F.one_hot(_node_gt.long(), self.output_dim_node)
        _node_gt = _node_gt[:,:,:,1:] # (bs, n, n, max_num_node)
        _egde_gt = edge_gt*(~_eye.unsqueeze(-1).gt(0))*1.0
        out = torch.concat((_egde_gt, _node_gt), dim=-1) # (bs, n, n, 1 + max_num_bond + max_num_node)
        assert not (out.sum(-1) > 1).any()
        return inp, out
    
    def time_encoding(self, t_batch):
        return t_batch
    
    def _process_input(self, data):

        if len(data) == 5: # mols
            node_gt, edge_gt, eigvec = mask_minibatch(data)
        else:
            raise ValueError('data format not correct !')

        mask = node_gt[:,:,1:].sum(-1).gt(0) # (bs, n)
        device  = edge_gt.device
        bs, n = edge_gt.shape[0],  edge_gt.shape[1]
        edge_mask, diag_mask = self.get_edge_mask(mask, return_diag=True) # (bs, n, n, 1)
        
        if self.input_dim <= eigvec.shape[-1]:
            eigvec = eigvec[:,:,:self.input_dim]
        else:
            bs, n, dim = eigvec.shape[0],  eigvec.shape[1], eigvec.shape[2]
            _eigvec = torch.zeros((bs,n,self.input_dim), device=eigvec.device)
            _eigvec[:,:,:dim] = eigvec
            eigvec = _eigvec
            del _eigvec
            
        eigvec = self.scale_eigvec(eigvec, mask)
        edge_gt = F.one_hot(edge_gt.long(), self.output_dim_edge).float()
        inp, out = self.get_model_data(eigvec, node_gt, edge_gt, device)
        
        return inp, out, mask, edge_mask, diag_mask, device
    
    def get_noise_samples(self, data, device, mask=None):
        '''
        @params
            data: (bs, n, n, f)
            mask: (bs, n)
                node mask
        '''
        bs, n = data.shape[0], data.shape[1]
        
        noise = torch.randn_like(data, device=device)
        
        t_batch = torch.randint(0,self.num_diffusion_timesteps, (bs,))
        t_batch = t_batch.unsqueeze(-1).unsqueeze(-1).repeat(1, n, n)
        _t_batch = t_batch.reshape(1,-1)

        sac = self.sqrt_alphas_cumprod[_t_batch].to(device).reshape(bs, n, n).unsqueeze(-1) # (bs, n, n, 1)
        somac = self.sqrt_one_minus_alphas_cumprod[_t_batch].to(device).reshape(bs, n, n).unsqueeze(-1)  # (bs, n, n, 1)

        t_batch = t_batch.unsqueeze(-1).to(device) # (bs, n, n , 1)
        t_batch = t_batch/self.num_diffusion_timesteps

        noise_data = sac*data + somac*noise # (bs, n, n, f)

        t_batch = self.time_encoding(t_batch)

        if mask is None:
            return t_batch, noise_data
        else:
            _mask = self.get_edge_mask(mask) # (bs, n, n, 1)
            return t_batch*_mask, noise_data * _mask

    def get_z(self, bs, n, device, z0=None, grad=None):
        '''
        '''
        if z0 is None:
            z0 = torch.zeros((bs, n, n, self.latent_dim), device=device, requires_grad=True)
        else:
            if grad is None:
                raise ValueError('Grad must be not None')
            z0 = z0 - self.lr_in*grad
        return z0

    def inner_opt(self, y_t, inp, bs, n, edge_mask,  device, is_train):
        '''inner optimization
        '''
        if is_train:
            self.model.eval()
        self.model.zero_grad()

        with torch.enable_grad():
            z0 = self.get_z(bs, n, device) 

            if not z0.requires_grad:
                raise ValueError('z_0 should be initialized with grads !') 
            
            y_t = y_t[edge_mask.squeeze()] 
            for _ in range(self.num_inner_steps):
                pred = self.model(inp, z0)[edge_mask.squeeze()] # post-masking
                loss = self.loss_mse_inner(y_t, pred)
                
                if self.reg_z:
                    loss_z = z0.norm()
                    loss = loss + self.loss_weight*loss_z 
                    
                grad = torch.autograd.grad(loss, [z0], allow_unused=True, retain_graph=False)[0]
                z0 = self.get_z(bs, n, device, z0, grad) 

        if is_train:
            self.model.train()       
        self.model.zero_grad()
        
        return z0.detach()

    def forward(self, y_t, inp_t, inp, edge_mask):
        '''
        @params:
            inp_t:
                concat input and time
        '''
        is_train = self.model.training
        bs, n = inp.shape[0], inp.shape[1]
        z_t = self.inner_opt(y_t, inp, bs, n, edge_mask, y_t.device, is_train)

        # outer opt
        if is_train:
            pred_y_0 = self.denoiser(inp_t, z_t)
            pred_y_t = self.model(inp, z_t)
            
            return pred_y_0, pred_y_t
        else:
            with torch.no_grad():
                pred_y_0 = self.denoiser(inp_t, z_t)
            return pred_y_0

    def training_step(self, data):
        '''
        '''
        inp, y_0, mask, edge_mask, _, device = self._process_input(data)

        t_batch, y_t = self.get_noise_samples(y_0, device, mask)
        inp_t = torch.concat([inp, t_batch], dim=-1).float()

        pred_y_0, pred_y_t = self(y_t, inp_t, inp_t, edge_mask)

        # element-wise prediction
        y_0 = y_0[edge_mask.squeeze()].argmax(-1)
        pred_y_0 = pred_y_0[edge_mask.squeeze()]

        y_t = y_t[edge_mask.squeeze()]
        pred_y_t = pred_y_t[edge_mask.squeeze()]

        y0_loss = self.loss_ce(pred_y_0, y_0)
        yt_loss = self.loss_mse(pred_y_t, y_t)

        loss = yt_loss + y0_loss
        
        self.log('train/y0-loss', y0_loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log('train/maximisation-loss', loss, prog_bar=True, sync_dist=self.sync_dist)
        return loss
    
    def reverse_diff(self, batch):
        '''reverse diffusion
        '''
        inp, y_0, mask, edge_mask, diag_mask, device = self._process_input(batch)
        bs, n = inp.shape[0], inp.shape[1]

        y_t = torch.randn_like(y_0, device=device)
        y_t = 1/2*(y_t + y_t.transpose(1,2))

        for i in range(1, self.num_diffusion_timesteps):
            current_t = self.num_diffusion_timesteps - i
            t_batch = self.get_batch_t(bs, n, current_t, device)
            inp_t = torch.concat([inp, t_batch], dim=-1).float()

            pred_y_0 = self(y_t, inp_t, inp_t, edge_mask)

            noise = torch.randn_like(y_t, device=device)
            
            y_t = self.sampling(bs, n, noise, y_t, pred_y_0, current_t, device, edge_mask)
            y_t = 1/2*(y_t + y_t.transpose(1,2))
            
        node_pred, edge_pred = self.get_pred_xa(y_t, mask, edge_mask, diag_mask, bs, n, device)
        self.sample_outputs.append([node_pred.long().detach().cpu(), edge_pred.long().detach().cpu()])

    def validation_step(self, batch, batch_idx):
        '''
        '''
        if self.current_epoch > 0:
            self.reverse_diff(batch)

    def test_step(self, batch, batch_idx):
        '''
        '''
        self.reverse_diff(batch)

    def on_validation_epoch_end(self):
        '''
        '''
        if self.current_epoch > 0:
            self.run_metric(is_train=True)

    def on_test_epoch_end(self):
        '''
        '''
        self.run_metric(is_train=False)

    def run_metric(self, is_train=True):
        '''
        @params
            eval_data:
                run metric on train data
        '''
        edge_preds, node_preds = [], []
        for batch in self.sample_outputs:
            for i in range(len(batch[0])):
                node_preds.append(batch[0][i])
                edge_preds.append(batch[1][i])
                
        self.sample_outputs.clear()  # free memory
        self.compute_mol_metric(node_preds, edge_preds, is_train=is_train)

    @torch.no_grad()
    def sampling(self, bs, n, noise, y_t, pred_y_0, current_t, device, edge_mask):
        '''
        '''
        t_batch = current_t*torch.ones((bs,n,n))
        t_batch_prev = current_t*torch.ones((bs,n,n)) - 1
        
        sqrt_alpha = self.sqrt_alphas[t_batch.reshape(1,-1).long()].reshape(bs,n,n).unsqueeze(-1).to(device)
        one_minus_alpha_cumprod = self.one_minus_alphas_cumprod[t_batch.reshape(1,-1).long()].reshape(bs,n,n).unsqueeze(-1).to(device)
        sqrt_alpha_cumprod_prev = self.sqrt_alphas_cumprod[t_batch_prev.reshape(1,-1).long()].reshape(bs,n,n).unsqueeze(-1).to(device)
        one_minus_alpha_cumprod_prev = self.one_minus_alphas_cumprod[t_batch_prev.reshape(1,-1).long()].reshape(bs,n,n).unsqueeze(-1).to(device)
        beta = self.betas[t_batch.reshape(1,-1).long()].reshape(bs,n,n).unsqueeze(-1).to(device)
        
        beta_tilde = one_minus_alpha_cumprod_prev/one_minus_alpha_cumprod*beta
        mean_posterior_coeff1 = sqrt_alpha_cumprod_prev/one_minus_alpha_cumprod*beta
        mean_posterior_coeff2 = sqrt_alpha/one_minus_alpha_cumprod*one_minus_alpha_cumprod_prev
        
        y_t_prev = mean_posterior_coeff1*pred_y_0 + mean_posterior_coeff2*y_t + torch.sqrt(beta_tilde)*noise
        return y_t_prev*edge_mask   
    
    def get_batch_t(self, bs, n, current_t, device):
        '''
        '''
        t_batch = current_t*torch.ones((bs,n,n))
        t_batch = t_batch/self.num_diffusion_timesteps
        t_batch = self.time_encoding(t_batch).unsqueeze(-1).to(device)
        return t_batch
    
    def get_pred_xa(self, out_t, mask, edge_mask, diag_mask, bs, n, device):
        '''
        '''
        out_t =  out_t * edge_mask
        edge_pred = out_t[:,:,:,:self.output_dim_edge]*diag_mask
        edge_pred = edge_pred.argmax(-1)
        
        if self.output_dim_node > 1:
            node_pred = torch.concat((out_t[:,:,:,:1], out_t[:,:,:,self.output_dim_edge:]), dim=-1)
            node_pred = node_pred*(~diag_mask)
            node_pred = node_pred.argmax(-1).float()
            node_pred = torch.bmm(node_pred, torch.ones((bs, n, 1), device=device)).squeeze()
            node_pred = node_pred * mask
        else:
            raise ValueError('node_pred is invalid!')
        return node_pred, edge_pred

    def compute_mol_metric(self, node_preds, edge_preds, is_train= True):
        '''
        '''
        task = 'val' if is_train else 'sampling'
        _, result = self.metric(node_preds, edge_preds)

        unique_metric = ''
        for key in result.keys():
            if 'unique' in key:
                unique_metric = key

        prod = result['valid_wo_correct']*result[f'{unique_metric}']*result['Novelty']
        self.log(f'{task}/valid_wo_correct', result['valid_wo_correct'], prog_bar=True, sync_dist=self.sync_dist)
        self.log(f'{task}/Novelty', result['Novelty'], prog_bar=True, sync_dist=self.sync_dist)
        self.log(f'{task}/VUN', prod, prog_bar=True, sync_dist=self.sync_dist)
        self.log(f'{task}/{unique_metric}', result[f'{unique_metric}'], prog_bar=True, sync_dist=self.sync_dist)
        self.log(f'{task}/FCD', result['FCD/Test'], prog_bar=True, sync_dist=self.sync_dist)
        
        if self.dataset != 'MOSES':
            self.log(f'{task}/nspdk_mmd', result['nspdk_mmd'], prog_bar=True, sync_dist=self.sync_dist)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_out)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )
        sch_dict = {"scheduler": scheduler, "monitor": "train/maximisation-loss", "frequency": 1} # training monitor
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}