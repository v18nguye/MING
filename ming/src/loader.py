import pdb
import os
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from .base import MolMetric
from .dataset import MolDataset, MOSESDataset
from .diffusion import Diffuser

NUM_MOL_SAMPLER = int(os.environ['NUM_MOL_SAMPLER'])
SAMPLER_BATCH = int(os.environ['SAMPLER_BATCH'])

def _load_mol_data(wdir, dataset, batch_size, is_sampling):
    '''load dataset
    @params 
        dataset: str
            name of dataset
        is_sampling: bool
            if sampling phase
    '''
    proc_dir = wdir + '/processed_data/'
    num_threads =  min(torch.get_num_threads(), 8)
    name = dataset
    
    # load only train data
    if dataset != 'MOSES':
        train_data = MolDataset(is_train=True, root=proc_dir, name=name)[0]
    else:
        train_data = MOSESDataset(stage='train', root=proc_dir, name=name)[0]
    
    if NUM_MOL_SAMPLER is not None:
        if is_sampling:
            # 3x samplings
            sampler_data = [Subset(train_data, torch.randperm(len(train_data))[:NUM_MOL_SAMPLER]) for _ in range(3)]
            sampler_loader = [DataLoader(x, batch_size=SAMPLER_BATCH, 
                                    shuffle=False, pin_memory=True, num_workers=num_threads) for x in sampler_data]
        else:
            # training
            sampler_data = Subset(train_data, torch.randperm(len(train_data))[:NUM_MOL_SAMPLER])
            sampler_loader = DataLoader(sampler_data, batch_size=SAMPLER_BATCH, 
                                           shuffle=False, pin_memory=True, num_workers=num_threads)
    else:
        raise ValueError('NUM_MOL_SAMPLER is None')

    train_loader = DataLoader(train_data, batch_size=batch_size, 
                        shuffle=True, pin_memory=True, num_workers=num_threads)
    num_fourier_data = train_data[0][3].shape[-1]
    print('load data: {}, train: {}, sampler: {}'.format(dataset, len(train_data), len(sampler_data)))
    return train_loader, sampler_loader, num_fourier_data

def load_data(wdir, dataset, batch_size, is_sampling=False):
    if dataset in ['ZINC250k', 'QM9', 'MOSES']:
        train_loader, sampler_loader, num_fourier_data = _load_mol_data(wdir, dataset, batch_size, is_sampling)
    else:
        raise ValueError('Cannot find the dataset: ' + dataset)
    return train_loader, sampler_loader, num_fourier_data

def load_diffuser(cfg, input_dim, metric):
    '''load diffuser
    @params
        ckpt_base: str
            ckpt base path
    '''
    lr_in=cfg.train.lr_inloop
    lr_out=cfg.train.lr_outloop
    num_inner_steps=cfg.train.num_inner_steps
    lr_patience=cfg.train.lr_patience
    reg_z=cfg.train.reg_z
    loss_weight=cfg.train.loss_weight
    
    hidden_dim=cfg.model.hidden_dim
    latent_dim=cfg.model.latent_dim
    n_layers=cfg.model.n_layers
    beta_schedule=cfg.model.beta_schedule
    beta_start=cfg.model.beta_start
    beta_end=cfg.model.beta_end
    num_diffusion_timesteps=cfg.model.num_diffusion_timesteps
    
    output_dim_node = cfg.data.output_node
    output_dim_edge = cfg.data.output_edge
    dataset = cfg.data.name
    
    diffuser = Diffuser(input_dim,
        output_dim_node,
        output_dim_edge,             
        hidden_dim,
        n_layers,
        lr_in,
        lr_out,
        num_inner_steps,
        lr_patience,
        latent_dim,
        beta_schedule,
        beta_start,
        beta_end,
        num_diffusion_timesteps,
        metric,
        reg_z,
        loss_weight,
        dataset
        )
    return diffuser

def load_metric(dataset):
    if dataset in ['QM9', 'ZINC250k', 'MOSES']:
        metrics = MolMetric(dataset)
    else:
        raise ValueError(f'Metrics not exist for {dataset}')
    return metrics