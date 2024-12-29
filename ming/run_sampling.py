import os, json
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.loader import load_data, load_diffuser, load_metric
from src.utils import create_exp_dir

def count_num_param(model):
    return sum(p.numel() for p in model.parameters())

def collect_res(res_list, dataset):
    ''' collect result
    '''
    if dataset != 'MOSES':
        res_dict = {
            'valid_wo_correct': [],
            'Novelty': [],
            'Unique': [],
            'nspdk_mmd': [],
            'FCD': [],
            'VUN': []}
    else:
        res_dict = {
            'valid_wo_correct': [],
            'Novelty': [],
            'Unique': [],
            'FCD': [],
            'VUN': []}

    for res in res_list:
        for k in res.keys():
            if 'test' in k:
                if 'unique' in k.lower():
                    res_dict['Unique'].append(res[k])
                else:
                    res_dict[k.split('/')[-1]].append(res[k])
                    
    for k in res_dict.keys():
        res_dict[k] = res_dict[k] + [np.mean(res_dict[k]), np.std(res_dict[k])]
    return res_dict

@hydra.main(config_path='../config', config_name='base')
def main(cfg: DictConfig):
    # get wd
    pl.seed_everything(cfg.train.seed)
    wdir = os.getcwd().split('outputs')[0]
    os.chdir(wdir)
    
    # hyparams
    _dataset = cfg.data.name
    _batch_size = cfg.train.bsize
    _max_epochs = cfg.train.max_epochs
    _name_exp = cfg.train.name_exp
    _proj = cfg.train.proj
    _offline = True # set online
    _measure_on = cfg.train.measure_on
    _mode = 'min' if _measure_on == 'val/FCD' else 'max'
    _check_val_every_n_epoch = cfg.train.check_val_every_n_epoch

    # callback monitor 
    dir_path = '../ckpts/'+_proj+'/'+_name_exp
    create_exp_dir(dir_path)
    
    
    checkpoint_cb = ModelCheckpoint(monitor=_measure_on, mode=_mode, filename="best", dirpath=dir_path, save_last=False) # val monitor, save last of eval
    logger = WandbLogger(project=_proj, name=_name_exp, save_dir=dir_path, offline=_offline)
    
    # data loader 
    _, sampler_data, num_fourier_data = load_data(wdir, _dataset, _batch_size, is_sampling=True)
    
    # trainer
    trainer = pl.Trainer(
        log_every_n_steps=1, 
        check_val_every_n_epoch=_check_val_every_n_epoch,
        max_epochs=_max_epochs, 
        logger=logger,
        devices=torch.cuda.device_count(),
        accelerator='gpu',
        inference_mode=False,
        strategy="ddp" if torch.cuda.device_count() > 1 else None
    )

    # load metric
    metric = load_metric(_dataset)
    
    # load model
    _n_fourier = cfg.model.n_fourier
    if _n_fourier is None:
        input_dim = num_fourier_data
    else:
        input_dim = min(_n_fourier, num_fourier_data)
    
    diffuser = load_diffuser(cfg, input_dim, metric=metric)
    diffuser = diffuser.load_from_checkpoint(checkpoint_cb.dirpath+'/best.ckpt')
    diffuser.dataset = _dataset
    diffuser.metric = metric
    diffuser.eval()
    
    # sampling
    res_list = []
    for loader in sampler_data:
        res = trainer.test(diffuser, loader) # set val -> train
        res_list.append(res[0])
    combine_res = collect_res(res_list, _dataset)
    file_save = dir_path+f'/sampling_result.json'

    with open(file_save, 'w') as f:
        json.dump(combine_res, f)

if __name__ == '__main__':
    main()