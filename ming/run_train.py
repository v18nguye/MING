import os, pdb
import hydra
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from src.loader import load_data, load_diffuser, load_metric
from src.utils import create_exp_dir, check_ckpt_exist

@hydra.main(config_path='../config', config_name='base')
def main(cfg: DictConfig):
    # get wd
    pl.seed_everything(cfg.train.seed)
    wdir = os.getcwd().split('outputs')[0]
    os.chdir(wdir)
    
    # hyparams
    _dataset = cfg.data.name
    _patience=cfg.train.patience
    _batch_size = cfg.train.bsize
    _max_epochs = cfg.train.max_epochs
    _name_exp = cfg.train.name_exp
    _proj = cfg.train.proj
    _offline = cfg.train.offline
    _measure_on = cfg.train.measure_on
    _mode = 'min' if _measure_on == 'val/FCD' else 'max'
    _check_val_every_n_epoch = cfg.train.check_val_every_n_epoch

    dir_path = '../ckpts/'+_proj+'/'+_name_exp
    create_exp_dir(dir_path)
    
    checkpoint_cb = ModelCheckpoint(monitor=_measure_on, mode=_mode, filename="best", dirpath=dir_path, save_last=False)
    last_ckpt_save = ModelCheckpoint(dirpath=dir_path, filename='save_last', every_n_epochs=None, save_on_train_epoch_end=True)
    earlystopping_cb = EarlyStopping(monitor="train/main-loss", patience=_patience)
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(project=_proj, name=_name_exp, save_dir=dir_path, offline=_offline)
    
    # trainer
    trainer = pl.Trainer(
        log_every_n_steps=1, 
        check_val_every_n_epoch=_check_val_every_n_epoch,
        max_epochs=_max_epochs, 
        callbacks=[checkpoint_cb, earlystopping_cb, lrmonitor_cb, last_ckpt_save], # default monitor on val metric
        logger=logger,
        devices=torch.cuda.device_count(),
        accelerator='gpu',
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
    )
    
    # data loader 
    train_loader, sampler_data, num_fourier_data = load_data(wdir, _dataset, _batch_size)

    # load metric
    metric = load_metric(_dataset)

    # load model
    _n_fourier = cfg.model.n_fourier
    if _n_fourier is None:
        input_dim = num_fourier_data
    else:
        input_dim = min(_n_fourier, num_fourier_data)
        
    diffuser = load_diffuser(cfg, input_dim, metric=metric)
    
    # train from last checkpoint
    _continue, last_ckpt = check_ckpt_exist(dir_path, is_last=True)
    
    if not _continue:
        trainer.fit(diffuser, train_loader, sampler_data) # set val -> train
    else:
        trainer.fit(diffuser, train_loader, sampler_data, ckpt_path=checkpoint_cb.dirpath+f'/{last_ckpt}')
    
if __name__ == '__main__':
    main()