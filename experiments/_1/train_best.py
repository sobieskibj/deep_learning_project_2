# optionally finetunes on train + val
import sys
sys.path.append('./')

from pt_dataset.sc_dataset_3 import SpeechCommands, Subset
from models.conv_m5 import ConvM5

import wandb

import torch

import torchaudio

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from utils.utils import collate_pad_2

if __name__ == '__main__':
    # use best config from grid to train it from scratch on train + val
    config = {
        'general':{
            'seed': 0
        },
        'dataset': {
            'root': 'data',
            'use_silence': True,
            'only_test_labels': True
        },
        'logger': {
            'project': 'deep_learning_project_2',
            'group': 'exp_1',
            'name': 'finetuning_exp_1_bs=128_lr=0.001_n_ch=64_seed=0_n_classes_12_new_weighing____',
            'version': 'finetuning_exp_1_bs=128_lr=0.001_n_ch=64_seed=0_n_classes_12_new_weighing____'
        },
        'model': {
            'transform': torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000),
            'n_channel': 64,
            'optimizer_params': {
                'lr': 1e-2,
                'weight_decay': 1e-4
            },
        },
        'train_dataloader': {
            'batch_size': 128, 
            'shuffle': True,
            'num_workers': 0,
            'collate_fn': collate_pad_2
        },
        'trainer': {
            'callbacks': [
                EarlyStopping(monitor = "train/acc", mode = "max", patience = 10), # no validation, tracking only train metrics
                DeviceStatsMonitor(cpu_stats = True),
                ModelCheckpoint(monitor = 'train/acc', save_last = True, mode = 'max')],
            'max_epochs': 1000,
            'profiler': 'simple',
            'fast_dev_run': False,
            'enable_checkpointing': True
        },
        'trainer_fit': {},
    }

    seed_everything(config['general']['seed'], workers = True)

    wandb_logger = WandbLogger(
        config = config,
        **config['logger'])
    
    train_dataset = SpeechCommands(**config['dataset'], subset = Subset.TRAIN | Subset.VALID | Subset.TEST)

    model = ConvM5(
        train_loss_weight = train_dataset.get_class_weights(),
        **config['model'])

    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        **config['train_dataloader'])

    trainer = pl.Trainer(
        logger = wandb_logger,
        **config['trainer'])

    trainer.fit(
        model = model, 
        train_dataloaders = train_dataloader,
        **config['trainer_fit'])
    
    wandb.finish()
