import sys
sys.path.append('./')

from models.conv_m5 import main

import torchaudio
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint

from utils.utils import collate_pad_2, make_configs, get_last_ckpt_path

if __name__ == '__main__':
    
    base_config = {
        'general':{
            'seed': 0
        },
        'dataset': {
            'root': 'data',
            'use_silence': True,
            'aggregate_unknown': False
        },
        'logger': {
            'project': 'deep_learning_project_2',
            'group': 'exp_1',
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
            'num_workers': 2,
            'collate_fn': collate_pad_2
        },
        'val_dataloader': {
            'batch_size': 128, 
            'shuffle': False,
            'num_workers': 2,
            'collate_fn': collate_pad_2
        },
        'trainer': {
            'callbacks': [
                EarlyStopping(monitor = "val_acc", mode = "max", patience = 10),
                DeviceStatsMonitor(cpu_stats = True),
                ModelCheckpoint(monitor = 'val_acc', save_last = True, mode = 'max')],
            'max_epochs': 1000,
            'profiler': 'simple',
            'fast_dev_run': False,
            'enable_checkpointing': True
        },
        'trainer_fit': {},
    }

    combinations = {
        'bs': {
            'dict_path': ['train_dataloader', 'batch_size'],
            'values': [128, 256, 512]
        },
        'lr': {
            'dict_path': ['model', 'optimizer_params', 'lr'],
            'values': [1e-2, 1e-3, 1e-4]        
        },
        'n_ch': {
            'dict_path': ['model', 'n_channel'],
            'values': [64] # single values are provided so that they are included in name and version
        },
        'seed': {
            'dict_path': ['general', 'seed'],
            'values': [0]
        }
    }

    configs = make_configs(base_config, combinations)
    
    for config in configs:
        ckpt_path = get_last_ckpt_path(config)
        if ckpt_path is not None:
            if '_final' in ckpt_path.name:
                print(f'Skipping {ckpt_path}')
                continue
            else:
                print(f'Resuming training from {ckpt_path}')
                config['trainer_fit']['ckpt_path'] = ckpt_path # will resume training from the saved state
        else:
            import pdb; pdb.set_trace()
            print('Starting run from scratch')
        main(config)
