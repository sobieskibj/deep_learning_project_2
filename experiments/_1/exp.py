import sys
sys.path.append('./')

from models.conv_m5 import main

import torchaudio
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor

from utils.utils import collate_pad_2, make_configs

if __name__ == '__main__':
    
    base_config = {
        'general':{
            'seed': 0
        },
        'dataset': {
            'path': 'data'
        },
        'logger': {
            'project': 'deep_learning_project_2',
            'group': 'test',
        },
        'model': {
            'transform': torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000),
            'n_channel': 64,
            'optimizer_params': {
                'lr': 1e-2,
                'weight_decay': 1e-4
            }
        },
        'train_dataloader': {
            'batch_size': 128, 
            'shuffle': True,
            'num_workers': 8,
            'collate_fn': collate_pad_2,
            'pin_memory': True
        },
        'val_dataloader': {
            'batch_size': 128, 
            'shuffle': False,
            'num_workers': 8,
            'collate_fn': collate_pad_2,
            'pin_memory': True
        },
        'trainer': {
            'callbacks': [
                EarlyStopping(monitor = "val_acc", mode = "max", patience = 10),
                DeviceStatsMonitor(cpu_stats = True)],
            'max_epochs': 1000,
            'profiler': 'simple',
            'fast_dev_run': False,
            'enable_checkpointing': True
        }
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
            'values': [32] # single values are provided so that they are included in name and version
        },
        'seed': {
            'dict_path': ['general', 'seed'],
            'values': [0]
        }
    }

    configs = make_configs(base_config, combinations)
    
    for config in configs:
        main(config)
