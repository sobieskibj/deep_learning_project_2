# produces final kaggle predictions
import sys
sys.path.append('./')

from models.conv_m5 import ConvM5
from pt_dataset.sc_dataset_2 import SpeechCommandsKaggle, SpeechCommands, Subset

import torch
from torch.utils.data import DataLoader

import torchaudio

import lightning.pytorch as pl

if __name__ == '__main__':
    PATH_CKPT = 'deep_learning_project_2/exp_1_bs=128_lr=0.001_n_ch=64_seed=0/checkpoints/epoch=46-step=21432_final.ckpt'
    PATH_DATA = 'data'
    BATCH_SIZE = 512

    model = ConvM5.load_from_checkpoint(
        PATH_CKPT, 
        transform = torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000))
    
    train_dataset = SpeechCommands(
        PATH_DATA, 
        Subset.TRAIN, 
        use_silence = 
        True, 
        aggregate_unknown = False)
    test_dataset = SpeechCommandsKaggle(PATH_DATA)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size = BATCH_SIZE)
    
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders = test_dataloader)
    import pdb; pdb.set_trace()
