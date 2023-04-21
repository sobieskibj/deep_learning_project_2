'''
Produces confusion matrix on train + val
'''
import sys
sys.path.append('./')

from models.conv_m5 import ConvM5
from pt_dataset.sc_dataset_2 import SpeechCommandsKaggle, SpeechCommands, Subset
from utils.utils import collate_pad_2, make_conf_matrix

import torch
from torch.utils.data import DataLoader

import torchaudio

import lightning.pytorch as pl

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
    
if __name__ == '__main__':
    PATH_CKPT = '/Users/bartlomiejsobieski/Osobisty/VSC/deep_learning_project_2/deep_learning_project_2/with_weighing/checkpoints/epoch=14-step=6840.ckpt'
    PATH_DATA = 'data'
    PATH_SAVE_CONF_MAT = 'plots/conf_mats/weighing_bs=128_lr=0.001_n_ch=64_seed=0.png'
    BATCH_SIZE = 512

    model = ConvM5.load_from_checkpoint(
        PATH_CKPT,
        transform = torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000))
    
    dataset = SpeechCommands(
        PATH_DATA, 
        Subset.TRAIN | Subset.VALID | Subset.TEST, 
        use_silence = True, 
        aggregate_unknown = True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size = BATCH_SIZE,
        collate_fn = collate_pad_2)
    
    trainer = pl.Trainer(enable_checkpointing = False, logger = False)
    predictions = trainer.predict(model, dataloaders = dataloader)

    make_conf_matrix(predictions, PATH_SAVE_CONF_MAT)