'''
Produces confusion matrix on train + val
'''
import sys
sys.path.append('./')

from models.conv_m5 import ConvM5
from pt_dataset.sc_dataset_2 import SpeechCommandsKaggle, SpeechCommands, Subset
from utils.utils import collate_pad_2

import torch
from torch.utils.data import DataLoader

import torchaudio

import lightning.pytorch as pl

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def make_conf_matrix(predictions, path_save):
    all_preds = torch.cat([e[0] for e in predictions]).numpy()
    all_targets = torch.cat([e[1].squeeze() for e in predictions]).numpy()
    fig, ax = plt.subplots(figsize = (32, 32))
    conf_mat = ConfusionMatrixDisplay.from_predictions(all_preds, all_targets, normalize = 'true',values_format = '.2f',ax = ax)
    plt.savefig(path_save)
    
if __name__ == '__main__':
    PATH_CKPT = 'deep_learning_project_2/exp_1_bs=128_lr=0.001_n_ch=64_seed=0/checkpoints/epoch=46-step=21432_final.ckpt'
    PATH_DATA = 'data'
    PATH_SAVE_CONF_MAT = 'plots/conf_mats/exp_1_bs=128_lr=0.001_n_ch=64_seed=0.png'
    BATCH_SIZE = 512

    model = ConvM5.load_from_checkpoint(
        PATH_CKPT, 
        transform = torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000))
    
    dataset = SpeechCommands(
        PATH_DATA, 
        Subset.TRAIN | Subset.VALID | Subset.TEST, 
        use_silence = True, 
        aggregate_unknown = False)
    
    dataloader = DataLoader(
        dataset, 
        batch_size = BATCH_SIZE,
        collate_fn = collate_pad_2)
    
    trainer = pl.Trainer(enable_checkpointing = False, logger = False)
    predictions = trainer.predict(model, dataloaders = dataloader)

    make_conf_matrix(predictions, PATH_SAVE_CONF_MAT)