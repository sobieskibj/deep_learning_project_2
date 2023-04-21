'''
Produces Kaggle predictions csv
'''
import sys
sys.path.append('./')

from models.conv_m5 import ConvM5
from pt_dataset.sc_dataset_2 import SpeechCommandsKaggle, SpeechCommands, Subset

import torch
from torch.utils.data import DataLoader

import torchaudio

import lightning.pytorch as pl

import pandas as pd

def process_test_predictions(predictions, label_map):
    all_targets = [target for tuple_of_targets in \
                   [batch_preds[0] for batch_preds in predictions] \
                    for target in tuple_of_targets]
    all_preds = torch.cat([e[1] for e in predictions]).squeeze().tolist()
    all_preds = label_map(all_preds)
    df = pd.DataFrame(index = all_targets, data = all_preds)
    df.index = df.index.rename('fname')
    df.columns = ['label']
    return df

if __name__ == '__main__':
    PATH_CKPT = 'deep_learning_project_2/exp_1_bs=128_lr=0.001_n_ch=64_seed=0/checkpoints/epoch=46-step=21432_final.ckpt'
    PATH_DATA = 'data'
    PATH_SAVE_PREDS = 'results/exp_1_bs=128_lr=0.001_n_ch=64_seed=0.csv'
    BATCH_SIZE = 1024

    model = ConvM5.load_from_checkpoint(
        PATH_CKPT, 
        transform = torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000))
    
    # read train dataset only to extract labels
    train_dataset = SpeechCommands( 
        PATH_DATA, 
        Subset.TRAIN, 
        use_silence = True, 
        aggregate_unknown = False)

    # read test for inference
    test_dataset = SpeechCommandsKaggle(PATH_DATA)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size = BATCH_SIZE)
    
    trainer = pl.Trainer(enable_checkpointing = False, logger = False)
    predictions = trainer.predict(model, dataloaders = test_dataloader)

    # label_mapping = train_dataset.index_to_label
    label_dict = {
        'down': 0, 
        'go': 1, 
        'left': 2, 
        'no': 3, 
        'off': 4, 
        'on': 5, 
        'right': 6, 
        'stop': 7, 
        'up': 8, 
        'yes': 9, 
        # 'unknown': 10-29,
        'silence': 30}

    rev_l_dict = {v: k for k, v in label_dict.items()}

    def label_mapping(predicted_labels, rev_l_dict):
        for i in range(len(predicted_labels)):
            label = predicted_labels[i]
            if label in rev_l_dict.keys():
                predicted_labels[i] = rev_l_dict[label]
            else:
                predicted_labels[i] = 'unknown'
        return predicted_labels

    df = process_test_predictions(
        predictions,
        label_map = lambda x: label_mapping(x, rev_l_dict = rev_l_dict))
    
    df.to_csv(PATH_SAVE_PREDS)
