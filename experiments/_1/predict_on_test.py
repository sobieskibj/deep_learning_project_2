'''
Produces Kaggle predictions csv
'''
import sys
sys.path.append('./')

from models.conv_m5 import ConvM5
from pt_dataset.sc_dataset_3 import SpeechCommandsKaggle, SpeechCommands, Subset

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
    all_preds = ['silence' if '_' in label_map(idx) else label_map(idx) for idx in all_preds]
    df = pd.DataFrame(index = all_targets, data = all_preds)
    df.index = df.index.rename('fname')
    df.columns = ['label']
    return df

if __name__ == '__main__':
    PATH_CKPT = 'deep_learning_project_2/finetuning_exp_1_bs=128_lr=0.001_n_ch=64_seed=0_n_classes_12_new_weighing____/checkpoints/epoch=53-step=27486.ckpt'
    PATH_SAVE_PREDS = 'results/exp_1_bs=128_lr=0.001_n_ch=64_seed=0_n_classes12_mg_weighing_finetuned.csv'
    PATH_DATA = 'data'
    BATCH_SIZE = 1024

    # read train dataset only to extract labels
    train_dataset = SpeechCommands( 
        PATH_DATA, 
        Subset.TRAIN, 
        use_silence = True, 
        only_test_labels = True)

    # read test for inference
    test_dataset = SpeechCommandsKaggle(PATH_DATA)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size = BATCH_SIZE)
    
    model = ConvM5.load_from_checkpoint(
        PATH_CKPT, 
        transform = torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000),
        train_loss_weight = train_dataset.get_class_weights(),) # debug
        # val_loss_weight = train_dataset.get_class_weights()) # debug
    
    trainer = pl.Trainer(enable_checkpointing = False, logger = False)
    predictions = trainer.predict(model, dataloaders = test_dataloader)

    df = process_test_predictions(predictions, train_dataset.index_to_label)
    df.to_csv(PATH_SAVE_PREDS)
