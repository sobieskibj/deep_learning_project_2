import sys
sys.path.append('./')

import torch
import torch.nn as nn
import lightning.pytorch as pl
from utils.transforms import Pad
from torchaudio.transforms import MelSpectrogram, Resample
from models.m5 import *
from models.m5_lstm import *
from models.mel_spec import *
from models.mel_spec_lstm import *
from utils.datasets import *
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

def process_test_predictions(predictions, label_map, labels_core):
    all_targets = [target for tuple_of_targets in \
                   [batch_preds[1] for batch_preds in predictions] \
                    for target in tuple_of_targets]
    all_preds = torch.cat([e[0] for e in predictions]).squeeze().tolist()
    all_preds = [label_map(idx) if label_map(idx) in labels_core \
                    or label_map(idx) == 'silence' else 'unknown' for idx in all_preds ]
    df = pd.DataFrame(index = all_targets, data = all_preds)
    df.index = df.index.rename('fname')
    df.columns = ['label']
    return df

if __name__ == '__main__':
    
    PATH_CKPTS = 'checkpoints'
    path = Path(PATH_CKPTS)
    all_ckpts_paths = list(path.glob('**/epoch*.ckpt'))
    for i, p in enumerate(all_ckpts_paths):
        exp_name = p.parts[1]
        run_params = p.parts[2]
        PATH_CKPT = p
        PATH_DATA = 'data'
        PATH_SAVE_PREDS = f'results/{exp_name}_{run_params}.csv'

        if Path(PATH_SAVE_PREDS).exists():
            print('Predictions are already saved, continuing')
            continue

        BATCH_SIZE = 4096
        print(f'{i}:')
        print('PATH_CKPT:', p)
        print('PATH_SAVE_PREDS:', PATH_SAVE_PREDS, '\n')
        LABELS_CORE = [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ]

        train_dataset = SpeechCommands(
            PATH_DATA,
            Subset.TRAIN | Subset.VALID | Subset.TEST,
            use_silence = True,
            only_test_labels = False)

        if '1' in exp_name:
            model_type = M5Model
            system_type = M5System

        elif '2' in exp_name:
            model_type = M5LSTMModel
            system_type = M5LSTMSystem

        elif '3' in exp_name:
            model_type = MelSpecModel
            system_type = MelSpecSystem

        elif '4' in exp_name:
            model_type = MelSpecLSTMModel
            system_type = MelSpecLSTMSystem

        if '3' in exp_name or '4' in exp_name:
            transform = nn.Sequential(
                Resample(16000, 8000), Pad(8000), MelSpectrogram(8000, n_mels=120)
            )
        else:
            transform = nn.Sequential(Resample(16000, 8000), Pad(8000))

        if 'a' in exp_name:
            n_outputs = 31
        else:
            n_outputs = 12

        test_dataset = SpeechCommandsKaggle(
            PATH_DATA,
            transform = transform)
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size = BATCH_SIZE)

        model = torch.load(PATH_CKPT, map_location = 'cpu')
        try:
            system = system_type.load_from_checkpoint(
                PATH_CKPT,
                model = model_type(n_outputs),
                train_loss_weights = 1, 
                valid_loss_weights = 1, 
                learning_rate = 1e-4,
                class_labels = ['label'])
        except:
            system = system_type.load_from_checkpoint(
                PATH_CKPT,
                model = model_type(n_outputs),
                train_loss_weights = 1, 
                valid_loss_weights = 1, 
                learning_rate = 1e-4,
                weight_decay = 0.0001,
                class_labels = ['label'])
        
        trainer = pl.Trainer()
        preds = trainer.predict(system, test_dataloader)
        preds_df = process_test_predictions(preds, train_dataset.index_to_label, LABELS_CORE)
        preds_df.to_csv(PATH_SAVE_PREDS)