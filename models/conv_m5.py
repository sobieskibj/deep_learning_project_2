import sys
sys.path.append('./')

from pt_dataset.sc_dataset_2 import SpeechCommands, Subset

import torch
from torch import optim, nn
import torch.nn.functional as F

import torchaudio

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

from utils.utils import get_accuracy, collate_pad_2, get_class_weights

class ConvM5(pl.LightningModule):

    def __init__(
            self, 
            transform = None, 
            n_input = 1, 
            n_output = 31, 
            stride = 16, 
            n_channel = 32,
            optimizer_params = {},
            train_loss_params = {},
            val_loss_params = {},
            train_loss_weight = None,
            val_loss_weight = None):
        super().__init__()
        self.transform = transform
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size = 80, stride = stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size = 3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size = 3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size = 3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

        self.optimizer_params = optimizer_params
        self.train_loss_params = train_loss_params
        self.val_loss_params = val_loss_params

        if train_loss_weight is not None:
            self.train_loss_weight = torch.tensor(train_loss_weight, device = self.device)
        if val_loss_weight is not None:
            self.val_loss_weight = torch.tensor(val_loss_weight, device = self.device)

    def forward(self, x):
        x = self.transform(x)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        weight = torch.tensor(self.train_loss_params, device = self.device)
        loss =  F.nll_loss(output.squeeze(), y.long(), weight = weight)
        self.log("train_loss", loss, on_epoch = True, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        weight = torch.tensor(self.val_loss_params, device = self.device)
        val_loss =  F.nll_loss(output.squeeze(), y.long(), weight = weight)
        val_acc = get_accuracy(output, y)
        self.log("val_loss", val_loss, prog_bar = True)
        self.log("val_acc", val_acc, on_epoch = True, prog_bar = True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer

def get_class_weights(counts):
    return [1/e for e in counts.values()]

def main(config):
    seed_everything(config['general']['seed'], workers = True)

    wandb_logger = WandbLogger(
        config = config,
        **config['logger'])
    
    train_dataset = SpeechCommands(**config['dataset'], subset = Subset.TRAIN | Subset.TEST)
    val_dataset = SpeechCommands(**config['dataset'], subset = Subset.VALID)

    # train_loss_weights = get_class_weights(train_dataset.get_counts())
    # val_loss_weights = get_class_weights(val_dataset.get_counts())

    # train_loss_params = {'weight': train_loss_weights}
    # val_loss_params = {'weight': val_loss_weights}

    model = ConvM5(
        train_loss_params = get_class_weights(train_dataset.get_counts()),
        val_loss_params = get_class_weights(val_dataset.get_counts()),
        **config['model'])

    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        **config['train_dataloader'])
    
    val_dataloader = torch.utils.data.DataLoader(
        dataset = val_dataset,
        **config['val_dataloader'])

    trainer = pl.Trainer(
        logger = wandb_logger,
        **config['trainer']
    )

    trainer.fit(
        model = model, 
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader)

if __name__ == '__main__':

    config = {
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
            'group': 'test',
            'name': 'dasdasdas'
        },
        'model': {
            'transform': torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000),
            'n_channel': 32,
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

    main(config)
