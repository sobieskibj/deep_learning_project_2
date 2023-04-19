import sys
sys.path.append('./')

from pt_dataset.sc_dataset import SCDataset

import torch
from torch import optim, nn
import torch.nn.functional as F

import torchaudio

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

from utils.utils import get_accuracy, collate_pad_2

class ConvM5(pl.LightningModule):

    def __init__(
            self, 
            transform = None, 
            n_input = 1, 
            n_output = 31, 
            stride = 16, 
            n_channel = 32,
            optimizer_params = {},
            loss_params = {}):
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
        self.loss_params = loss_params

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
        loss =  F.nll_loss(output.squeeze(), y.long(), **self.loss_params)
        self.log("train_loss", loss, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        val_loss =  F.nll_loss(output.squeeze(), y.long())
        val_acc = get_accuracy(output, y)
        self.log("val_loss", val_loss, prog_bar = True)
        self.log("val_acc", val_acc, on_epoch = True, prog_bar = True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer

def main(config):
    seed_everything(config['general']['seed'], workers = True)

    wandb_logger = WandbLogger(
        config = config,
        **config['logger'])

    train_dataset = SCDataset(type = 'train', **config['dataset'])
    val_dataset = SCDataset(type = 'val', **config['dataset'])
    model = ConvM5(**config['model'])

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
            'path': 'data'
        },
        'logger': {
            'project': 'deep_learning_project_2',
            'name': 'test',
            'group': 'test',
        },
        'model': {
            'transform': torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000),
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
                EarlyStopping(monitor = "val_acc", mode = "max", patience = 15),
                DeviceStatsMonitor(cpu_stats = True)],
            'max_epochs': 100,
            'profiler': 'simple',
            'fast_dev_run': False,
            'enable_checkpointing': False
        }
    }

    main(config)
