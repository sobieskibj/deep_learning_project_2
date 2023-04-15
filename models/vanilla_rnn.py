import sys
sys.path.append('./')

from pt_dataset.sc_dataset import SCDataset
from utils.utils import collate_fn_rnn

import torch
from torch import optim, nn

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

import torchmetrics
from torchmetrics import classification

class VanillaRNN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
                input_size = 1,
                hidden_size = 128,
                num_layers = 20)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 31))

        self.calc_acc = classification.MulticlassAccuracy(num_classes = 31, validate_args = False)
    
    def forward(self, x):
        _, h_n = self.rnn(x)
        ## different way ##
        # out, h_n = self.rnn(x) + 
        # torch.nn.utils.rnn.unpack_sequence(out)
        # for a batch of all hidden states for each sequence
        return self.output(h_n[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.functional.cross_entropy(output, y)
        self.log("train_loss", loss, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.int32)
        output = self(x)
        val_loss = nn.functional.cross_entropy(output, y)
        self.log("val_loss", val_loss, batch_size = 128, prog_bar = True)
        self.log("val_acc", self.calc_acc(output, y), batch_size = 128, prog_bar = True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer

if __name__ == '__main__':
    PATH_DATA = 'data'

    wandb_logger = WandbLogger(
        project = 'deep_learning_project_2',
        name = 'test',
        group = 'test',
        save_dir = '.')

    model = VanillaRNN()

    train_dataset = SCDataset(PATH_DATA, 'train')
    val_dataset = SCDataset(PATH_DATA, 'val')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 128, 
        shuffle = True, 
        collate_fn = collate_fn_rnn,
        num_workers = 8)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = 128, 
        shuffle = False, 
        collate_fn = collate_fn_rnn,
        num_workers = 8)

    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor = "val_acc", mode = "max"),
            DeviceStatsMonitor(cpu_stats = True)],
        max_epochs = 10,
        profiler = "simple",
        logger = wandb_logger)
    
    trainer.fit(
        model = model, 
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader)


    

