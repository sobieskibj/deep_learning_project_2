import sys
sys.path.append('./')

from pt_dataset.sc_dataset_2 import SpeechCommands, Subset

import wandb

import torch
from torch import optim, nn
import torch.nn.functional as F

import torchmetrics
from torchmetrics.classification import MulticlassAccuracy

import torchaudio

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from utils.utils import get_accuracy, collate_pad_2, get_class_weights, mark_ckpt_as_finished, get_likely_index

class ConvM5(pl.LightningModule):

    def __init__(
            self, 
            transform = None, 
            n_input = 1, 
            n_output = 31, 
            stride = 16, 
            n_channel = 64,
            optimizer_params = {},
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

        self.n_output = n_output

        self.optimizer_params = optimizer_params

        self.register_buffer("train_loss_weight", train_loss_weight)
        self.register_buffer("val_loss_weight", val_loss_weight)

        self.metrics_collection_train = torchmetrics.MetricCollection({
            'tm_micro_acc': MulticlassAccuracy(n_output, average = 'micro'),
            'tm_macro_acc': MulticlassAccuracy(n_output, average = 'macro'),
            'tm_weighted_acc': MulticlassAccuracy(n_output, average = 'weighted')},
            prefix = 'train/')

        self.metrics_collection_val = torchmetrics.MetricCollection({
            'tm_micro_acc': MulticlassAccuracy(n_output, average = 'micro'),
            'tm_macro_acc': MulticlassAccuracy(n_output, average = 'macro'),
            'tm_weighted_acc': MulticlassAccuracy(n_output, average = 'weighted')},
            prefix = 'val/')

        self.balanced_accuracy = MulticlassAccuracy(n_output, average = 'weighted')

        self.training_step_outputs = []
        self.validation_step_outputs = []

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
        output = output.squeeze()
        loss =  F.nll_loss(output, y.long(), weight = self.train_loss_weight)
        metrics = self.metrics_collection_train(output, y)
        metrics["train/loss"] = loss
        metrics['train/acc'] = get_accuracy(output, y) # manual accuracy sanity check
        self.log_dict(metrics, on_epoch = True, prog_bar = True, on_step = False)
        return loss
    
    def on_train_epoch_end(self):
        bal_acc = self.get_balanced_accuracy(self.training_step_outputs)
        self.log("train/tm_balanced_acc", bal_acc, prog_bar = True) # manual balanced accuracy
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        output = output.squeeze()
        val_loss =  F.nll_loss(output, y.long(), weight = self.val_loss_weight)
        metrics = self.metrics_collection_val(output, y)
        metrics['val/loss'] = val_loss
        metrics['val/acc'] = get_accuracy(output, y) # manual accuracy sanity check
        self.log_dict(metrics, on_epoch = True, prog_bar = True)
        self.validation_step_outputs.append((get_likely_index(output), y))
    
    def on_validation_epoch_end(self):
        bal_acc = self.get_balanced_accuracy(self.validation_step_outputs)
        self.log("val/tm_balanced_acc", bal_acc, prog_bar = True) # manual balanced accuracy
        self.validation_step_outputs.clear()
    
    def get_balanced_accuracy(self, outputs):
        all_preds = torch.cat([e[0] for e in outputs])
        all_targets = torch.cat([e[1] for e in outputs])
        bal_acc = self.balanced_accuracy(all_preds, all_targets)
        return bal_acc

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        x, y = batch
        outputs = self(x)
        return y, get_likely_index(outputs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer

def main(config):
    seed_everything(config['general']['seed'], workers = True)

    wandb_logger = WandbLogger(
        config = config,
        **config['logger'])
    
    train_dataset = SpeechCommands(**config['dataset'], subset = Subset.TRAIN | Subset.TEST)
    val_dataset = SpeechCommands(**config['dataset'], subset = Subset.VALID)

    model = ConvM5(
        train_loss_weight = get_class_weights(train_dataset.get_counts()),
        val_loss_weight = get_class_weights(val_dataset.get_counts()),
        **config['model'])

    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        **config['train_dataloader'])
    
    val_dataloader = torch.utils.data.DataLoader(
        dataset = val_dataset,
        **config['val_dataloader'])

    trainer = pl.Trainer(
        logger = wandb_logger,
        **config['trainer'])

    trainer.fit(
        model = model, 
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader,
        **config['trainer_fit'])
    
    wandb.finish()
    project_name = config['logger']['project']
    run_name = config['logger']['name']
    mark_ckpt_as_finished(project_name, run_name)

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
            'group': 'test_3',
            'name': 'some_test'
        },
        'model': {
            'transform': torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 8000),
            'n_channel': 64,
            'optimizer_params': {
                'lr': 1e-2,
                'weight_decay': 1e-4
            },
        },
        'train_dataloader': {
            'batch_size': 128, 
            'shuffle': True,
            'num_workers': 0,
            'collate_fn': collate_pad_2
        },
        'val_dataloader': {
            'batch_size': 128, 
            'shuffle': True,
            'num_workers': 0,
            'collate_fn': collate_pad_2
        },
        'trainer': {
            'callbacks': [
                EarlyStopping(monitor = "val/acc", mode = "max", patience = 10),
                DeviceStatsMonitor(cpu_stats = True),
                ModelCheckpoint(monitor = 'val/acc', save_last = True, mode = 'max')],
            'max_epochs': 1000,
            'profiler': 'simple',
            'fast_dev_run': False,
            'enable_checkpointing': True
        },
        'trainer_fit': {},
    }

    main(config)
