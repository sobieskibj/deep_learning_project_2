import sys
sys.path.append('./')

from pt_dataset.sc_dataset import SCDataset
from utils.utils import collate_pad_and_pack, collate_pad

import torch
from torch import optim, nn
import torch.nn.functional as F

import torchaudio

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

from torchmetrics import classification

class WaveformToSpecgram(nn.Module):

    def __init__(
        self,
        input_freq = 16000,
        resample_freq = 8000,
        n_fft = 1024,
        n_mel = 256,
        stretch_factor = 0.8,
    ):
        super().__init__()
        self.pad = F.pad
        self.resample = torchaudio.transforms.Resample(orig_freq = input_freq, new_freq = resample_freq)
        self.spec = torchaudio.transforms.Spectrogram(n_fft = n_fft, power = 2)
        self.spec_aug = torch.nn.Sequential(
            torchaudio.transforms.TimeStretch(stretch_factor, fixed_rate = True),
            torchaudio.transforms.FrequencyMasking(freq_mask_param = 80),
            torchaudio.transforms.TimeMasking(time_mask_param = 80),)
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels = n_mel, sample_rate = resample_freq, n_stft = n_fft // 2 + 1)

    def forward(self, waveform):
        # Resample the input
        waveform = self.pad(waveform, (0, 16000 - waveform.shape[1]))
        # print('Waveform shape after pad:', waveform.shape)
        resampled = self.resample(waveform)
        # print('Resampled shape:', resampled.shape)
        # Convert to power spectrogram
        spec = self.spec(resampled)
        # print('Specgram shape:', spec.shape)
        # Apply SpecAugment
        spec = self.spec_aug(spec)
        # print('Augmented specgram shape:', spec.shape)
        # Convert to mel-scale
        mel = self.mel_scale(spec)
        # print('Mel-scaled specgram shape:', mel.shape)
        mel = torch.squeeze(mel)
        return mel

class ConvRNN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels = 16, 
                out_channels = 256, 
                kernel_size = 3, 
                padding = 1),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = 256, 
                out_channels = 512, 
                kernel_size = 3, 
                padding = 1),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = 512, 
                out_channels = 256, 
                kernel_size = 3, 
                padding = 1),
            nn.ReLU())
        self.rnn = nn.RNN(
                input_size = 256,
                hidden_size = 256,
                num_layers = 8,
                batch_first = True)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 31))
    
        self.calc_acc = classification.MulticlassAccuracy(num_classes = 31)

    def forward(self, x):
        # print('Shape after preprocessing:', x.shape)
        x = torch.transpose(x, -1, -2) # batch x freqs x time -> batch x time x freqs
        # print('Shape of cnn input:', x.shape)
        x = self.cnn(x)
        # print('Shape of cnn output:', x.shape)
        _, h_n = self.rnn(x)
        # print('Shape of rnn output:', h_n.shape)
        return self.output(h_n[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.functional.cross_entropy(output, y)
        self.log("train_loss", loss, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        val_loss = nn.functional.cross_entropy(output, y)
        self.log("val_loss", val_loss, batch_size = 256, prog_bar = True)
        self.log("val_acc", self.calc_acc(output, y), batch_size = 256, prog_bar = True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer

def get_model_id():
    return f'test_conv_rnn'

if __name__ == '__main__':
    PATH_DATA = 'data'
    NAME = 'test'
    SEED = 0

    seed_everything(SEED, workers = True)

    wandb_logger = WandbLogger(
        project = 'deep_learning_project_2',
        name = NAME,
        id = get_model_id(),
        group = 'test')

    model = ConvRNN()
    transforms = WaveformToSpecgram()
    train_dataset = SCDataset(PATH_DATA, 'train', transforms = transforms)
    val_dataset = SCDataset(PATH_DATA, 'val', transforms = transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 256, 
        shuffle = True, 
        num_workers = 8)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = 256, 
        shuffle = False, 
        num_workers = 8)

    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor = "val_acc", mode = "max", patience = 5),
            DeviceStatsMonitor(cpu_stats = True)],
        max_epochs = 10,
        profiler = "simple",
        logger = wandb_logger,
        fast_dev_run = False)

    trainer.fit(
        model = model, 
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader)