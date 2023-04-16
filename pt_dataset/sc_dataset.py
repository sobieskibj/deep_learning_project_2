import re
import torch
import torchaudio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class SCDataset(Dataset):

    def __init__(self, path, type = 'train', transforms = [], use_sliced_bg = True):
        '''
        type - 'train', 'val', 'test'
        use_sliced_bg - if True, slices all original .wav files in 
            _background_noise_ to 1 second long clips
        '''
        super().__init__()
        self.path = Path(path)
        self.type = type
        self.transforms = transforms
        self.use_sliced_bg = use_sliced_bg

        self.sample_rate = 16000
        self.train_size_ratio = 9

        if self.type == 'train':
            paths_train = set(self.path.glob('train/audio/*/*.wav'))
            with (self.path / 'train' / 'validation_list.txt').open() as f:
                paths_val = {self.path / 'train' / 'audio' /  p.split('\n')[0] for p in f.readlines()}
            self.paths_list = list(paths_train.difference(paths_val))
            if self.use_sliced_bg:
                self._slice_background_noises()

        elif self.type == 'val':
            with (self.path / 'train' / 'validation_list.txt').open() as f:
                self.paths_list = [self.path / 'train' / 'audio' /  p.split('\n')[0] for p in f.readlines()]
            if self.use_sliced_bg:
                self._slice_background_noises()

        elif self.type == 'test':
            self.paths_list = list(self.path.glob('test/audio/*.wav'))

        self.labels = np.unique([p.parts[-2] for p in self.paths_list])
        self.labels_encoding = {label: i for i, label in enumerate(self.labels)}
        self.labels_decoding = {i: l for l, i in self.labels_encoding.items()}

    def __len__(self):
        return len(self.paths_list)
        
    def __getitem__(self, index):
        p = self.paths_list[index]
        waveform, _ = torchaudio.load(p)
        if self.transforms:
            waveform = self.transforms(waveform)
        if self.type != 'test': # class label
            label = torch.tensor(self.labels_encoding[p.parts[-2]], dtype = torch.int32)
        else: # filename
            label = p.parts[-1]
        return waveform, label

    def _slice_background_noises(self):
        if self.type == 'train':
            paths = [p for p in self.paths_list if '_background_noise_' in p.parts]
        elif self.type == 'val':
            paths = [p for p in self.path.glob('train/audio/*/*.wav') if '_background_noise_' in p.parts]

        if len(paths) == 6:
            print('Creating sliced background noises')
            for p in paths:
                full_wav, _ = torchaudio.load(p)
                n_subwavs = full_wav.shape[1] // self.sample_rate
                for i in range(n_subwavs):
                    frame_offset = self.sample_rate * i
                    new_wav, _ = torchaudio.load(p, frame_offset, self.sample_rate)
                    new_path = p.with_stem(f'{p.stem}_{i}')
                    torchaudio.save(new_path, new_wav, self.sample_rate)
                    paths.append(new_path)

            paths_new = [p for p in paths if re.search(r'\d+$', p.stem)]
            paths_new.sort()
            for i, p in enumerate(paths_new):
                if self.type == 'train' and i % self.train_size_ratio == 0:
                    self.paths_list.append(p)
                elif self.type == 'val' and i % self.train_size_ratio != 0:
                    self.paths_list.append(p)

        else:
            print('Background noises already sliced')
            paths_new = [p for p in paths if re.search(r'\d+$', p.stem)]
            paths_new.sort()
            for i, p in enumerate(paths_new):
                if self.type == 'train' and i % self.train_size_ratio != 0:
                    self.paths_list.pop(self.paths_list.index(p))
                elif self.type == 'val' and i % self.train_size_ratio == 0:
                    self.paths_list.append(p)            

        if self.type == 'train':
            paths_orig = [p for p in paths if not re.search(r'\d+$', p.stem)]
            for p in paths_orig:
                self.paths_list.pop(self.paths_list.index(p))

    def plot_example_waveform(self):
        waveform, _ = self[0]
        waveform = waveform.numpy()
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / self.sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle('Waveform')
        plt.show(block = False)

    def plot_example_specgram(self):
        waveform, _ = self[0]
        waveform = waveform.numpy()
        num_channels, _ = waveform.shape

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=self.sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle(f'Spectrogram')
        plt.show(block = False)  

if __name__ == '__main__':
    path = 'data'
    type = 'test'
    dataset = SCDataset(path, type)
    print(dataset[0])
    
    
    
    