import re
import torch
import torchaudio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class SCDataset(Dataset):

    def __init__(self, path, type = 'train', use_sliced_bg = True):
        '''
        type - 'train', 'val', 'test'
        use_sliced_bg - if True, slices all original .wav files in 
            _background_noise_ to 1 second long clips
        '''
        super().__init__()
        self.path = Path(path)
        self.type = type
        self.sample_rate = 16000
        self.use_sliced_bg = use_sliced_bg

        if type == 'train':
            self.paths_list = list(self.path.glob('train/audio/*/*.wav'))
            if self.use_sliced_bg:
                self._slice_background_noises()

        elif type == 'val':
            with (self.path / 'train' / 'validation_list.txt').open() as f:
                self.paths_list = [self.path / 'train' / 'audio' /  p.split('\n')[0] for p in f.readlines()]
            if self.use_sliced_bg:
                self._slice_background_noises()

        elif type == 'test':
            self.paths_list = list(self.path.glob('test/audio/*.wav'))
        
        self.labels = np.unique([p.parts[-2] for p in self.paths_list])
        self.labels_encoding = {label: i for i, label in enumerate(self.labels)}
        self.labels_encoding['_background_noise_'] = 'silence'
        self.rev_labels_encoding = {i: l for l, i in self.labels_encoding.items()}

    def __len__(self):
        return len(self.paths_list)
        
    def __getitem__(self, index):
        p = self.paths_list[index]
        waveform, _ = torchaudio.load(p)
        label = torch.tensor(self.labels_encoding[p.parts[-2]])
        return waveform, label

    def _slice_background_noises(self):
        paths = [p for p in self.paths_list if '_background_noise_' in p.parts]
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
                    self.paths_list.append(new_path)
                self.paths_list.pop(self.paths_list.index(p)) # pop original files
        else:
            print('Background noises already sliced')
            # pop original files from the paths list
            paths_to_pop = [p for p in paths if not re.search(r'\d+$', p.stem)]
            for p in paths_to_pop:
                self.paths_list.pop(self.paths_list.index(p))

    def plot_example_waveform(self):
        waveform, _ = self[0]
        waveform = waveform.numpy()
        sample_rate = 16000
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

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
        sample_rate = 16000

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle(f'Spectrogram')
        plt.show(block = False)  

if __name__ == '__main__':
    path = 'data'
    type = 'val'
    dataset = SCDataset(path, type)
    print(dataset[-1])
    
    
    
    