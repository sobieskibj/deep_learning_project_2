import torch.nn.functional as F
from torch import nn


class Pad(nn.Module):
    def __init__(self, n_frames):
        super().__init__()

        self.n_frames = n_frames

    def forward(self, input):
        output = F.pad(input, (0, self.n_frames - input.shape[-1]))
        return output
