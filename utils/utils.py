import sys
sys.path.append('./')

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence

def collate_pad(batch_list):
    batch_seqs = [e[0].reshape(-1, 1) for e in batch_list]
    batch_seqs = [torch.randn(16000, 1)] + batch_seqs
    batch_seqs = pad_sequence(batch_seqs)[:, 1:]
    batch_filenames = torch.stack([e[1].long() for e in batch_list])
    return batch_seqs, batch_filenames

def collate_pad_and_pack(batch_list):
    batch_seqs = [e[0].reshape(-1, 1)[::1000] for e in batch_list] # ::1000 for quick debugging
    batch_seqs = pack_sequence(batch_seqs, enforce_sorted = False)
    batch_labels = torch.stack([e[1].long() for e in batch_list])
    return batch_seqs, batch_labels

def collate_pad_and_pack_test(batch_list):
    batch_seqs = [e[0].reshape(-1, 1)[::1000] for e in batch_list]
    batch_seqs = pack_sequence(batch_seqs, enforce_sorted = False)
    batch_filenames = [e[1].long() for e in batch_list]
    return batch_seqs, batch_filenames
