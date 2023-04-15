import sys
sys.path.append('./')

import torch
from torch.nn.utils.rnn import pack_sequence

def collate_fn_rnn(batch_list):
    batch_seqs = [e[0].reshape(-1, 1)[::1000] for e in batch_list]
    batch_seqs = pack_sequence(batch_seqs, enforce_sorted = False)
    batch_labels = torch.stack([e[1] for e in batch_list])
    return batch_seqs, batch_labels

