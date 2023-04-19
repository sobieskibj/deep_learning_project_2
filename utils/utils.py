import sys
sys.path.append('./')

import copy
import torch
import itertools
import numpy as np
from torch.nn.utils.rnn import pack_sequence, pad_sequence

# configs

def make_configs(base_config, combinations):
    product_input = [p['values'] for p in combinations.values()]
    product = [p for p in itertools.product(*product_input)]
    configs = []
    print(f'Running {len(product)} configurations:')
    for n, p in enumerate(product): # for each combination
        config = copy.deepcopy(base_config)
        str_reprs = []
        for i, (param_name, parameter) in enumerate(combinations.items()): # for each parameter in config
            pointer = config
            for name in parameter['dict_path'][:-1]: # finish when pointing at second-last element from path
                pointer = pointer[name]
            pointer[parameter['dict_path'][-1]] = p[i] # set desired value
            str_reprs.append(f"_{param_name}={p[i]}")
        id = f"{config['logger']['group']}" + ''.join(str_reprs)
        config['logger']['name'] = id
        config['logger']['version'] = id
        print(f'{n}. {id}')
        configs.append(config)
    return configs

# collate fns

def collate_pad_2(batch):
    
    def _pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first = True, padding_value=0.)
        batch = batch.permute(0, 2, 1)
        return batch
    
    tensors, targets = [], []
    for waveform, label in batch:
        tensors += [waveform]
        targets += [label]

    tensors = _pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets

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

# metrics

def get_accuracy(output, target):

    def number_of_correct(pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    def get_likely_index(tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim = -1)
    
    pred = get_likely_index(output)
    correct = number_of_correct(pred, target)
    accuracy = correct / output.shape[0]
    return accuracy

# loss weighing

def get_class_weights(counts):
    pass