import sys
sys.path.append('./')
from pt_dataset.sc_dataset import SCDataset

from torch import optim, nn, utils
import lightning.pytorch as pl

class VanillaRNN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential()

if __name__ == '__main__':
    PATH_DATA = 'data'
    DATASET_TYPE = 'train'

    dataset = SCDataset(PATH_DATA, DATASET_TYPE)

    

