import sys

sys.path.append("./")

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, Resample

import wandb
from models.mel_spec_lstm import MelSpecLSTMModel, MelSpecLSTMSystem
from utils.datasets import SpeechCommands, Subset
from utils.misc import get_last_ckpt_path, make_configs, mark_ckpt_as_finished
from utils.transforms import Pad


def main(config):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(config["general"]["seed"], workers=True)

    train_dataset = SpeechCommands(
        subset=Subset.TRAIN | Subset.TEST, **config["dataset"]
    )
    valid_dataset = SpeechCommands(subset=Subset.VALID, **config["dataset"])

    train_dataloader = DataLoader(train_dataset, shuffle=True, **config["dataloader"])
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, **config["dataloader"])

    model = MelSpecLSTMModel(**config["model"])
    system = MelSpecLSTMSystem(
        model,
        train_loss_weights=train_dataset.get_class_weights(),
        valid_loss_weights=valid_dataset.get_class_weights(),
        class_labels=train_dataset.get_class_labels(),
        **config["system"],
    )

    logger = WandbLogger(config=config, **config["logger"])
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="valid_loss", min_delta=0.01, patience=10),
            ModelCheckpoint(
                dirpath=f"./checkpoints/{config['logger']['group']}/{config['logger']['name']}",
                monitor="valid_loss",
                save_last=True,
            ),
        ],
        **config["trainer"],
    )

    trainer.fit(system, train_dataloader, valid_dataloader, **config["trainer_fit"])

    mark_ckpt_as_finished(config["logger"]["group"], config["logger"]["name"])
    wandb.finish()


if __name__ == "__main__":
    base_config = {
        "general": {
            "seed": 0,
        },
        "dataset": {
            "root": "./data",
            "use_silence": True,
            "only_test_labels": False,
            "transform": nn.Sequential(
                Resample(16000, 8000), Pad(8000), MelSpectrogram(8000, n_mels=120)
            ),
        },
        "dataloader": {
            "batch_size": None,
            "num_workers": 8,
            "pin_memory": True,
        },
        "model": {
            "n_outputs": 31,
        },
        "system": {
            "learning_rate": None,
        },
        "logger": {
            "project": "deep_learning_project_2",
            "group": "exp_4a",
            "name": None,
        },
        "trainer": {
            "max_epochs": 50,
            "num_sanity_val_steps": 0,
            "deterministic": True,
            "profiler": "simple",
        },
        "trainer_fit": {},
    }

    combinations = {
        "bs": {
            "dict_path": ["dataloader", "batch_size"],
            "values": [128, 256, 512],
        },
        "lr": {
            "dict_path": ["system", "learning_rate"],
            "values": [1e-4, 1e-3, 1e-2],
        },
    }

    configs = make_configs(base_config, combinations)

    for config in configs:
        ckpt_path = get_last_ckpt_path(config)

        if ckpt_path is not None:
            if "_final" in ckpt_path.name:
                print(f"Skipping {ckpt_path}")
                continue
            else:
                print(f"Resuming training from {ckpt_path}")
                config["trainer_fit"]["ckpt_path"] = ckpt_path
        else:
            print("Starting run from scratch")
        main(config)
