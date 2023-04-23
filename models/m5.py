import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn, optim
from torchmetrics.functional.classification import multiclass_accuracy


class M5Model(nn.Module):
    def __init__(self, n_outputs, channels_base=32):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv1d(1, channels_base, kernel_size=80, stride=16),
            nn.BatchNorm1d(channels_base),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(channels_base, channels_base, kernel_size=3),
            nn.BatchNorm1d(channels_base),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(channels_base, 2 * channels_base, kernel_size=3),
            nn.BatchNorm1d(2 * channels_base),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(2 * channels_base, 2 * channels_base, kernel_size=3),
            nn.BatchNorm1d(2 * channels_base),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.classifier = nn.Linear(2 * channels_base, n_outputs)

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        outputs = F.avg_pool1d(outputs, outputs.shape[-1])
        outputs = torch.squeeze(outputs)
        outputs = self.classifier(outputs)
        outputs = F.log_softmax(outputs, dim=-1)

        return outputs


class M5System(pl.LightningModule):
    def __init__(
        self,
        model,
        train_loss_weights,
        valid_loss_weights,
        learning_rate,
        weight_decay,
        class_labels,
    ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_labels = class_labels

        self.register_buffer("train_loss_weights", torch.tensor(train_loss_weights))
        self.register_buffer("valid_loss_weights", torch.tensor(valid_loss_weights))

        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        loss = F.nll_loss(outputs, targets, weight=self.train_loss_weights)
        accuracy = multiclass_accuracy(
            outputs, targets, num_classes=len(self.class_labels), average="micro"
        )

        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        loss = F.nll_loss(outputs, targets, weight=self.valid_loss_weights)
        accuracy = multiclass_accuracy(
            outputs, targets, num_classes=len(self.class_labels), average="micro"
        )

        self.log_dict(
            {"valid_loss": loss, "valid_accuracy": accuracy},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        preds = torch.argmax(outputs, dim=-1)
        self.validation_step_outputs.append((preds, targets))

    def on_validation_epoch_end(self):
        preds = (
            torch.cat([output[0] for output in self.validation_step_outputs])
            .cpu()
            .numpy()
        )
        targets = (
            torch.cat([output[1] for output in self.validation_step_outputs])
            .cpu()
            .numpy()
        )

        self.log_confusion_matrix(preds, targets)
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        outputs = self.model(inputs)

        preds = torch.argmax(outputs, dim=-1)

        return preds, targets

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def log_confusion_matrix(self, preds, targets):
        fig, ax = plt.subplots(figsize=(11, 11))

        ConfusionMatrixDisplay.from_predictions(
            preds,
            targets,
            normalize="true",
            xticks_rotation="vertical",
            values_format=".2f",
            display_labels=self.class_labels,
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"Epoch {self.current_epoch}")
        plt.tight_layout()

        self.logger.experiment.log({"valid_confusion_matrix": fig})

        plt.close()


if __name__ == "__main__":
    import sys

    sys.path.append("./")
    from utils.datasets import SpeechCommands, Subset
    from torch.utils.data import DataLoader
    from utils.transforms import Pad
    from torchaudio.transforms import Resample

    dataset = SpeechCommands(
        "./data",
        Subset.TRAIN,
        False,
        False,
        nn.Sequential(Resample(16000, 8000), Pad(8000)),
    )
    dataloader = DataLoader(dataset, 2)
    model = M5Model(30)

    x = next(iter(dataloader))[0]
    print(x.shape)

    y = model(x)
    print(y.shape)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
