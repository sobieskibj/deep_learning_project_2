from enum import Flag, auto
from pathlib import Path
from typing import Callable

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class Subset(Flag):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class SpeechCommands(Dataset):
    KNOWN_LABELS = {
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    }

    def __init__(
        self,
        root: str | Path,
        subset: Subset,
        use_silence: bool,
        aggregate_unknown: bool,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()

        self.root = Path(root).absolute()
        self.subset = subset
        self.use_silence = use_silence
        self.aggregate_unknown = aggregate_unknown
        self.transform = transform

        valid_paths = self.read_list(self.root, self.root / "train/validation_list.txt")
        test_paths = self.read_list(self.root, self.root / "train/testing_list.txt")

        train_paths = (
            set((self.root / "train/audio").glob("[!_]*/*.wav"))
            - valid_paths
            - test_paths
        )

        if self.use_silence:
            for silence_paths in [
                sorted((self.root / "train/audio/_silence_").glob(f"{path.stem}_*.wav"))
                for path in (self.root / "train/audio/_background_noise_").glob("*.wav")
            ]:
                step = len(silence_paths) // 10
                train_paths.update(silence_paths[: 8 * step])
                valid_paths.update(silence_paths[8 * step : 9 * step])
                test_paths.update(silence_paths[9 * step :])

        paths = set()
        if Subset.TRAIN in subset:
            paths |= train_paths
        if Subset.VALID in subset:
            paths |= valid_paths
        if Subset.TEST in subset:
            paths |= test_paths
        self.paths = sorted(paths)

        all_labels = {path.stem for path in (self.root / "train/audio").glob("[!_]*")}

        labels_idx = {
            label: i for i, label in enumerate(sorted(all_labels & self.KNOWN_LABELS))
        }

        if aggregate_unknown:
            labels_idx |= {
                label: max(labels_idx.values()) + 1
                for label in sorted(all_labels - self.KNOWN_LABELS)
            }
        else:
            labels_idx |= {
                label: max(labels_idx.values()) + 1 + i
                for i, label in enumerate(sorted(all_labels - self.KNOWN_LABELS))
            }

        if use_silence:
            labels_idx |= {"_silence_": max(labels_idx.values()) + 1}

        self.labels_idx = labels_idx

    def get_metadata(self, index: int) -> tuple[Path, str]:
        path = self.paths[index]
        label = path.parts[-2]

        return path, label

    def __getitem__(self, index) -> tuple[Tensor, int]:
        path, label = self.get_metadata(index)

        waveform, _ = torchaudio.load(path)
        if self.transform is not None:
            waveform = self.transform(waveform)

        label_idx = self.labels_idx[label]

        return waveform, label_idx

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def read_list(root: str | Path, path: str | Path) -> set[Path]:
        with open(path) as fp:
            paths = {root / "train/audio" / line.strip() for line in fp.readlines()}

        return paths


class SpeechCommandsKaggle(Dataset):
    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        super().__init__()

        self.root = Path(root).absolute()
        self.transform = transform

        self.paths = sorted((self.root / "test/audio").glob("*.wav"))

    def get_metadata(self, index: int) -> tuple[Path, str]:
        path = self.paths[index]
        filename = path.name

        return path, filename

    def __getitem__(self, index) -> tuple[Tensor, str]:
        path, filename = self.get_metadata(index)

        waveform, _ = torchaudio.load(path)
        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, filename

    def __len__(self) -> int:
        return len(self.paths)
