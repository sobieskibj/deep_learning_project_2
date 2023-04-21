from collections import Counter
from enum import Flag, auto
from pathlib import Path
from copy import copy
from typing import Callable

import torch
import torchaudio


class Subset(Flag):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class SpeechCommands(torch.utils.data.Dataset):
    TEST_LABELS = {
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
        "_silence_",
    }

    def __init__(
        self,
        root: str | Path,
        subset: Subset,
        use_silence: bool,
        only_test_labels: bool,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()

        self.root = Path(root).absolute()
        self.subset = subset
        self.use_silence = use_silence
        self.only_test_labels = only_test_labels
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

        all_labels = set(map(lambda path: path.parts[-2], self.paths))
        known_labels = copy(all_labels)
        if only_test_labels:
            known_labels &= self.TEST_LABELS
        self.all_labels = sorted(all_labels)
        self.known_labels = sorted(known_labels)

    def label_to_index(self, label: str) -> int:
        if label in self.known_labels:
            index = self.known_labels.index(label)
        elif label in self.all_labels:
            index = len(self.known_labels)
        else:
            raise ValueError("wrong label")

        return index

    def index_to_label(self, index: int) -> str:
        if index < len(self.known_labels):
            label = self.known_labels[index]
        elif index == len(self.known_labels):
            label = "unknown"
        else:
            raise ValueError("wrong index")

        return label

    def get_metadata(self, index: int) -> tuple[Path, str]:
        path = self.paths[index]
        label = path.parts[-2]

        return path, label

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.get_metadata(index)

        waveform, _ = torchaudio.load(path)
        if self.transform is not None:
            waveform = self.transform(waveform)

        label_index = torch.tensor(self.label_to_index(label))

        return waveform, label_index

    def __len__(self) -> int:
        return len(self.paths)

    def get_counts(self) -> list[int]:
        counter = Counter(
            self.label_to_index(self.get_metadata(i)[1]) for i in range(len(self))
        )

        return [counter[key] for key in sorted(counter.keys())]

    @staticmethod
    def read_list(root: str | Path, path: str | Path) -> set[Path]:
        with open(path) as fp:
            paths = {root / "train/audio" / line.strip() for line in fp.readlines()}

        return paths


class SpeechCommandsKaggle(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        super().__init__()

        self.root = Path(root).absolute()
        self.transform = transform

        self.paths = sorted((self.root / "test/audio").glob("*.wav"))

    def get_metadata(self, index: int) -> tuple[Path, str]:
        path = self.paths[index]
        filename = path.name

        return path, filename

    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        path, filename = self.get_metadata(index)

        waveform, _ = torchaudio.load(path)
        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, filename

    def __len__(self) -> int:
        return len(self.paths)