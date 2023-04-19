from pathlib import Path

import torchaudio

DURATION = 1
SAMPLE_RATE = 16000

IN_PATH = Path("./data/train/audio/_background_noise_")
OUT_PATH = Path("./data/train/audio/_silence_")

if __name__ == "__main__":
    OUT_PATH.mkdir(exist_ok=True)

    for path in IN_PATH.glob("*.wav"):
        n_frames = torchaudio.info(path).num_frames

        for i in range(n_frames // (DURATION * SAMPLE_RATE)):
            waveform, _ = torchaudio.load(path, i * SAMPLE_RATE, DURATION * SAMPLE_RATE)
            torchaudio.save(OUT_PATH / f"{path.stem}_{i}.wav", waveform, SAMPLE_RATE)