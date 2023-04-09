# deep_learning_project_2

# Dataset info

Test labels:
- *yes*, *no*, *up*, *down*, *left*, *right*, *on*, *off*, *stop*, *go*, 
- *silence* (sounds from `_background_noise_`),
- *unknown* (all other commands from the training set)

# Torchaudio features

*Waveform* - time domain

*Spectrogram* - frequency domain

*MelSpectrogram* - spectrogram with its frequencies converted to mel-scale:

$$m=2595\log _{10}\left(1+{\frac {f}{700}}\right),$$

where $m$ - mel frequency value, $f$ - initial frequency value.

*Waveform -> Spectrogram -> MelSpectrogram* in *Torchaudio*:

`torchaudio.transforms.MelSpectrogram()(waveform)`

