from dataset_loader import *
from utils import *
# Imports
import os
import random
import torchaudio
import torch
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from pathlib import Path

# Dataset directory from dataset_loader 1 output
LIBRISPEECH_DIR = "" #update according to your env

# Example file
# sample_file = next(Path(LIBRISPEECH_DIR).rglob("*.flac"))
# print("Example file:", sample_file)
 

class ContrastiveSpeechDataset(Dataset):
    def __init__(self, root_dir, clip_duration=3.0, sample_rate=16000, transform=None, max_files=1000): 
        #change max_files=None, or remove the param for full implementation
        self.sample_rate = sample_rate
        self.clip_length = int(clip_duration * sample_rate)
        self.file_list = list(Path(root_dir).rglob("*.flac"))[:max_files]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def _load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if waveform.size(1) < self.clip_length:
            pad_size = self.clip_length - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_size))
        else:
            start = random.randint(0, waveform.size(1) - self.clip_length)
            waveform = waveform[:, start:start + self.clip_length]

        return waveform

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        clean_waveform = self._load_audio(file_path)
        noisy_waveform = add_noise_snr(clean_waveform, snr_db=5.0)

        if self.transform:
            clean_spec = self.transform(clean_waveform)
            noisy_spec = self.transform(noisy_waveform)
        else:
            clean_spec, noisy_spec = clean_waveform, noisy_waveform

        return clean_spec, noisy_spec

mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

dataset = ContrastiveSpeechDataset(
    root_dir=LIBRISPEECH_DIR,
    clip_duration=3.0,
    sample_rate=16000,
    transform=mel_transform,
    max_files=1000
)

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Sanity check
batch = next(iter(loader))
print("Clean batch shape:", batch[0].shape)
print("Noisy batch shape:", batch[1].shape)

