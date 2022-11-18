from dataclasses import dataclass, field

@dataclass
class SpectConfig:
    sample_rate: int = 16000  # The sample rate for the data/model features
    hop_length: int = 512
    n_fft: int = 1024
    n_mels: int = 128