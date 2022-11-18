import librosa
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import re
import matplotlib.pyplot as plt


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
	fig, axs = plt.subplots(1, 1)
	axs.set_title(title or "Spectrogram (db)")
	axs.set_ylabel(ylabel)
	axs.set_xlabel("frame")
	im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
	fig.colorbar(im, ax=axs)
	plt.show(block=False)
	
def plot_waveform(waveform, sr, title="Waveform"):
	waveform = waveform.numpy()

	num_channels, num_frames = waveform.shape
	time_axis = torch.arange(0, num_frames) / sr

	figure, axes = plt.subplots(num_channels, 1)
	axes.plot(time_axis, waveform[0], linewidth=1)
	axes.grid(True)
	figure.suptitle(title)
	plt.show(block=False)

def load_audio(path):
	waveform, sample_rate = torchaudio.load(file_path)
	return waveform.numpy()

class TextProcess:

	def __init__(self):
		self.char_to_int = {
		"'": 0,
		"<SPACE>": 1,
		"a": 2,
		"b": 3,
		"c": 4,
		"d": 5,
		"e": 6,
		"f": 7,
		"g": 8,
		"h": 9,
		"i": 10,
		"j": 11,
		"k": 12,
		"l": 13,
		"m": 14,
		"n": 15,
		"o": 16,
		"p": 17,
		"q": 18,
		"r": 19,
		"s": 20,
		"t": 21,
		"u": 22,
		"v": 23,
		"w": 24,
		"x": 25,
		"y": 26,
		"z": 27,
		}
		self.int_to_char = {value: key for key, value in self.char_to_int.items()}
		self.int_to_char[1] = " "

	def text_to_int(self, text):
		seq = []
		text = re.sub(r'[^\w\s]','',text)
		for char in text.lower():
			if char == " ":
				out = self.char_to_int["<SPACE>"]
			else:
				out = self.char_to_int[char]
			seq.append(out)

		return seq

	def int_to_text(self, seq):
		text = []
		for i in seq:
			text.append(self.int_to_char[i])
		return "".join(text).replace("<SPACE>", " ")

class RandomNoise(nn.Module):

	def __init__(self, noise_percentage_factor=0.1, rate=0.4):
		super(RandomNoise, self).__init__()

		self.rate = rate
		self.noise_percentage_factor =  noise_percentage_factor

		
	def forward(self, x):
		probability = torch.rand(1, 1).item()
		if self.rate > probability:
			noise = np.random.normal(0, x.std(), x.size)
			augmented_signal = x + noise * self.noise_percentage_factor
			return augmented_signal
		return x

class SpecAugment(nn.Module):

	def __init__(self, rate=0.4, freq_mask=15, time_mask=35):
		super(SpecAugment, self).__init__()

		self.rate = rate

		self.specaug = nn.Sequential(
			torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
			torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
		)
		
	def forward(self, x):
		probability = torch.rand(1, 1).item()
		if self.rate > probability:
			return  self.specaug(x)
		return x

class MelSpectrogram(nn.Module):

	def __init__(self, audio_conf, normalized: bool = False):
		super(MelSpectrogram, self).__init__()

		self.transform = torchaudio.transforms.MelSpectrogram(
											sample_rate=audio_conf.sample_rate,
											n_fft=audio_conf.n_fft,
											hop_length=audio_conf.hop_length,
											n_mels=audio_conf.n_mels,
											normalized=normalized)

	def forward(self, x):
		x = self.transform(x)
		x = np.log(x + 1e-14) 
		return x

