import torch 
import torchaudio
import torch.nn as nn
import pandas as pd

from utils import TextProcess, MelSpectrogram, SpecAugment, RandomNoise
from configs.train_config import SpectConfig 

class Dataset(torch.utils.data.Dataset):

	def __init__(self, json_path, log=False):
		self.log = log
		self.text_process = TextProcess()

		print("loading json data from file ", json_path)
		self.data = pd.read_json(json_path)

		# self.audio_transforms = MelSpectrogram(audio_conf=SpectConfig)
		self.audio_transforms = nn.Sequential(
			RandomNoise(),
			MelSpectrogram(audio_conf=SpectConfig),
			SpecAugment(),
		)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.item()

		try:
			file_path = self.data.key.iloc[idx]
			waveform, _ = torchaudio.load(file_path)
			label = self.text_process.text_to_int(self.data['text'].iloc[idx])
			spectrogram = self.audio_transforms(waveform) # (channel, feature, time)
			spec_len = spectrogram.shape[-1] // 2
			label_len = len(label)
			if spec_len < label_len:
				raise Exception('spectrogram len is bigger then label len')
			if spectrogram.shape[0] > 1:
				raise Exception('dual channel, skipping audio file %s'%file_path)
			if label_len == 0:
				raise Exception('label len is zero... skipping %s'%file_path)
		except Exception as e:
			if self.log:
				print(str(e), self.data.key.iloc[idx])
			return self.__getitem__(idx - 1 if idx != 0 else idx + 1)
		return spectrogram, label, spec_len, label_len

def collate_fn_padd(data):
	'''
	Padds batch of variable length
	note: it converts things ToTensor manually here since the ToTensor transform
	assume it takes in images rather than arbitrary tensors.
	'''
	# print(data)
	spectrograms = []
	labels = []
	input_lengths = []
	label_lengths = []
	for (spectrogram, label, input_length, label_length) in data:
		if spectrogram is None:
			continue
		spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
		labels.append(torch.Tensor(label))
		input_lengths.append(input_length)
		label_lengths.append(label_length)
	spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
	labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
	input_lengths = input_lengths
	# print(spectrograms.shape)
	label_lengths = label_lengths
	# ## compute mask
	# mask = (batch != 0).cuda(gpu)
	# return batch, lengths, mask
	return spectrograms, labels, input_lengths, label_lengths
