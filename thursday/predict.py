from dataset import MelSpectrogram, TextProcess
from train import SpeechModule
import torch
import torchaudio
from torch.nn import functional as F
from configs.train_config import SpectConfig

class PredictorModule():
	def __init__(self, ckpt_path):
		self.model = SpeechModule.load_from_checkpoint(ckpt_path, num_cnn_layers=2, num_rnn_layers=1, rnn_dim=1024, num_classes=29, n_feats=128)
		self.text_process = TextProcess()
		self.audio_transform = MelSpectrogram(audio_conf=SpectConfig)

	def decode_greedy(self, output, blank_label=28, collapse_repeated=True):
		arg_maxes = torch.argmax(output, dim=2).squeeze(1)
		decode = []
		for i, index in enumerate(arg_maxes):
			if index != blank_label:
				if collapse_repeated and i != 0 and index == arg_maxes[i -1]:
					continue
				decode.append(index.item())
		return decode


	def predict(self, file_path):
		self.model.eval()
		waveform, _ = torchaudio.load(file_path)
		spectrogram = self.audio_transform(waveform) # (channel, feature, time)
		output = self.model(spectrogram.unsqueeze(0))
		output = F.log_softmax(output, dim=2)
		output = output.transpose(0, 1).detach()
		res = self.decode_greedy(output)
		return self.text_process.int_to_text(res)

if __name__ == '__main__':
	predictor = PredictorModule("thursday/063df85c21704531893cf7f45aeb42f5/checkpoints/epoch=9-step=7590.ckpt")
	print(predictor.predict("data/wav_clips/de1323d4adbeb3df830bb4ac10e84b0407f28dcfeea1786d34ba36b0bd84f6333dc4c519238cb5b48459df722a6ff0cd650a19ed63687a3a0c0c7b685ee9c2c1.wav"))
	print("I could hear a man crying out in pain in the dentist's office.")

	# [{"key": "data/wav_clips/965f13335faf6116dbf74a25e6582460d9bfd646134e315eabc762b16652f738def8fe2892471f3b50dca40cb123958a7fbf35d08350a1aa7f9369a55bc7e1b4.wav", "text": "I could hear a man crying out in pain in the dentist's office."}, {"key": "data/wav_clips/28d2c6b26c2abcc76c15441d1de31dcb2626d5dc461df3abaa01f7de9b282be3dd39505ccc64034809a9d0ffb6576d667958ff92ad32413f2a1f6c86d1cafd06.wav", "text": "Historic mills were usually powered by water, wind or horse power."}, {"key": "data/wav_clips/a8626bacb816d9ab7c639c6881ce44b2e5abf569e5bbb2d70e676cd131b361986afd58bdfb408c677ee1cd823553d14294902c3fef79f6e57bfef65f8e1b9de4.wav", "text": "I'd better hustle him up!"}, {"key": "data/wav_clips/de1323d4adbeb3df830bb4ac10e84b0407f28dcfeea1786d34ba36b0bd84f6333dc4c519238cb5b48459df722a6ff0cd650a19ed63687a3a0c0c7b685ee9c2c1.wav", "text": "\"Everything in life is an omen,\" said the Englishman, now closing the journal he was reading."}