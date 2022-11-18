import torch

class WordErrorRate(object):
	"""docstring for WordErrorRate"""
	def __init__(self, arg):
		super(WordErrorRate, self).__init__()
		self.wer = torch.torchmetrics.WordErrorRate()

	def calculate_wer(self, s1, s2):
		return self.wer([s1], [s2])
		