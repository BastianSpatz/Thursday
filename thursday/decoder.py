import torch 
from data_loader import TextProcess

class GreedyDecoder(object):
	"""docstring for GreedyDecoder"""
	def __init__(self, arg):
		super(GreedyDecoder, self).__init__()
		self.text_process = TextProcess()

	def decode(model_output, blank_label=28, collapse_repeated=True):

		arg_maxes = torch.argmax(output, dim=2).squeeze(1)
		decode = []
		for i, index in enumerate(arg_maxes):
			if index != blank_label:
				if collapse_repeated and i != 0 and index == arg_maxes[i -1]:
					continue
				decode.append(index.item())
		return self.text_process.int_to_text(decode)