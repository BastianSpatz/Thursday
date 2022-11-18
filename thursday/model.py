import torch
import torch.nn as nn
from torch.nn import functional as F

class LSTMLayer(nn.Module):
	def __init__(self, input_size, hidden_size, bidirectional=False, batch_first=True) -> None:
		super(LSTMLayer, self).__init__()
		self.bidirectional = bidirectional
		self.lstm = nn.LSTM(
						input_size=input_size,
						hidden_size=hidden_size,
						batch_first=batch_first,
						bias=True,
						bidirectional=bidirectional)
		self.layer_norm = nn.LayerNorm(input_size) # Batchnorm?

	def forward(self, x):
		x = self.layer_norm(x)
		x = F.gelu(x)
		x, _ = self.lstm(x)
		if self.bidirectional:
			x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
		return x
		
class DeepSpeech(nn.Module):
	def __init__(self, hidden_layers=5):
		super(DeepSpeech, self).__init__() 
		self.conv = nn.Sequential(
					nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
					nn.BatchNorm2d(32),
					nn.Hardtanh(0, 20, inplace=True),
					nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
					nn.BatchNorm2d(32),
					nn.Hardtanh(0, 20, inplace=True)
				)
		self.rnns = nn.Sequential(
			LSTMLayer(input_size=1024,
						hidden_size=1024,
						batch_first=True,
						bidirectional=True),
						*(
							LSTMLayer(input_size=1024,
								hidden_size=1024,
								batch_first=True,
								bidirectional=True
								) for _ in range(hidden_layers-1)
						)
		)
		self.fully_connected = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 29, bias=False)
        )
	def forward(self, x):
		x = self.conv(x)
		sizes = x.size()
		x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
		x = x.transpose(1, 2)  # TxNxH
		for rnn in self.rnns:
			x = rnn(x)
		x = self.fully_connected(x)
		return x


class ActDropNormCNN1D(nn.Module):
	def __init__(self, n_feats, dropout, keep_shape=False):
		super(ActDropNormCNN1D, self).__init__()
		self.dropout = nn.Dropout(dropout)
		self.norm = nn.LayerNorm(n_feats)
		self.keep_shape = keep_shape
	
	def forward(self, x):
		x = x.transpose(1, 2)
		# x = self.norm(self.dropout(F.gelu(x)))
		x = self.dropout(F.gelu(self.norm(x)))
		if self.keep_shape:
			return x.transpose(1, 2)
		else:
			return x

class CNNLayerNorm(nn.Module):
	"""Layer normalization built for cnns input"""
	def __init__(self, n_feats):
		super(CNNLayerNorm, self).__init__()
		self.layer_norm = nn.LayerNorm(n_feats)

	def forward(self, x):
		# x (batch, channel, feature, time)
		x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
		x = self.layer_norm(x)
		return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):

	def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
		super(ResidualCNN, self).__init__()
		self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
		self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
		self.layer_norm = CNNLayerNorm(n_feats)
		self.dropout_layer = nn.Dropout(dropout)

	def forward(self, x):
		residual = x
		x = self.layer_norm(x)
		x = F.gelu(x)
		x = self.dropout_layer(x)
		x = self.cnn1(x) # batch, time, feature
		x = self.layer_norm(x)
		x = F.gelu(x)
		x = self.dropout_layer(x)
		x = self.cnn2(x) # batch, time, feature
		x += residual
		return x

class BidirectionalLSTM(nn.Module):

	def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
		super(BidirectionalLSTM, self).__init__()
		self.hidden_size = hidden_size

		self.lstm = nn.LSTM(
						input_size=rnn_dim,
						hidden_size=hidden_size,
						num_layers=1,
						batch_first=batch_first,
						dropout=dropout,
						bidirectional=True)
		self.layer_norm = nn.LayerNorm(rnn_dim)
		self.dropout_layer = nn.Dropout(dropout)

	def forward(self, x):
		x = self.layer_norm(x)
		x = F.gelu(x)
		x, (h, c) = self.lstm(x)
		x = self.dropout_layer(x)
		return x

class SpeechRecognition(nn.Module):
	hyper_parameters = {
		"num_classes": 29,
		"n_feats": 81,
		"dropout": 0.1,
		"hidden_size": 1024,
		"num_layers": 1
	}

	def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout=0.35):
		super(SpeechRecognition, self).__init__()
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.cnn = nn.Sequential(
			nn.Conv1d(n_feats, n_feats, 10, 2, padding=10//2),
			ActDropNormCNN1D(n_feats, dropout),
		)
		self.dense = nn.Sequential(
			nn.Linear(n_feats, 128),
			nn.LayerNorm(128),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(128, 128),
			nn.LayerNorm(128),
			nn.GELU(),
			nn.Dropout(dropout),
		)
		self.lstm = nn.Sequential(nn.LSTM(input_size=128, hidden_size=hidden_size,
											num_layers=num_layers, dropout=0.1,
											bidirectional=False),
								nn.LSTM(input_size=hidden_size, hidden_size=hidden_size//2,
											num_layers=num_layers, dropout=0.1,
											bidirectional=False))
		self.layer_norm2 = nn.LayerNorm(hidden_size//2)
		self.dropout2 = nn.Dropout(dropout)
		self.final_fc = nn.Linear(hidden_size, num_classes)

	def _init_hidden(self, batch_size):
		n, hs = self.num_layers, self.hidden_size
		return (torch.zeros(n*1, batch_size, hs),
				torch.zeros(n*1, batch_size, hs))

	def forward(self, x, hidden):
		x = x.squeeze(1)  # batch, feature, time
		x = self.cnn(x) # batch, time, feature
		x = self.dense(x) # batch, time, feature
		x = x.transpose(0, 1) # time, batch, feature
		out, (hn, cn) = self.lstm(x, hidden)
		x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
		return self.final_fc(x), (hn, cn)

class SpeechRecognitionModel(nn.Module):

	hyper_parameters = {
		"num_cnn_layers": 2, 
		"num_rnn_layers": 3, 
		"rnn_dim": 512,
		"num_classes": 29,
		"n_feats": 128
	}

	def __init__(self, num_cnn_layers, num_rnn_layers, rnn_dim, num_classes, n_feats, stride=2, dropout=0.1):
		super(SpeechRecognitionModel, self).__init__()
		n_feats = n_feats//2
		# cnn for extractinh hierachrical features
		# self.cnn = nn.Sequential(
		#     nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
		#     nn.BatchNorm2d(32),
		#     nn.Hardtanh(0, 20, inplace=True),
		#     nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
		#     nn.BatchNorm2d(32),
		#     nn.Hardtanh(0, 20, inplace=True)
		# )
		self.cnn = nn.Sequential(nn.Conv2d(1, 32, 3, stride=stride, padding=3//2))

		self.res_cnn_layers = nn.Sequential(*[
				ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
						for _ in range(num_cnn_layers)
											])
		self.dense_layer = nn.Linear(n_feats*32, rnn_dim)
		self.lstm_layers = nn.Sequential(*[
				BidirectionalLSTM(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
									hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
						for i in range(num_rnn_layers)
											])
		self.classifier = nn.Sequential(
							nn.Linear(rnn_dim*2, rnn_dim),
							nn.GELU(),
							nn.Dropout(dropout),
							nn.Linear(rnn_dim, num_classes))

	def forward(self, x):
		x = self.cnn(x)
		x = self.res_cnn_layers(x)
		sizes = x.size()
		x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
		x = x.transpose(1, 2)
		x = self.dense_layer(x)
		x = self.lstm_layers(x)
		x = self.classifier(x)
		return x