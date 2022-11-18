import os
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import argparse

import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import Dataset, collate_fn_padd
from model import SpeechRecognitionModel, SpeechRecognition, DeepSpeech

from pytorch_lightning.loggers import CometLogger

from configs.comet_config import comet_config

class SpeechModule(LightningModule):

	def __init__(self, model, args):
		super(SpeechModule, self).__init__()
		self.model = model
		self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
		self.args = args
		self.save_hyperparameters()

	# def step(self, batch):
	# 	spectrograms, labels, input_lengths, label_lengths = batch 
	# 	bs = spectrograms.shape[0]
	# 	hidden = self.model._init_hidden(bs)
	# 	hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
	# 	output, _ = self(spectrograms, (hn, c0))
	# 	output = F.log_softmax(output, dim=2)
	# 	loss = self.criterion(output, labels, input_lengths, label_lengths)
	# 	return loss

	def forward(self, x):
		output = self.model(x)
		return output
		

	def training_step(self, batch, batch_idx):
		# loss = self.step(batch)
		spectrogram, labels, spec_len, label_len = batch
		output = self.model(spectrogram)
		output = F.log_softmax(output, dim=2)
		output = output.transpose(0, 1) # (time, batch, n_class)
		loss = self.criterion(output, labels, spec_len, label_len)
		# self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, on_step=False, logger=True, batch_size=self.args.batch_size)
		self.log("train_loss",  loss.detach(), on_epoch=True, batch_size=self.args.batch_size)

		return loss

	def validation_step(self, batch, batch_idx):
		# loss = self.step(batch)
		spectrogram, labels, spec_len, label_len = batch
		output = self.model(spectrogram)
		output = F.log_softmax(output, dim=2)
		output = output.transpose(0, 1) # (time, batch, n_class)
		loss = self.criterion(output, labels, spec_len, label_len)
		self.log("val_loss",  loss.detach(), on_epoch=True, batch_size=self.args.batch_size)

	# def validation_epoch_end(self, outputs):
	# 	avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
	# 	self.scheduler.step(avg_loss)
	# 	self.logger.log_metrics({"avg_val_loss": avg_loss})
	# 	logs = {'val_loss': avg_loss}
	# 	self.log("avg_val_loss", avg_loss, batch_size=self.args.batch_size)
	# 	return {'val_loss': avg_loss, 'log': logs}

	def configure_optimizers(self):
		self.optimizer = optim.SGD(self.model.parameters(), 
									lr=self.args.learning_rate, 
									momentum=0.9, # Annealing applied to learning rate after each epoch
									nesterov=True,
									weight_decay = 1e-5)  # Initial Weight Decay)
		self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
		return {
		   'optimizer': self.optimizer,
		   'lr_scheduler': self.scheduler, # Changed scheduler to lr_scheduler
				   }
	

	def train_dataloader(self):
		train_dataset = Dataset(json_path=self.args.train_file)
		return DataLoader(dataset=train_dataset,
							num_workers=self.args.num_workers,
							batch_size=self.args.batch_size,
							pin_memory=True,
							collate_fn=collate_fn_padd)

	def val_dataloader(self):
		val_dataset = Dataset(json_path=self.args.val_file)
		return DataLoader(dataset=val_dataset,
							num_workers=self.args.num_workers,
							batch_size=self.args.batch_size,
							pin_memory=True,
							collate_fn=collate_fn_padd)


def train(args):

	# h_params = SpeechRecognitionModel.hyper_parameters
	# h_params.update(args.hparams_override)
	# model = SpeechRecognitionModel(**h_params)
	model = DeepSpeech()

	if args.load_from_checkpoint:
		speech_module = SpeechModule.load_from_checkpoint(args.ckp_path, model=model, args=args)
	else:
		speech_module = SpeechModule(model, args)
	# speech_module = SpeechModule(num_cnn_layers=1, num_rnn_layers=2, rnn_dim=512, num_classes=29, n_feats=128)

	logger = CometLogger(
		api_key=comet_config["api_key"],
		project_name=comet_config["project_name"],
		workspace=comet_config["workspace"],
		experiment_name="thursday")
	ckpt_callback = ModelCheckpoint(save_top_k=-1)
	trainer = Trainer(
		max_epochs=args.epochs, 
		accelerator='gpu',
		devices=1,
		logger=logger,
		# val_check_interval=args.valid_every,
		callbacks=ckpt_callback,
		resume_from_checkpoint=args.ckp_path
		)

	trainer.fit(speech_module)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# distributed training setup
	parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
	parser.add_argument('-w', '--num_workers', default=4, type=int,
						help='n data loading workers, default 0 = main process only')

	# train and valid
	parser.add_argument('--train_file', default="data/train.json", type=str,
						help='json file to load training data')
	parser.add_argument('--val_file', default="data/test.json", type=str,
						help='json file to load testing data')
	parser.add_argument('--valid_every', default=750, required=False, type=int,
						help='valid after every N iteration')

	# dir and path for models and logs
	parser.add_argument('--save_model_path', default="logs/speech_recognition/", type=str,
						help='path to save model')
	parser.add_argument('--ckp_path', default=None, required=False, type=str,
						help='path to load a pretrain model to continue training')
	parser.add_argument('--load_from_checkpoint', default=False, required=False, type=bool,
						help='check path to resume from')

	# general
	parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
	parser.add_argument('--batch_size', default=16, type=int, help='size of batch')
	parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
	parser.add_argument('--pct_start', default=0.3, type=float, help='percentage of growth phase in one cycle')
	parser.add_argument('--div_factor', default=100, type=int, help='div factor for one cycle')
	parser.add_argument("--hparams_override", default={},
						type=str, required=False, 
						help='override the hyper parameters, should be in form of dict. ie. {"attention_layers": 16 }')


	args = parser.parse_args()
	# args.hparams_override = ast.literal_eval(args.hparams_override)
	if args.save_model_path:
	   if not os.path.isdir(os.path.dirname(args.save_model_path)):
		   raise Exception("the directory for path {} does not exist".format(args.save_model_path))

	train(args)
