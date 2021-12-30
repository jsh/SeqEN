#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from os.path import dirname
import wandb
from torch.nn import Module, Sequential, Dropout, NLLLoss, MSELoss
from torch.nn import Linear, Conv1d, ConvTranspose1d
from torch.nn import Tanh, ReLU, LogSoftmax, Softmax
from torch.nn import MaxPool1d, Flatten, Unflatten
from torch.nn.functional import one_hot
from torch import tensor, zeros, ones, optim, randperm, transpose, float32
from torch import cuda, device, save
from SeqEN2.utils.data_loader import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pathlib import Path
from IPython.display import clear_output


class CustomLRScheduler(optim.lr_scheduler.ReduceLROnPlateau):

    def __init__(self, *args, **kwargs):
        super(CustomLRScheduler, self).__init__(*args, **kwargs)
        self._last_lr = None

    def get_last_lr(self):
        return self._last_lr


class AdversarialAutoencoder(Module):

    def __init__(self, d0, d1, dn, w):
        super(AdversarialAutoencoder, self).__init__()
        self.d0 = d0
        self.d1 = d1
        self.dn = dn
        self.w = w
        self.vectorizer = Sequential(
            Linear(self.d0, self.d1),
            Tanh()
        )
        self.encoder = Sequential(
            Conv1d(self.d1, 64, (8,)),
            Tanh(),
            Conv1d(64, 32, (6,)),
            Tanh(),
            Flatten(),
            Linear(256, self.dn),
            Tanh()
        )
        self.decoder = Sequential(
            Linear(self.dn, 256),
            ReLU(),
            Unflatten(1, (32, 8)),
            ConvTranspose1d(32, 32, (6,)),
            ReLU(),
            ConvTranspose1d(32, self.d1, (8,)),
            ReLU()
        )
        self.devectorizer = Sequential(
            Linear(self.d1, self.d0),
            LogSoftmax(dim=1)
        )
        self.classifier = Sequential(
            Linear(self.dn, 256),
            ReLU(),
            Linear(256, 2),
            Softmax(dim=1)
        )
        self.discriminator = Sequential(
            Linear(self.dn, 256),
            ReLU(),
            Linear(256, 2),
            LogSoftmax(dim=1)
        )

    def forward_encoder_decoder(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        return devectorized

    def forward_generator(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        discriminator_output = self.discriminator(encoded)
        return discriminator_output

    def forward_discriminator(self, one_hot_input):
        return self.forward_generator(one_hot_input)

    def forward_classifier(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        classifier_output = self.classifier(encoded)
        return classifier_output


class Model():

    _root = Path(dirname(__file__)).parent.parent

    def __init__(self, name, train_data, test_data, d0=21, d1=8, dn=10, w=20, lr=0.01):
        self.name = name
        self.path = self._root / 'models' / f'{self.name}'
        self.versions_path = self.path / 'versions'
        self.d0 = d0
        self.d1 = d1
        self.dn = dn
        self.lr = lr
        self.w = w
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        self.autoencoder = AdversarialAutoencoder(self.d0, self.d1, self.dn, self.w)
        self.autoencoder.to(self.device)
        self.data_loader = DataLoader(train_data, test_data)
        wandb.init(project=self.name)
        self.config = wandb.config
        self.config.learning_rate = self.lr
        # wandb_logger = WandbLogger()
        # self.trainer = Trainer(logger=wandb_logger)
        if not self.path.exists():
            self.path.mkdir()
            self.versions_path.mkdir()
        # define customized optimizers
        self.reconstructor_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.classifier_optimizer = None
        self.criterion_NLLLoss = NLLLoss()
        self.criterion_MSELoss = MSELoss()


    def train_batch(self, input_vals):
        self.autoencoder.train()
        input_ndx = tensor(input_vals[:, :self.w], device=self.device).long()
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        # train encoder_decoder
        self.reconstructor_optimizer.zero_grad()
        reconstructor_output = self.autoencoder.forward_encoder_decoder(one_hot_input)
        reconstructor_loss = self.criterion_NLLLoss(reconstructor_output, input_ndx.reshape((-1,)))
        reconstructor_loss.backward()
        self.reconstructor_optimizer.step()
        wandb.log({"reconstructor_loss": reconstructor_loss.item()})
        self.reconstructor_lr_scheduler.step(reconstructor_loss.item())
        # train generator
        self.generator_optimizer.zero_grad()
        generator_output = self.autoencoder.forward_generator(one_hot_input)
        generator_loss = self.criterion_NLLLoss(generator_output, zeros((generator_output.shape[0],)).long())
        generator_loss.backward()
        self.generator_optimizer.step()
        wandb.log({"generator_loss": generator_loss.item()})
        self.generator_lr_scheduler.step(generator_loss.item())
        # train discriminator
        self.discriminator_optimizer.zero_grad()
        ndx = randperm(self.w)
        discriminator_output = self.autoencoder.forward_discriminator(one_hot_input[:, ndx, :])
        discriminator_loss = self.criterion_NLLLoss(discriminator_output, ones((discriminator_output.shape[0],)).long())
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        wandb.log({"discriminator_loss": discriminator_loss.item()})
        self.discriminator_lr_scheduler.step(discriminator_loss.item())
        # train classifier
        self.classifier_optimizer.zero_grad()
        classifier_target = tensor(input_vals[:, self.w:], device=self.device, dtype=float32)
        classifier_output = self.autoencoder.forward_classifier(one_hot_input)
        classifier_loss = self.criterion_MSELoss(classifier_output, classifier_target)
        classifier_loss.backward()
        self.classifier_optimizer.step()
        wandb.log({"classifier_loss": classifier_loss.item()})
        self.classifier_lr_scheduler.step(classifier_loss.item())


    def initialize_training_components(self):
        # define customized optimizers
        self.reconstructor_optimizer = optim.SGD([
            {'params': self.autoencoder.vectorizer.parameters()},
            {'params': self.autoencoder.encoder.parameters()},
            {'params': self.autoencoder.decoder.parameters()},
            {'params': self.autoencoder.devectorizer.parameters()}
        ], lr=self.lr)
        self.reconstructor_lr_scheduler = CustomLRScheduler(
            self.reconstructor_optimizer, factor=0.99, patience=10000, cooldown=100, min_lr=0.000001)
        ###
        self.generator_optimizer = optim.SGD([
            {'params': self.autoencoder.vectorizer.parameters()},
            {'params': self.autoencoder.encoder.parameters()},
            {'params': self.autoencoder.discriminator.parameters()}
        ], lr=self.lr)
        self.generator_lr_scheduler = CustomLRScheduler(
            self.generator_optimizer, factor=0.99, patience=10000, cooldown=100, min_lr=0.000001)
        ###
        self.discriminator_optimizer = optim.SGD([
            {'params': self.autoencoder.discriminator.parameters()}
        ], lr=self.lr)
        self.discriminator_lr_scheduler = CustomLRScheduler(
            self.discriminator_optimizer, factor=0.99, patience=10000, cooldown=100, min_lr=0.000001)
        ###
        self.classifier_optimizer = optim.SGD([
            {'params': self.autoencoder.vectorizer.parameters()},
            {'params': self.autoencoder.encoder.parameters()},
            {'params': self.autoencoder.classifier.parameters()}
        ], lr=self.lr)
        self.classifier_lr_scheduler = CustomLRScheduler(
            self.classifier_optimizer, factor=0.99, patience=10000, cooldown=100, min_lr=0.000001)


    def train(self, epochs=10, batch_size=128, run_title=''):
        self.initialize_training_components()
        self.config.batch_size = batch_size
        wandb.watch(self.autoencoder)
        model = wandb.Artifact(f'{self.name}_model', type='model')
        for epoch in range(epochs):
            self.config.epoch = epoch
            for batch in self.data_loader.get_train_batch(batch_size):
                self.train_batch(batch)
                # print(f'Reconstructor LR: {self.reconstructor_lr_scheduler.get_last_lr()[0]}')
                # print(f'Generator LR: {self.generator_lr_scheduler.get_last_lr()[0]}')
                # print(f'Discriminator LR: {self.discriminator_lr_scheduler.get_last_lr()[0]}')
                # print(f'Classifier LR: {self.classifier_lr_scheduler.get_last_lr()[0]}')
                # clear_output(wait=True)
            model_path = str(self.versions_path / f'epoch_{epoch}_{run_title}.model')
            save(self.autoencoder, model_path)
            model.add_file(model_path)










