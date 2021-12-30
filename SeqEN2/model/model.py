#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from os.path import dirname
import wandb
from glob import glob
from torch.nn import Module, Sequential, NLLLoss, MSELoss
from torch.nn import Linear, Conv1d, ConvTranspose1d
from torch.nn import Tanh, ReLU, LogSoftmax, Softmax
from torch.nn import Flatten, Unflatten
from torch.nn.functional import one_hot
from torch import tensor, zeros, ones, optim, randperm, transpose, float32
from torch import cuda, device, no_grad, argmax
from torch import sum as torch_sum
from SeqEN2.utils.data_loader import DataLoader
from torch import save as torch_save
from torch import load as torch_load
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning import Trainer
from pathlib import Path


class CustomLRScheduler(optim.lr_scheduler.ReduceLROnPlateau):

    def __init__(self, *args, **kwargs):
        super(CustomLRScheduler, self).__init__(*args, **kwargs)
        self._last_lr = None

    def get_last_lr(self):
        if self._last_lr is None:
            return [0.01]
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

    def forward_test(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        discriminator_output = self.discriminator(encoded)
        classifier_output = self.classifier(encoded)
        return devectorized, discriminator_output, classifier_output

    def save(self, model_dir, epoch):
        torch_save(self.vectorizer, model_dir / f'vectorizer_{epoch}.m')
        torch_save(self.encoder, model_dir / f'encoder_{epoch}.m')
        torch_save(self.decoder, model_dir / f'decoder_{epoch}.m')
        torch_save(self.devectorizer, model_dir / f'devectorizer_{epoch}.m')
        torch_save(self.classifier, model_dir / f'classifier_{epoch}.m')
        torch_save(self.discriminator, model_dir / f'discriminator_{epoch}.m')

    def load(self, model_dir, version, map_location):
        self.vectorizer = torch_load(model_dir / f'vectorizer_{version}.m', map_location=map_location)
        self.encoder = torch_load(model_dir / f'encoder_{version}.m', map_location=map_location)
        self.decoder = torch_load(model_dir / f'decoder_{version}.m', map_location=map_location)
        self.devectorizer = torch_load(model_dir / f'devectorizer_{version}.m', map_location=map_location)
        self.classifier = torch_load(model_dir / f'classifier_{version}.m', map_location=map_location)
        self.discriminator = torch_load(model_dir / f'discriminator_{version}.m', map_location=map_location)


class Model:

    root = Path(dirname(__file__)).parent.parent

    def __init__(self, name, train_data, test_data, d0=21, d1=8, dn=10, w=20, lr=0.01):
        self.name = name
        self.path = self.root / 'models' / f'{self.name}'
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
        self.config = None
        # wandb_logger = WandbLogger()
        # self.trainer = Trainer(logger=wandb_logger)
        if not self.path.exists():
            self.path.mkdir()
            self.versions_path.mkdir()
        # define customized optimizers
        self.reconstructor_optimizer = optim.SGD([
            {'params': self.autoencoder.vectorizer.parameters()},
            {'params': self.autoencoder.encoder.parameters()},
            {'params': self.autoencoder.decoder.parameters()},
            {'params': self.autoencoder.devectorizer.parameters()}
        ], lr=self.lr)
        self.reconstructor_lr_scheduler = CustomLRScheduler(
            self.reconstructor_optimizer, factor=0.99, patience=1000, min_lr=0.00001)
        ###
        self.generator_optimizer = optim.SGD([
            {'params': self.autoencoder.vectorizer.parameters()},
            {'params': self.autoencoder.encoder.parameters()},
            {'params': self.autoencoder.discriminator.parameters()}
        ], lr=self.lr)
        self.generator_lr_scheduler = CustomLRScheduler(
            self.generator_optimizer, factor=0.99, patience=1000, min_lr=0.00001)
        ###
        self.discriminator_optimizer = optim.SGD([
            {'params': self.autoencoder.discriminator.parameters()}
        ], lr=self.lr)
        self.discriminator_lr_scheduler = CustomLRScheduler(
            self.discriminator_optimizer, factor=0.99, patience=1000, min_lr=0.00001)
        ###
        self.classifier_optimizer = optim.SGD([
            {'params': self.autoencoder.vectorizer.parameters()},
            {'params': self.autoencoder.encoder.parameters()},
            {'params': self.autoencoder.classifier.parameters()}
        ], lr=self.lr)
        self.classifier_lr_scheduler = CustomLRScheduler(
            self.classifier_optimizer, factor=0.99, patience=1000, min_lr=0.00001)
        # Loss functions
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
        wandb.log({"reconstructor_LR": self.reconstructor_lr_scheduler.get_last_lr()[0]})
        self.reconstructor_lr_scheduler.step(reconstructor_loss.item())
        # train generator
        self.generator_optimizer.zero_grad()
        generator_output = self.autoencoder.forward_generator(one_hot_input)
        generator_loss = self.criterion_NLLLoss(
            generator_output, zeros((generator_output.shape[0],), device=self.device).long()
        )
        generator_loss.backward()
        self.generator_optimizer.step()
        wandb.log({"generator_loss": generator_loss.item()})
        wandb.log({"generator_LR": self.generator_lr_scheduler.get_last_lr()[0]})
        # train discriminator
        self.discriminator_optimizer.zero_grad()
        ndx = randperm(self.w)
        discriminator_output = self.autoencoder.forward_discriminator(one_hot_input[:, ndx, :])
        discriminator_loss = self.criterion_NLLLoss(
            discriminator_output, ones((discriminator_output.shape[0],), device=self.device).long()
        )
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        wandb.log({"discriminator_loss": discriminator_loss.item()})
        wandb.log({"discriminator_LR": self.discriminator_lr_scheduler.get_last_lr()[0]})
        gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
        self.generator_lr_scheduler.step(gen_disc_loss)
        self.discriminator_lr_scheduler.step(gen_disc_loss)
        # train classifier
        self.classifier_optimizer.zero_grad()
        classifier_target = tensor(input_vals[:, self.w:], device=self.device, dtype=float32)
        classifier_output = self.autoencoder.forward_classifier(one_hot_input)
        classifier_loss = self.criterion_MSELoss(classifier_output, classifier_target)
        classifier_loss.backward()
        self.classifier_optimizer.step()
        wandb.log({"classifier_loss": classifier_loss.item()})
        wandb.log({"classifier_LR": self.classifier_lr_scheduler.get_last_lr()[0]})
        self.classifier_lr_scheduler.step(classifier_loss.item())

    def test_batch(self, input_vals):
        with no_grad():
            input_ndx = tensor(input_vals[:, :self.w], device=self.device).long()
            one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
            reconstructor_output, generator_output, classifier_output = self.autoencoder.forward_test(one_hot_input)
            reconstructor_loss = self.criterion_NLLLoss(reconstructor_output, input_ndx.reshape((-1,)))
            generator_loss = self.criterion_NLLLoss(
                generator_output, zeros((generator_output.shape[0],), device=self.device).long()
            )
            classifier_target = tensor(input_vals[:, self.w:], device=self.device, dtype=float32)
            classifier_loss = self.criterion_MSELoss(classifier_output, classifier_target)
            # reconstructor acc
            reconstructor_ndx = argmax(reconstructor_output, dim=1)
            reconstructor_accuracy = torch_sum(reconstructor_ndx == input_ndx.reshape((-1,)))/reconstructor_ndx.shape[0]
            # reconstruction_loss, discriminator_loss, classifier_loss
            wandb.log({"test_reconstructor_loss": reconstructor_loss.item()})
            wandb.log({"test_generator_loss": generator_loss.item()})
            wandb.log({"test_classifier_loss": classifier_loss.item()})
            wandb.log({"test_reconstructor_accuracy": reconstructor_accuracy.item()})

    def train(self, run_title, epochs=10, batch_size=128, num_test_items=1, test_interval=100):
        wandb.init(project=self.name, name=run_title)
        self.config = wandb.config
        self.config.learning_rate = self.lr
        self.config.batch_size = batch_size
        wandb.watch(self.autoencoder)
        model = wandb.Artifact(f'{self.name}_model', type='model')
        train_dir = self.versions_path / f'{run_title}'
        if not train_dir.exists():
            train_dir.mkdir()
            start_epoch = 0
        else:
            saved_models = [
                int(fp.split('/')[-1].split('.')[0].split('_')[-1]) for fp in glob(str(train_dir) + f'/epoch_*.model')
            ]
            start_epoch = max(saved_models)
        iter_for_test = 0
        for epoch in range(start_epoch, start_epoch+epochs):
            wandb.log({"epoch": epoch})
            for batch in self.data_loader.get_train_batch(batch_size=batch_size):
                self.train_batch(batch)
                iter_for_test += 1
                if iter_for_test == test_interval:
                    iter_for_test = 0
                    for test_batch in self.data_loader.get_test_batch(num_test_items=num_test_items):
                        self.test_batch(test_batch)
            model_path = str(train_dir / f'epoch_{epoch}.model')
            torch_save(self.autoencoder, model_path)
            model.add_file(model_path)
            self.autoencoder.save(train_dir, epoch)

    def load_model(self, model_id, map_location):
        version, model_name, run_title = model_id.split(',')          # 0,test,run_title
        model_dir = self.root / 'models' / model_name / run_title
        self.autoencoder.load(model_dir, version, map_location=map_location)
