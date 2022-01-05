#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from os.path import dirname
import wandb
from glob import glob

from torch.nn.functional import one_hot
from torch import tensor, zeros, ones, randperm, float32
from torch import cuda, device, no_grad, argmax
from torch import sum as torch_sum
from SeqEN2.utils.data_loader import DataLoader
from torch import save as torch_save
from pathlib import Path
from SeqEN2.autoencoder.adversarial_autoencoder import AdversarialAutoencoder


class Model:

    root = Path(dirname(__file__)).parent.parent

    def __init__(self, name, arch, d0=21, d1=8, dn=10, w=20):
        self.name = name
        self.path = self.root / "models" / f"{self.name}"
        self.versions_path = self.path / "versions"
        self.d0 = d0
        self.d1 = d1
        self.dn = dn
        self.w = w
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.autoencoder = AdversarialAutoencoder(
            self.d0, self.d1, self.dn, self.w, arch
        )
        self.autoencoder.to(self.device)
        self.train_data = None
        self.test_data = None
        self.data_loader = None
        self.config = None
        if not self.path.exists():
            self.path.mkdir()
            self.versions_path.mkdir()

    def load_data(self, train_data, test_data):
        if self.test_data is None:
            self.test_data = test_data
        if self.train_data is None:
            self.train_data = train_data
        self.data_loader = DataLoader(train_data, test_data)

    def train_batch(self, input_vals):
        self.autoencoder.train()
        input_ndx = tensor(input_vals[:, : self.w], device=self.device).long()
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        # train encoder_decoder
        self.autoencoder.reconstructor_optimizer.zero_grad()
        reconstructor_output = self.autoencoder.forward_encoder_decoder(one_hot_input)
        reconstructor_loss = self.autoencoder.criterion_NLLLoss(
            reconstructor_output, input_ndx.reshape((-1,))
        )
        reconstructor_loss.backward()
        self.autoencoder.reconstructor_optimizer.step()
        wandb.log({"reconstructor_loss": reconstructor_loss.item()})
        wandb.log(
            {
                "reconstructor_LR": self.autoencoder.reconstructor_lr_scheduler.get_last_lr()[
                    0
                ]
            }
        )
        self.autoencoder.reconstructor_lr_scheduler.step(reconstructor_loss.item())
        # train generator
        self.autoencoder.generator_optimizer.zero_grad()
        generator_output = self.autoencoder.forward_generator(one_hot_input)
        generator_loss = self.autoencoder.criterion_NLLLoss(
            generator_output,
            zeros((generator_output.shape[0],), device=self.device).long(),
        )
        generator_loss.backward()
        self.autoencoder.generator_optimizer.step()
        wandb.log({"generator_loss": generator_loss.item()})
        wandb.log(
            {"generator_LR": self.autoencoder.generator_lr_scheduler.get_last_lr()[0]}
        )
        # train discriminator
        self.autoencoder.discriminator_optimizer.zero_grad()
        ndx = randperm(self.w)
        discriminator_output = self.autoencoder.forward_discriminator(
            one_hot_input[:, ndx, :]
        )
        discriminator_loss = self.autoencoder.criterion_NLLLoss(
            discriminator_output,
            ones((discriminator_output.shape[0],), device=self.device).long(),
        )
        discriminator_loss.backward()
        self.autoencoder.discriminator_optimizer.step()
        wandb.log({"discriminator_loss": discriminator_loss.item()})
        wandb.log(
            {
                "discriminator_LR": self.autoencoder.discriminator_lr_scheduler.get_last_lr()[
                    0
                ]
            }
        )
        gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
        self.autoencoder.generator_lr_scheduler.step(gen_disc_loss)
        self.autoencoder.discriminator_lr_scheduler.step(gen_disc_loss)
        # train classifier
        self.autoencoder.classifier_optimizer.zero_grad()
        classifier_target = tensor(
            input_vals[:, self.w :], device=self.device, dtype=float32
        )
        classifier_output = self.autoencoder.forward_classifier(one_hot_input)
        classifier_loss = self.autoencoder.criterion_MSELoss(
            classifier_output, classifier_target
        )
        classifier_loss.backward()
        self.autoencoder.classifier_optimizer.step()
        wandb.log({"classifier_loss": classifier_loss.item()})
        wandb.log(
            {"classifier_LR": self.autoencoder.classifier_lr_scheduler.get_last_lr()[0]}
        )
        self.autoencoder.classifier_lr_scheduler.step(classifier_loss.item())

    def test_batch(self, input_vals):
        with no_grad():
            input_ndx = tensor(input_vals[:, : self.w], device=self.device).long()
            one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
            (
                reconstructor_output,
                generator_output,
                classifier_output,
            ) = self.autoencoder.forward_test(one_hot_input)
            reconstructor_loss = self.autoencoder.criterion_NLLLoss(
                reconstructor_output, input_ndx.reshape((-1,))
            )
            generator_loss = self.autoencoder.criterion_NLLLoss(
                generator_output,
                zeros((generator_output.shape[0],), device=self.device).long(),
            )
            classifier_target = tensor(
                input_vals[:, self.w :], device=self.device, dtype=float32
            )
            classifier_loss = self.autoencoder.criterion_MSELoss(
                classifier_output, classifier_target
            )
            # reconstructor acc
            reconstructor_ndx = argmax(reconstructor_output, dim=1)
            reconstructor_accuracy = (
                torch_sum(reconstructor_ndx == input_ndx.reshape((-1,)))
                / reconstructor_ndx.shape[0]
            )
            # reconstruction_loss, discriminator_loss, classifier_loss
            wandb.log({"test_reconstructor_loss": reconstructor_loss.item()})
            wandb.log({"test_generator_loss": generator_loss.item()})
            wandb.log({"test_classifier_loss": classifier_loss.item()})
            wandb.log({"test_reconstructor_accuracy": reconstructor_accuracy.item()})

    def train(
        self,
        run_title,
        epochs=10,
        batch_size=128,
        num_test_items=1,
        test_interval=100,
        training_params=None,
    ):
        wandb.init(project=self.name, name=run_title)
        self.config = wandb.config
        self.config.learning_rate = training_params
        self.config.batch_size = batch_size
        self.autoencoder.initialize_training_components(training_params=training_params)
        wandb.watch(self.autoencoder)
        model = wandb.Artifact(f"{self.name}_model", type="model")
        train_dir = self.versions_path / f"{run_title}"
        if not train_dir.exists():
            train_dir.mkdir()
            start_epoch = 0
        else:
            saved_models = [
                int(fp.split("/")[-1].split(".")[0].split("_")[-1])
                for fp in glob(str(train_dir) + f"/epoch_*.model")
            ]
            start_epoch = max(saved_models)
        iter_for_test = 0
        for epoch in range(start_epoch, start_epoch + epochs):
            wandb.log({"epoch": epoch})
            for batch in self.data_loader.get_train_batch(batch_size=batch_size):
                self.train_batch(batch)
                iter_for_test += 1
                if iter_for_test == test_interval:
                    iter_for_test = 0
                    for test_batch in self.data_loader.get_test_batch(
                        num_test_items=num_test_items
                    ):
                        self.test_batch(test_batch)
            model_path = str(train_dir / f"epoch_{epoch}.model")
            torch_save(self.autoencoder, model_path)
            model.add_file(model_path)
            self.autoencoder.save(train_dir, epoch)

    # def load_model(self, model_id, map_location):
    #     version, model_name, run_title = model_id.split(',')          # 0,test,run_title
    #     try:
    #         model_dir = self.root / 'models' / model_name / 'versions' / run_title
    #         self.autoencoder.load(model_dir, version, map_location=map_location)
    #         print('first method is working')
    #     except FileNotFoundError:
    #         model_dir = Path('/mnt/home/nayebiga/SeqEncoder/SeqEN/models') / model_name / 'versions' / run_title
    #         self.autoencoder.load(model_dir, version, map_location=map_location)
    #         print('second method is working')
