#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


# imports
from torch.nn import Sequential
from torch.nn import Linear, Conv1d, ConvTranspose1d
from torch.nn import Tanh, ReLU, LogSoftmax, Softmax, Sigmoid
from torch.nn import Flatten, Unflatten, MaxPool1d
from torch import optim


# common funcs


class Architecture(object):
    def __init__(self, architecture):
        if isinstance(architecture, dict):
            self.architecture = architecture
        else:
            raise TypeError(
                f"Architecture must be of type dict. {type(architecture)} is received."
            )
        self.name = self.architecture["name"]
        self.type = self.architecture["type"]
        self.vectorizer = None
        self.devectorizer = None
        self.encoder = None
        self.decoder = None
        self.discriminator = None
        self.classifier = None
        self.parse_architecture()

    def parse_architecture(self):
        for key, item in self.architecture.items():
            if key == "vectorizer":
                self.vectorizer = item
            elif key == "devectorizer":
                self.devectorizer = item
            elif key == "encoder":
                self.encoder = item
            elif key == "decoder":
                self.decoder = item
            elif key == "discriminator":
                self.discriminator = item
            elif key == "classifier":
                self.classifier = item


class LayerMaker(object):
    def make(self, arch):
        layers = []
        for layer in arch:
            layers.append(self.make_layer(layer))
        return Sequential(*layers)

    def make_layer(self, layer):
        if layer["type"] == "Linear":
            return Linear(layer["in"], layer["out"])
        elif layer["type"] == "Tanh":
            return Tanh()
        elif layer["type"] == "Sigmoid":
            return Sigmoid()
        elif layer["type"] == "ReLU":
            return ReLU()
        elif layer["type"] == "Conv1d":
            return Conv1d(layer["in"], layer["out"], layer["kernel"])
        elif layer["type"] == "LogSoftmax":
            return LogSoftmax(dim=1)
        elif layer["type"] == "Softmax":
            return Softmax(dim=1)
        elif layer["type"] == "MaxPool1d":
            return MaxPool1d(layer["kernel"])
        elif layer["type"] == "Flatten":
            return Flatten()
        elif layer["type"] == "Unflatten":
            return Unflatten(1, (layer["in"], layer["out"]))
        elif layer["type"] == "ConvTranspose1d":
            return ConvTranspose1d(layer["in"], layer["out"], layer["kernel"])


class CustomLRScheduler(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super(CustomLRScheduler, self).__init__(*args, **kwargs)
        self._last_lr = None

    def get_last_lr(self, default=0.01):
        if self._last_lr is None:
            return default
        return self._last_lr[0]
