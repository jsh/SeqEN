#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


import gzip
import json

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# data tools
def read_fasta(filename):
    data_dict = {}
    with open(filename, "r") as file:
        for line in file.readlines():
            if line.startswith(">"):
                key = line.strip()[1:]
                data_dict[key] = ""
            else:
                data_dict[key] += line.strip()
    return data_dict


def read_json(filename):
    if filename.endswith(".json.gz"):
        with gzip.open(filename, "r") as file:
            json_bytes = file.read()
            json_str = json_bytes.decode("utf-8")
            return json.loads(json_str)
    elif filename.endswith(".json"):
        with open(filename, "r") as file:
            return json.load(file)
    else:
        raise IOError("File format must be .gz or .json.gz")


def write_json(data_dict, filename, encoder=None):
    if encoder == "numpy":
        encoder = NumpyEncoder
    if filename.endswith(".json.gz"):
        json_str = json.dumps(data_dict, cls=encoder) + "\n"
        json_bytes = json_str.encode("utf-8")
        with gzip.open(filename, "w") as file:
            file.write(json_bytes)
    elif filename.endswith(".json"):
        with open(filename, "w") as file:
            json.dump(data_dict, file)
    else:
        raise IOError("File format must be .gz or .json.gz")


def load(filename):
    data = None
    if isinstance(filename, list):
        data = pd.concat([pd.read_csv(fp, index_col=0) for fp in filename])
    elif filename.endswith("csv.gz") or filename.endswith(".csv"):
        data = pd.read_csv(filename, index_col=0)
    return data


class DataLoader:
    def __init__(self):
        self.train_data_files = None
        self.test_data_files = None
        self.train_data = None
        self.test_data = None
        self.test_items = None

    def set_train_data_files(self, train_data_files):
        self.train_data_files = train_data_files

    def set_test_data_files(self, test_data_files):
        self.test_data_files = test_data_files

    def prepare_test_data(self):
        if self.test_data is None:
            self.test_data = load(self.test_data_files)
            self.test_items = np.unique(
                ["_".join(item.split("_")[:-1]) for item in self.test_data.index]
            )

    def get_train_batch(self, batch_size=128):
        if self.train_data_files is not None:
            for train_data_file in self.train_data_files:
                self.train_data = load(train_data_file).sample(frac=1).reset_index(drop=True)
                num_batch = len(self.train_data) // batch_size
                for i in range(num_batch + 1):
                    yield self.train_data.iloc[i * batch_size : (i + 1) * batch_size].values
        # else: raise NotDefinedError('train data files are not defined')

    def get_test_batch(self, num_test_items=10, random=True):
        if self.train_data_files is not None:
            self.prepare_test_data()
            if num_test_items == -1:
                num_test_items = len(self.test_items)
            if random:
                for item in np.random.choice(self.test_items, size=num_test_items):
                    yield self.test_data[self.test_data.index.str.contains(item)].values
            else:
                for item in self.test_items[:num_test_items]:
                    yield self.test_data[self.test_data.index.str.contains(item)].values
        # else: raise NotDefinedError('test data files are not defined')
