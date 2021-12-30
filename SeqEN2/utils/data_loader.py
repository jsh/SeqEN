#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


# imports
import json
import pandas as pd
import numpy as np


# data tools
def read_fasta(filename):
    data_dict = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            if line.startswith('>'):
                key = line.strip()[1:]
                data_dict[key] = ''
            else:
                data_dict[key] += line.strip()
    return data_dict


def read_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def write_json(data_dict, filename):
    with open(filename, 'w') as file:
        json.dump(data_dict, file)


def load(filename):
    data = None
    if isinstance(filename, list):
        data = pd.concat([pd.read_csv(fp, index_col=0) for fp in filename])
    elif filename.endswith('csv.gz') or filename.endswith('.csv'):
        data = pd.read_csv(filename, index_col=0)
    return data


class DataLoader:

    def __init__(self, train_data, test_data):
        self.train = load(train_data)
        self.test = load(test_data)
        self.test_items = np.unique(['_'.join(item.split('_')[:-1]) for item in self.test.index])

    def get_train_batch(self, batch_size=128):
        self.train = self.train.sample(frac=1).reset_index(drop=True)
        num_batch = len(self.train) // batch_size
        for i in range(num_batch+1):
            yield self.train.iloc[i*batch_size:(i+1)*batch_size].values

    def get_test_batch(self, num_test_items=10):
        for item in np.random.choice(self.test_items, size=num_test_items):
            yield self.test[self.test.index.str.contains(item)].values
