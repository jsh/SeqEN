#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


# imports
import json


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

