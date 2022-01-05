#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from glob import glob
from os import system

from os.path import dirname
from pathlib import Path

from torch import cuda

from SeqEN2.model.model import Model
from SeqEN2.utils.custom_arg_parser import TrainSessionArgParser
from SeqEN2.utils.data_loader import read_json


def get_map_location():
    if cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"
    return map_location


class Session:

    root = Path(dirname(__file__)).parent

    def __init__(self):
        # setup dirs
        self.models_dir = self.root / "models"
        if not self.models_dir.exists():
            self.models_dir.mkdir()
        self.data_dir = self.root / "data"
        if not self.data_dir.exists():
            self.data_dir.mkdir()
        self.arch_dir = self.root / "arch"
        if not self.arch_dir.exists():
            self.arch_dir.mkdir()

        # model placeholder
        self.model = None

    def add_model(self, name, arch, d0=21, d1=8, dn=10, w=20):
        arch = self.load_arch(arch)
        if self.model is None:
            self.model = Model(name, arch, d0=d0, d1=d1, dn=dn, w=w)

    def load_data(self, dataset_name):
        data_files = sorted(glob(str(Model.root) + f"/data/{dataset_name}/*.csv.gz"))
        self.model.load_data(dataset_name, data_files)

    def load_arch(self, arch):
        arch_path = self.root / "arch" / f"{arch}.json"
        return read_json(arch_path)

    def train(
        self,
        run_title,
        epochs=10,
        batch_size=128,
        num_test_items=1,
        test_interval=100,
        training_params=None,
    ):
        self.model.train(
            run_title,
            epochs=epochs,
            batch_size=batch_size,
            num_test_items=num_test_items,
            test_interval=test_interval,
            training_params=training_params,
        )


def main(args):
    # session
    session = Session()
    session.add_model(
        args["Model Name"],
        read_json(args["Arch"]),
        d0=args["D0"],
        d1=args["D1"],
        dn=args["Dn"],
        w=args["W"],
    )
    # load datafiles
    session.load_data(args["Dataset"])
    # if args['Model ID'] != '':
    #     session.model.load_model(args['Model ID'], map_location=get_map_location())
    if args["Train Params"] is None:
        training_params = None
    else:
        training_params = read_json(args["Train Params"])
    session.train(
        args["Run Title"],
        epochs=args["Epochs"],
        batch_size=args["Train Batch"],
        num_test_items=args["Test Batch"],
        test_interval=args["Test Interval"],
        training_params=training_params,
    )


if __name__ == "__main__":
    # parse arguments
    parser = TrainSessionArgParser()
    parsed_args = parser.parsed()
    system("wandb login")
    main(parsed_args)
