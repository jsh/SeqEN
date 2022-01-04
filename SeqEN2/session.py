#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


# imports
from SeqEN2.model.model import Model
from os import system
from glob import glob
from SeqEN2.utils.custom_arg_parser import TrainSessionArgParser
from torch import cuda


def get_map_location():
    if cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"
    return map_location


class Session:
    def __init__(
        self, model_name, train_data, test_data, d0=21, d1=8, dn=10, w=20, lr=0.01
    ):
        self.model = Model(
            name=model_name,
            train_data=train_data,
            test_data=test_data,
            d0=d0,
            d1=d1,
            dn=dn,
            w=w,
            lr=lr,
        )

    def train(
        self, run_title, epochs=10, batch_size=128, num_test_items=1, test_interval=100
    ):
        self.model.train(
            epochs=epochs,
            batch_size=batch_size,
            num_test_items=num_test_items,
            test_interval=test_interval,
            run_title=run_title,
        )


def main(args):
    data_files = sorted(glob(str(Model.root) + f"/data/{args['Dataset']}/*.csv.gz"))
    train_data = data_files[2:]
    test_data = data_files[:2]
    # session
    session = Session(
        args["Model Name"],
        train_data,
        test_data,
        d0=args["D0"],
        d1=args["D1"],
        dn=args["Dn"],
        w=args["W"],
        lr=args["Learning Rate"],
    )
    if args["Model ID"] != "":
        session.model.load_model(args["Model ID"], map_location=get_map_location())

    session.train(
        args["Run Title"],
        epochs=args["Epochs"],
        batch_size=args["Train Batch"],
        num_test_items=args["Test Batch"],
        test_interval=args["Test Interval"],
    )


if __name__ == "__main__":
    # parse arguments
    parser = TrainSessionArgParser()
    parsed_args = parser.parsed()
    system("wandb login")
    main(parsed_args)
