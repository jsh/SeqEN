"""Unit test custom_arg_parser.py."""

import sys

from custom_arg_parser import CustomArgParser, SessionParser, TrainSessionArgParser


def test_custom_arg_parser_basic():
    """CustomArgParser object returns args from parse_args."""
    parser = CustomArgParser()
    parser.add_argument("--debug", "-d", action="store_true", help="I need somebody")
    parser.add_argument("--verbose", "-v", type=str, help="Not just anybody", default="Beatles")
    args = parser.parse_args(["--debug", "--verbose", "Dylan"])
    assert args.debug is True
    assert args.verbose == "Dylan"


def test_help_value_pairs_defaults():
    """CustomArgParser.help_value_pairs() returns correct default."""
    parser = CustomArgParser()
    parser.add_argument("--debug", "-d", action="store_true", help="I need somebody")
    parser.add_argument("--verbose", "-v", type=str, help="Not just anybody", default="Beatles")
    assert parser.help_value_pairs() == {
        "I need somebody": False,
        "Not just anybody": "Beatles",
    }


def test_help_value_pairs_from_args():
    """CustomArgParser.help_value_pairs() returns values set on command line."""
    parser = CustomArgParser()
    parser.add_argument("--debug", "-d", action="store_true", help="I need somebody")
    parser.add_argument("--verbose", "-v", type=str, help="Not just anybody", default="Beatles")
    sys.argv = [sys.argv[0], "--debug", "--verbose=Dylan"]  # hack, cough
    assert parser.help_value_pairs() == {
        "I need somebody": True,
        "Not just anybody": "Dylan",
    }


def test_session_parser_creation():
    """Can set SessionParser description."""
    session = SessionParser("my ob-")
    assert session.parser.description == "my ob-"


def test_session_parsed():
    """SessionParser.parsed() returns correcthelp_value_pairs."""
    session = SessionParser("Carol Kaye")
    session.parser.add_argument("--debug", "-d", action="store_true", help="I need somebody")
    session.parser.add_argument(
        "--verbose", "-v", type=str, help="Not just anybody", default="Beatles"
    )
    sys.argv = [sys.argv[0], "--verbose=Dylan"]  # hack, cough
    assert session.parsed() == {"I need somebody": False, "Not just anybody": "Dylan"}


def test_train_session_arg_parser():
    """TrainSessionArgParser returns correct defaults for options and flags."""
    train_session = TrainSessionArgParser()
    expected_pairs = {
        "Arch": "'Scott Persing'",
        "D0": 21,
        "D1": 8,
        "Dataset": "'All life'",
        "Dn": 10,
        "Epochs": 25,
        "Input Noise": 0.0,
        "Is Testing": False,
        "Learning Rate": 0.01,
        "Model ID": "",
        "Model Name": "T",
        "Model Type": "AE",
        "No Train": False,
        "Run Title": "Run-away",
        "Test Batch": 1,
        "Test Interval": 100,
        "Train Batch": 128,
        "Train Params": None,
        "W": 20,
    }
    sys.argv = [
        sys.argv[0],
        "--model=T",
        "--run_title=Run-away",
        "--dataset='All life'",
        "--arch='Scott Persing'",
    ]  # hack, cough
    assert train_session.parser.description == "Train a protein sequence autoencoder"
    assert train_session.parsed() == expected_pairs
