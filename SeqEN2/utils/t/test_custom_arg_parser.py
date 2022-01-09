"""Unit test custom_arg_parser.py."""

import argparse

from custom_arg_parser import CustomArgParser


def test_custom_arg_parser_control():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", help="I need somebody")
    parser.add_argument("--verbose", "-v", type=str, help="Not just anybody", default="Beatles")
    args = parser.parse_args(["--debug", "--verbose", "Dylan"])
    assert args.debug == True
    assert args.verbose == "Dylan"


def test_custom_arg_parser_basic():
    parser = CustomArgParser()
    parser.add_argument("--debug", "-d", action="store_true", help="I need somebody")
    parser.add_argument("--verbose", "-v", type=str, help="Not just anybody", default="Beatles")
    args = parser.parse_args(["--debug", "--verbose", "Dylan"])
    assert args.debug == True
    assert args.verbose == "Dylan"


def test_help_value_pairs_defaults():
    parser = CustomArgParser()
    parser.add_argument("--debug", "-d", action="store_true", help="I need somebody")
    parser.add_argument("--verbose", "-v", type=str, help="Not just anybody", default="Beatles")
    assert parser.help_value_pairs() == {"I need somebody": False, "Not just anybody": "Beatles"}


def test_help_value_pairs_from_args():
    parser = CustomArgParser()
    parser.add_argument("--debug", "-d", action="store_true", help="I need somebody")
    parser.add_argument("--verbose", "-v", type=str, help="Not just anybody", default="Beatles")
    args = parser.parse_args(["--debug", "--verbose", "Dylan"])
    assert args.debug == True
    assert args.verbose == "Dylan"
    assert parser.help_value_pairs() == {"I need somebody": True, "Not just anybody": "Dylan"}
