#!/usr/bin/env python3
"""Call CustomArgParser.help_value_pairs()."""

from custom_arg_parser import CustomArgParser

if __name__ == "__main__":

    parser = CustomArgParser()
    parser.add_argument("--debug", "-d", action="store_true", help="I need somebody")
    parser.add_argument("--verbose", "-v", type=str, help="Not just anybody", default="Beatles")
    args = parser.parse_args(["--debug", "--verbose", "Dylan"])
    assert args.debug is True
    assert args.verbose == "Dylan"
    assert parser.help_value_pairs() == {"I need somebody": True, "Not just anybody": "Dylan"}
