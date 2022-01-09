"""Module docstring."""
# TODO: Create annotated tag for version v0.0.2
# TODO: Add docstrings.
# TODO: Add unit tests.
# TODO: Add static typing.
# TODO: Review the code itself
# TODO: Add --verbose flag


from argparse import ArgumentParser


class CustomArgParser(ArgumentParser):
    """TODO: class docstring."""

    def help_value_pairs(self) -> dict[str, str]:
        """Parse CLI arguments, return dict of args, values."""
        parsed_args = self.parse_args()
        help_value_pair_dict = {}
        for _, item in self.__dict__["_option_string_actions"].items():
            if item.dest in parsed_args.__dict__.keys():
                help_value_pair_dict[item.help] = parsed_args.__dict__[item.dest]
        return help_value_pair_dict


class SessionParser:
    """TODO: Class docstring."""

    def __init__(self, desc) -> None:
        """Create parser with description."""
        self.parser = CustomArgParser(description=desc)
        self.initialize()

    def initialize(self) -> None:
        """No-op."""

    def parsed(self) -> dict[str, str]:
        """TODO: function docstring."""
        return self.parser.help_value_pairs()


class TrainSessionArgParser(SessionParser):
    """Capture options to configure training session."""

    def __init__(self) -> None:
        """Initialize & describe the class."""
        super().__init__("Train a protein sequence autoencoder")

    def initialize(self) -> None:
        """Specify command-line options to capture."""
        self.parser.add_argument("-n", "--model", type=str, help="Model Name", required=True)
        self.parser.add_argument("-rt", "--run_title", type=str, help="Run Title", required=True)
        self.parser.add_argument("-d", "--dataset", type=str, help="Dataset", required=True)
        self.parser.add_argument("-a", "--arch", type=str, help="Arch", required=True)
        # add one argument for test train split
        self.parser.add_argument("-d0", "--d0", type=int, help="D0", default=21)
        self.parser.add_argument("-d1", "--d1", type=int, help="D1", default=8)
        self.parser.add_argument("-dn", "--dn", type=int, help="Dn", default=10)
        self.parser.add_argument("-w", "--w", type=int, help="W", default=20)
        self.parser.add_argument(
            "-lr", "--learning_rate", type=float, help="Learning Rate", default=0.01
        )
        self.parser.add_argument("-e", "--epochs", type=int, help="Epochs", default=25)
        self.parser.add_argument("-trb", "--train_batch", type=int, help="Train Batch", default=128)
        self.parser.add_argument("-teb", "--test_batch", type=int, help="Test Batch", default=1)
        self.parser.add_argument(
            "-ti", "--test_interval", type=int, help="Test Interval", default=100
        )
        self.parser.add_argument("-no", "--noise", type=float, help="Input Noise", default=0.0)
        self.parser.add_argument("-mid", "--model_id", type=str, help="Model ID", default="")
        self.parser.add_argument(
            "-tp", "--train_params", type=str, help="Train Params", default=None
        )
        self.parser.add_argument("-mt", "--model_type", type=str, help="Model Type", default="AE")
        self.parser.add_argument(
            "-nt", "--no_train", help="No Train", action="store_true", default=False
        )
        self.parser.add_argument(
            "-it", "--is_testing", help="Is Testing", action="store_true", default=False
        )
