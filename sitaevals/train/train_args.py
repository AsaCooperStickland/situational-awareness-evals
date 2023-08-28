import argparse
import inspect
import os
from dataclasses import dataclass
from typing import Dict

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@dataclass
class TrainParams:
    # Required
    data_path: str
    experiment_name: str
    model_name: str
    project_name: str
    task_type: str

    # Data
    data_dir: str = "data/experiment_1"

    # Training
    batch_size: int = 2
    lr: float = 1e-5
    num_epochs: int = 1
    seed: int = 42

    # Extra
    debug: bool = False
    debug_port: int = 5678

    @classmethod
    def from_dict(cls, config: Dict):
        """Create a TrainParams object from a dictionary of variables."""
        return cls(
            **{
                k: v
                for k, v in config.items()
                if k in inspect.signature(cls).parameters
            }
        )

    @classmethod
    def from_argparse(cls, args: argparse.Namespace, parser: argparse.ArgumentParser):
        """Create a TrainParams object from an argparse.Namespace object."""

        # assert no defaults are set on the parser
        assert all(
            [action.default == argparse.SUPPRESS for action in parser._actions]
        ), f"Argparse arguments {[action.dest for action in parser._actions if action.default != argparse.SUPPRESS]} have defaults set. Instead, set defaults on the {cls.__name__} class."

        # assert all required class fields are also required by argparse
        class_fields = inspect.signature(cls).parameters
        required_class_fields = [
            k for k, v in class_fields.items() if v.default == inspect.Parameter.empty
        ]
        required_argparse_fields = [
            action.dest for action in parser._actions if action.required
        ]
        mismatched_required_fields = set(required_class_fields) - set(
            required_argparse_fields
        )
        assert not any(
            mismatched_required_fields
        ), f"Argparse arguments {mismatched_required_fields} must be updated to `required=True` because they don't have a default value in {cls.__name__}."

        return cls(**{k: v for k, v in vars(args).items() if k in class_fields})


def add_training_args(parser: argparse.ArgumentParser):
    training_args = parser.add_argument_group("Training")
    training_args.add_argument("--batch_size", type=int, help="Batch size")
    training_args.add_argument("--lr", type=float, help="Learning rate")
    training_args.add_argument("--num_epochs", type=int, help="Number of epochs")


def add_data_args(parser: argparse.ArgumentParser):
    data_args = parser.add_argument_group("Data")
    data_args.add_argument("--data_dir", type=str, help="Dataset root directory")
    data_args.add_argument(
        "--data_path",
        type=str,
        help="Dataset directory path, starting from `data_dir`",
        required=True,
    )


def add_model_args(parser: argparse.ArgumentParser):
    model_args = parser.add_argument_group("Model")
    model_args.add_argument(
        "--model_name", type=str, help="Model name, e.g. `davinci`", required=True
    )


def add_logging_args(parser: argparse.ArgumentParser):
    logging_args = parser.add_argument_group("Logging")
    logging_args.add_argument(
        "--experiment_name", type=str, help="Experiment name", required=True
    )
    logging_args.add_argument(
        "--project_name", type=str, help="W&B Project name", required=True
    )


def add_extra_args(parser: argparse.ArgumentParser):
    extra_args = parser.add_argument_group("Extra")
    extra_args.add_argument("--debug", action=argparse.BooleanOptionalAction)
    extra_args.add_argument("--debug_port", type=int)
    extra_args.add_argument("--job_id", type=str)
    extra_args.add_argument(
        "--local_rank", type=int, help="local rank passed from distributed launcher"
    )
    extra_args.add_argument("--task_id", type=str)
    extra_args.add_argument(
        "--task_type", type=str, help="Task type, e.g. `experiment_1`"
    )


def get_parser() -> argparse.ArgumentParser:
    # makes it so that arguments that are not set by the user are not included
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    add_training_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    add_logging_args(parser)
    add_extra_args(parser)
    return parser
