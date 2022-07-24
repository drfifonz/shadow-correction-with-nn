import argparse

from os import get_terminal_size


def arguments_parser():
    """
    Functrion parses arguments with argparse library.
    """
    description = "Parser"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--type", type=str, default="test", help="[test/train]")
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--threads", type=int, default=5, help="number of threads")
    parser.add_argument(
        "--in_channels", type=int, default=3, help=" number of input channels"
    )
    parser.add_argument(
        "--out_channels", type=int, default=3, help="number of output channels"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of epochs of training"
    )
    parser.add_argument("--dataset", type=str, default="ISTD", help="select dataset")

    return parser.parse_args()


def print_all_user_arguments(arguments):
    """
    Function prints all values setted in parser
    """
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET_COLOR = "\033[0;0m"

    terminal_size = get_terminal_size()

    print("\u2500" * terminal_size.columns)

    for argument in vars(arguments):
        value = getattr(arguments, argument)

        if type(value) is bool and value is True:
            value = GREEN + str(value) + RESET_COLOR
        elif type(value) is bool and value is not True:
            value = RED + str(value) + RESET_COLOR

        print(f"{argument.upper()} = {value}")

    print("\u2500" * terminal_size.columns)