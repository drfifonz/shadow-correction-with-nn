import argparse


def arguments_parser():
    """
    parsing arguments with argparse library
    """
    description = "Parser"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--type", type=str, default="test", help="[test/train]")
    parser.add_argument("--batch_size", type=int, default=1, help="Set a batch size")

    return parser.parse_args()
