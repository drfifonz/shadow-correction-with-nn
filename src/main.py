import argparse
import sys

from arguments_parser import arguments_parser
from train import train
from test import test
from dotenv import load_dotenv

load_dotenv()


def main():

    args = arguments_parser()

    if args.type == "test":
        train(args)
    elif args.type == "train":
        test(args)
    else:
        sys.exit("Bad type to run")


if __name__ == "__main__":
    main()
