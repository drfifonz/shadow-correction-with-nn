import argparse
import sys

from utils.arguments_parser import arguments_parser, print_all_user_arguments
from train import train
from test import test
from dotenv import load_dotenv

load_dotenv()


def main():

    args = arguments_parser()

    print_all_user_arguments(args)
    if args.type == "test":
        train(args)
    elif args.type == "train":
        test(args)
    else:
        sys.exit("Bad type to run")


if __name__ == "__main__":
    main()
