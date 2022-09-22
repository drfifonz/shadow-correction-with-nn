import argparse
import sys

from utils.arguments_parser import arguments_parser, print_all_user_arguments
from train import train
from test import test
from dotenv import load_dotenv

from utils.visualizer import print_memory_status

# load_dotenv()


def main():

    args = arguments_parser()

    print_all_user_arguments(args)
    if args.type == "test":
        test(args)
    elif args.type == "train":
        print_memory_status()
        train(args)
    else:
        sys.exit("Bad type to run")


if __name__ == "__main__":
    main()
