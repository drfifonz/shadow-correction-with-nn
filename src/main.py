import argparse
import sys


def parse_arguments():
    description = "Parser"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--type", type=str, default="test", help="[test/train]")
    parser.add_argument("--batch_size", type=int, default=1, help="Set a batch size")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.type == "test":
        pass
    elif args.type == "train":
        pass
    else:
        sys.exit("Bad type to run")


if __name__ == "__main__":
    main()
