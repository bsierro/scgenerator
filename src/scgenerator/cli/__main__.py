import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        description="scgenerator command",
        prog="scgenerator"
    )

    return parser


def main():
    parser = create_parser()
    subparsers = parser.add_subparsers(
        help="sub-command help"
    )

    newconfig = subparsers.add_parser(
        "newconfig",
        help="create a new configuration file"
    )


if __name__ == "__main__":
    main()
