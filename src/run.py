"""preprocessing, training, and evaluating recommenders.

drafted usage:
- python -m src.run preprocess
- python -m src.run train --model popular
- python -m src.run eval --model popular --split valid --k 10
"""

import argparse

from src.preprocess.mind_reader import build_processed_split, load_processed_split

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIND news recommender runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preprocess")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", choices=["baseline"], required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model", choices=["baseline"], required=True)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        for split in ["train", "test"]:
            build_processed_split(split)
        return
    
    else:
        news_train, beh_train = load_processed_split("train")
        news_test,  beh_test  = load_processed_split("test")


if __name__ == "__main__":
    main()
