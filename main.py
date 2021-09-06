import json
import logging
import os
import time
from argparse import ArgumentParser
import datetime

from finrl.apps import config


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    parser.add_argument(
        "--config_suffix",
        default=""
    )
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    import finrl.train
    if options.mode == "train":
        finrl.train.train_stock_trading()
    elif options.mode == "crypto_train":
        finrl.train.train_crypto_trading(config_suffix=options.config_suffix)


if __name__ == "__main__":
    main()
