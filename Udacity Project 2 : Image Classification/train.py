import argparse
import json
from model import FlowerRecognizor
from utils import create_data_loaders


def cli_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory")
    parser.add_argument("--save_directory", default="./")
    parser.add_argument("--arch", action="store", default="densenet121")
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--hidden_units", default=512, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--gpu", action="store_true", default=False)
    return parser.parse_args()


def train():
    args = cli_options()

    fr = FlowerRecognizor(args.arch, args.hidden_units,
                          args.learning_rate, args.gpu)
    train_loader, valid_loader, test_loader, class_to_idx = create_data_loaders(
        args.data_directory)

    fr.train(args.save_directory, train_loader,
             valid_loader, class_to_idx, args.epochs)
    fr.test(test_loader)


if __name__ == "__main__":
    train()