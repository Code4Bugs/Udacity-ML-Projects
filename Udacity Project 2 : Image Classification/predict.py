import argparse
import json
from model import FlowerRecognizor
from utils import process_image


def cli_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", action="store")
    parser.add_argument("checkpoint_path", action="store")
    parser.add_argument("--top_k", action="store", default=1, type=int)
    parser.add_argument("--category_names", action="store",
                        default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", default=False)
    return parser.parse_args()


def predict():
    args = cli_options()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    image = process_image(args.image_path)

    fr = FlowerRecognizor.load_checkpoint(args.checkpoint_path, args.gpu)

    print(f"Predicting flower class for image {args.image_path} ..")
    top_ps, top_class = fr.predict(image, args.top_k)
    for i, c in enumerate(top_class):
        print(f"Prediction {i+1}: "
              f"{cat_to_name[c]} .. "
              f"({100.0 * top_ps[i]:.3f}%)")


if __name__ == "__main__":
    predict()