# seed albumentations
import random
random.seed(1)

# seed numpy
import numpy as np
np.random.seed(1)

# seed keras backend (tensorflow)
import tensorflow as tf
tf.random.set_seed(1)

# avoid using GPU (crash)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from unet import train
from gen_sub import generate_submission


import argparse
import zipfile

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model and produce prediction. "
                                     "Additionally, you can load the weights of the pre-trained model.",
                                     formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=1024))

    parser.add_argument("-l", "--load-weights",
                        action="store_true",
                        help="Load pre-trained model weigths from file 'model_weights.h5'")

    args = parser.parse_args()

    ZIP_FN = "ml_p2_data.zip"
    if os.path.isfile(ZIP_FN):
        print("=== Unpacking data ===")
        with zipfile.ZipFile(ZIP_FN) as zip_f:
            zip_f.extractall()

    if not args.load_weights:
        print("=== Training model ===")
        train(root_path="./chicago/", model_file="model_weights.h5")
        print("=== Finished training ===")

    generate_submission(model_file="model_weights.h5")
