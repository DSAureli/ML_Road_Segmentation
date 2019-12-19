import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from datetime import datetime
import shutil

import cv2
import numpy as np

#from keras.models import load_model
import segmentation_models as sm

from mask_to_submission import masks_to_submission


def generate_submission(model_file: str = "./last_best_model.h5",
                        test_set_dir: str = "./test_set_images/",
                        pr_out_path: str = "./out/",
                        backbone: str = "efficientnetb4",
                        sub_fn: str = "submission.csv"):

    """
    Load weights, generate predictions and submission.

    Parameters
    ----------
    model_file : string
        Path of the file from which to load the weights of the model.
    test_set_dir : string
        Path of the directory containing the images to use for prediction.
    pr_out_path : string
        Path of directory for predictions output.
    backbone : string
        The backbone used for unet.
    sub_fn : string
        Path of the generated submission file.

    Returns
    -------

    -
    """

    if not os.path.exists(pr_out_path):
        os.makedirs(pr_out_path)

    print("=== Loading weights ===")

    model = sm.Unet(backbone, encoder_weights="imagenet")
    model.load_weights(model_file)

    print("=== Weights loaded ===")
    print("=== Generating predictions ===")

    def clean_line():
        print(" " * (shutil.get_terminal_size().columns-1), end='\r')

    for image_fn in os.listdir(test_set_dir):
        clean_line()
        print(f"Generating prediction: {image_fn}", end='\r')
        image = cv2.imread(os.path.join(test_set_dir, image_fn))
        image = np.array([image])
        pr_mask = model.predict(image).round()
        cv2.imwrite(os.path.join(pr_out_path, image_fn), pr_mask.squeeze() * 255)

    clean_line()
    print("=== Predictions generated ===")

    img_fns = [os.path.join(pr_out_path, x) for x in os.listdir(pr_out_path)]
    masks_to_submission(sub_fn, *img_fns)

    print("=== Submission generated ===")

if __name__ == "__main__":
    generate_submission()
