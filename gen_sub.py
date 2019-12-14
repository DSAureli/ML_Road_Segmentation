import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from datetime import datetime

import cv2
import numpy as np

from keras.models import load_model
import segmentation_models as sm

from mask_to_submission import masks_to_submission

def generate_submission():

    MODEL_NAME = "./last_best_model.h5"
    TEST_SET_DIR = "./test_set_images/"
    PR_OUT_PATH = "./out/"
    BACKBONE = "efficientnetb4"

    SUB_FN = "submission_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if not os.path.exists(PR_OUT_PATH):
        os.makedirs(PR_OUT_PATH)

    model = sm.Unet(BACKBONE, encoder_weights="imagenet")
    model.load_weights(MODEL_NAME)

    for image_fn in os.listdir(TEST_SET_DIR):
        image = cv2.imread(os.path.join(TEST_SET_DIR, image_fn))
        image = np.array([image])
        pr_mask = model.predict(image).round()
        cv2.imwrite(os.path.join(PR_OUT_PATH, image_fn), pr_mask.squeeze() * 255)

    img_fns = [os.path.join(PR_OUT_PATH, x) for x in os.listdir(PR_OUT_PATH)]
    masks_to_submission(SUB_FN, *img_fns)

generate_submission()