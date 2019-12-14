# pylint: disable=missing-module-docstring
# pylint: disable=wrong-import-position
# pylint: disable=invalid-name

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

import keras
import segmentation_models as sm

import albumentations as alb
from image_augmentation import AugmentedSequence


import matplotlib.pyplot as plt
def plot_training_history(history):
    """
    Plots model training history 
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(history.epoch, history.history["iou_score"], label="Train iou")
    ax_acc.plot(history.epoch, history.history["val_iou_score"], label="Validation iou")
    ax_acc.legend()
    return plt


if __name__ == "__main__":

    ROOT_PATH = "./chicago/"
    TRAIN_DIR = "training"
    VAL_DIR = "validation"
    IMG_DIR = "image"
    MASK_DIR = "mask"
    BACKBONE = "efficientnetb4"
    IMG_SIZE = 192  # Unet requires size to be multiple of 32
    AUGM_COUNT = 19
    BATCH_SIZE = 5
    LR = 0.0001
    EPOCHS = 10
    MODEL_FILE = "./last_best_model.h5"

    # define augmentation

    augm = alb.Compose([
        alb.Flip(),
        alb.Transpose(),
        alb.RandomRotate90(),
        alb.ShiftScaleRotate(),

        alb.RandomGamma(),
        alb.RandomBrightnessContrast(),
        alb.HueSaturationValue(),
        alb.RGBShift(),

        alb.OneOf([
            alb.ElasticTransform(),
            alb.GridDistortion(),
            alb.OpticalDistortion(distort_limit=0.1)
        ], p=0.8),

        alb.OneOf([
            alb.Blur(),
            alb.MotionBlur(),
            alb.MedianBlur(),
            alb.GaussianBlur()
        ]),

        alb.GaussNoise(p=0.25)
    ])

    # create model
    model = sm.Unet(BACKBONE, encoder_weights="imagenet") # activation sigmoid by default, ok for 1 class

    # define optimizer
    optim = keras.optimizers.Adam(learning_rate=LR)

    # define metrics
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimizer, loss and metrics
    model.compile(optim, sm.losses.binary_focal_dice_loss, metrics)

    # define train and validation generators

    train_generator = AugmentedSequence(root_path=os.path.join(ROOT_PATH, TRAIN_DIR),
                                        img_dir=IMG_DIR,
                                        mask_dir=MASK_DIR,
                                        img_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE,
                                        augm_count=AUGM_COUNT,
                                        augm_compose=augm
                                        )

    valid_generator = AugmentedSequence(root_path=os.path.join(ROOT_PATH, VAL_DIR),
                                        img_dir=IMG_DIR,
                                        mask_dir=MASK_DIR,
                                        img_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE,
                                        augm_count=0
                                        )

    # check shapes for errors
    assert train_generator[0][0].shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
    assert train_generator[0][1].shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), train_generator[0][1].shape

    # define callbacks for learning rate reduction and best checkpoint saving
    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_FILE, save_best_only=True, save_weights_only=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau()
    ]

    # train model
    history = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_generator,
        #use_multiprocessing=True  # crashes, BSODs, the apocalypse
    )

    plot = plot_training_history(history)
    plot.savefig(f"300i_{IMG_SIZE}_{AUGM_COUNT}a_{EPOCHS}e.png")
    plot.show()
    