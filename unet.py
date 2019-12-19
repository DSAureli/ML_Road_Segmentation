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


def train(root_path: str = "./data/",
          train_dir: str = "training",
          val_dir: str = "validation",
          img_dir: str = "image",
          mask_dir: str = "mask",
          backbone: str = "efficientnetb4",
          img_size: int = 192,
          augm_count: int = 19,
          batch_size: int = 5,
          LR: float = 0.0001,
          epochs: int = 10,
          model_file: str = "./last_best_model.h5",
          plot_history: bool = False):

    """
    Train unet on provided dataset, saving the best weights.

    Parameters
    ----------
    root_path : string
        The path of the directory which contains the training directory and the validation directory.
    train_dir : string
        The path of the training directory relative to 'root_path'.
    val_dir : string
        The path of the validation directory relative to 'root_path'.
    img_dir : string
        The path of the image directory relative to both 'img_dir' and 'mask_dir'.
    mask_dir : string
        The path of the mask directory relative to both 'img_dir' and 'mask_dir'.
    backbone : string
        The backbone used for unet.
    img_size : int
        Size of the crop for each image fed to the model.
    augm_count : int
        Number of augmentations for each image in the training dataset.
    batch_size : int
        Size of the batch of images fed to the model.
    LR : float
        Learning rate of the model.
    epochs : int
        Number of training epochs.
    model_file : string
        Path of the file in which to save the best weights of the model.
    plot_history : bool
        If True, save the training history to file.
        The file name has the pattern '{number_of_images}i_{img_size}_{augm_count}a_{epochs}e.png'.

    Returns
    -------

    -
    """

    #root_path = "./chicago/"
    #train_dir = "training"
    #val_dir = "validation"
    #img_dir = "image"
    #mask_dir = "mask"
    #backbone = "efficientnetb4"
    #img_size = 192  # Unet requires size to be multiple of 32
    #augm_count = 19
    #batch_size = 5
    #LR = 0.0001
    #EPOCHS = 10
    #model_file = "./last_best_model.h5"

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
    model = sm.Unet(backbone, encoder_weights="imagenet") # activation sigmoid by default, ok for 1 class

    # define optimizer
    optim = keras.optimizers.Adam(learning_rate=LR)

    # define metrics
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimizer, loss and metrics
    model.compile(optim, sm.losses.binary_focal_dice_loss, metrics)

    # define train and validation generators

    train_generator = AugmentedSequence(root_path=os.path.join(root_path, train_dir),
                                        img_dir=img_dir,
                                        mask_dir=mask_dir,
                                        img_size=img_size,
                                        batch_size=batch_size,
                                        augm_count=augm_count,
                                        augm_compose=augm
                                        )

    valid_generator = AugmentedSequence(root_path=os.path.join(root_path, val_dir),
                                        img_dir=img_dir,
                                        mask_dir=mask_dir,
                                        img_size=img_size,
                                        batch_size=batch_size,
                                        augm_count=0
                                        )

    # check shapes for errors
    assert train_generator[0][0].shape == (batch_size, img_size, img_size, 3)
    assert train_generator[0][1].shape == (batch_size, img_size, img_size, 1), train_generator[0][1].shape

    # define callbacks for learning rate reduction and best checkpoint saving
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_file, save_best_only=True, save_weights_only=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau()
    ]

    # train model
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=valid_generator,
        #use_multiprocessing=True  # crashes, BSODs, the apocalypse
    )

    if plot_history:
        plot = plot_training_history(history)
        plot.savefig(f"{len(train_generator.img_files)}i_{img_size}_{augm_count}a_{epochs}e.png")


if __name__ == "__main__":
    train()
