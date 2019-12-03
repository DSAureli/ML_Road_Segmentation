import random
random.seed(1)

import os
import math
import shutil

import numpy as np
import cv2
import albumentations as alb
from tensorflow.keras.utils import Sequence


class AugmentedSequence(Sequence):

    """
    Subclass of keras.utils.Sequence which generates the augmented image set.

    Attributes
    ----------
    root_path : string
        The path of the directory which contains the image directory and the mask directory.
    img_dir : string
        The path of the image directory relative to 'root_path'.
    mask_dir : string
        The path of the mask directory relative to 'root_path'.
    img_size : int
        The size of each generated image.
        A random crop is performed on each image in the process of generation.
    batch_size : int
        The maximum size of each batch.
        Note that if the mask of an image cannot be found, then the batch containing that image will be smaller,
        as the number of batches must not change.
    augm_count : int
        Number of augmentations generated per image.
        If less than 0, then the original images are returned without augmentation (except for cropping).
        If greater than 0, each batch contains only the equivalent number of augmentations and no original image.
    augm_compose : albumentations Compose
        Instance of alb.core.composition.Compose containing the transformations to perform.
        Note that a random crop of size img_size is always performed by the class prior to applying the provided composition.
    """

    def __init__(self,
                 root_path: str = "./training",
                 img_dir: str = "images",
                 mask_dir: str = "groundtruth",
                 img_size: int = 256,
                 batch_size: int = 1,
                 augm_count: int = 1,
                 augm_compose: alb.core.composition.Compose = None
                 ):

        if augm_count < 1:
            assert augm_compose is not None, "If argument 'augm_count' is less than 1, argument 'augm' shall not be None"
            assert isinstance(augm_compose, alb.core.composition.Compose), \
                "Argument 'augm' must be instance of albumentations.core.composition.Compose"

        self.img_path = os.path.join(root_path, img_dir)
        self.mask_path = os.path.join(root_path, mask_dir)

        self.img_files = sorted(os.listdir(self.img_path))
        # self.mask_files = sorted(os.listdir(self.mask_path))

        # if self.img_files != self.mask_files:
        #     diff = set(self.img_files) - set(self.mask_files)
        #     raise Exception(f"The following training images do not have a corresponding mask in \"{mask_path}\":\n{diff}")

        self.crop_compose = alb.Compose([alb.RandomCrop(width=img_size, height=img_size)])
        # TODO closest size divisable by 32?

        self.batch_size = batch_size
        self.augm_compose = augm_compose

        # self.gen_count = 1 + augm_count
        if augm_count < 1:
            self.gen_count = 1
            self.augm = False
        else:
            self.gen_count = augm_count
            self.augm = True


    def __len__(self):

        """
        Returns the number of batches
        """

        return math.ceil(len(self.img_files) * self.gen_count / self.batch_size)


    def __getitem__(self, idx):

        """
        Returns one batch of generated images
        """

        def load_image_mask(idx):
            img = cv2.imread(os.path.join(self.img_path, self.img_files[idx]))
            mask = cv2.imread(os.path.join(self.mask_path, self.img_files[idx]))
            return img, mask

        # retrieve current image index and current augmentation index

        images_done = idx * self.batch_size
        batch_img_idx, batch_augm_idx = divmod(images_done, self.gen_count)

        batch_img = []
        batch_mask = []

        img, mask = load_image_mask(batch_img_idx)
        augm_space_iter = 0

        # generate AT MOST self.batch_size images

        while augm_space_iter < self.batch_size:

            if batch_augm_idx < self.gen_count:

                # there are still augmentations to generate for current image
                # let's generate them

                if mask is None:
                    print(f"== WARNING: Image {self.img_files[batch_img_idx]}" +
                          f"does not have corresponding mask in \"{self.mask_path}\"; skipping ==")

                else:
                    crop_res = self.crop_compose(image=img, mask=mask)
                    img, mask = crop_res["image"], crop_res["mask"]

                    if self.augm:
                        augm_res = self.augm_compose(image=img, mask=mask)
                        batch_img.append(augm_res["image"])
                        batch_mask.append(augm_res["mask"])

                    else:
                        batch_img.append(img)
                        batch_mask.append(mask)

                batch_augm_idx += 1
                augm_space_iter += 1

            else:

                # all augmentations for current images have been generated
                # move to next image

                batch_img_idx += 1
                batch_augm_idx = 0

                if batch_img_idx < len(self.img_files):
                    img, mask = load_image_mask(batch_img_idx)
                else:
                    break

        return np.array(batch_img), np.array(batch_mask)


    def on_epoch_end(self):
        pass


if __name__ == "__main__":

    # test snippet for AugmentedSequence

    ROOT_PATH = "./training"
    IMG_DIR = "images"
    MASK_DIR = "groundtruth"

    TEST_PATH = "./augm_test"
    AUGM_IMG_DIR = IMG_DIR + "_"
    AUGM_MASK_DIR = MASK_DIR + "_"
    TEST_COUNT = 10

    if os.path.isdir(TEST_PATH):

        while True:
            regen = input(f"Test directory {TEST_PATH} already exists. Delete and regenerate before continuing? [y/n] ")

            if regen == 'n':
                break

            elif regen == 'y':
                shutil.rmtree(TEST_PATH)
                break

    test_augm_img_path = os.path.join(TEST_PATH, AUGM_IMG_DIR)
    test_augm_mask_path = os.path.join(TEST_PATH, AUGM_MASK_DIR)

    if not os.path.isdir(TEST_PATH):

        test_img_path = os.path.join(TEST_PATH, IMG_DIR)
        test_mask_path = os.path.join(TEST_PATH, MASK_DIR)

        os.mkdir(TEST_PATH)
        os.mkdir(test_img_path)
        os.mkdir(test_mask_path)
        os.mkdir(test_augm_img_path)
        os.mkdir(test_augm_mask_path)

        orig_img_path = os.path.join(ROOT_PATH, IMG_DIR)
        orig_mask_path = os.path.join(ROOT_PATH, MASK_DIR)

        for img_file in sorted(os.listdir(orig_img_path)) [:TEST_COUNT]:
            shutil.copy2(os.path.join(orig_img_path, img_file), test_img_path)
            shutil.copy2(os.path.join(orig_mask_path, img_file), test_mask_path)

    IMG_SIZE = 400

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
            alb.OpticalDistortion()                  
        ], p=0.8),

        alb.OneOf([
            alb.Blur(),
            alb.MotionBlur(),
            alb.MedianBlur(),
            alb.GaussianBlur()
        ]),

        alb.GaussNoise(p=0.25),
        #alb.Normalize()
        ])

    for batch_idx, (augm_img_batch, augm_mask_batch) in \
        enumerate(AugmentedSequence(root_path=TEST_PATH, img_size=IMG_SIZE, augm_compose=augm, augm_count=4, batch_size=5)):

        print(f"{batch_idx}: {len(augm_img_batch)}")
        print("---")

        for augm_img_idx, augm_img in enumerate(augm_img_batch):
            cv2.imwrite(os.path.join(test_augm_img_path, f"batch_{batch_idx}-{augm_img_idx}.png"), augm_img)

        for augm_mask_idx, augm_mask in enumerate(augm_mask_batch):
            cv2.imwrite(os.path.join(test_augm_mask_path, f"batch_{batch_idx}-{augm_mask_idx}.png"), augm_mask)
