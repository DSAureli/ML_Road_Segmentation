import os
import random
random.seed(1)

import numpy as np
import cv2
import albumentations as alb


def generate_augm_set(root_path : str = r"./training",
                      img_dir : str = r"images/",
                      mask_dir : str = r"groundtruth/",
                      augm : alb.core.composition.Compose = None,
                      number : int = 1
                      ) -> ((np.ndarray, np.ndarray), {str, int}):
    
    """
    Generate image augmentation set.

    Parameters
    ----------
    root_path : string
        The path of the directory which contains the image directory and the mask directory
    img_dir : string
        The path of the image directory relative to 'root_path'
    mask_dir : string
        The path of the mask directory relative to 'root_path'
    augm : albumentations Compose
        Instance of alb.core.composition.Compose containing the transformations to perform
    number : int
        Number of augmentations generated per image

    Returns
    -------
    ((image, mask), {"image", "num"}) : ((np.ndarray, np.ndarray), {str, int})
        Tuple containing a tuple with image and mask and a dictionary with the image name and number.
        If the image number is 0, the image returned is the original (non-augmented) image.
    """

    assert augm is not None, "Argument 'augm' cannot be None"
    assert isinstance(augm, alb.core.composition.Compose), "Argument 'augm' must be instance of albumentations.core.composition.Compose"

    def meta_dict(img, n) -> dict:
        return {"image": img, "num": n}

    img_path = os.path.join(root_path, img_dir)
    mask_path = os.path.join(root_path, mask_dir)

    for img_file in sorted(os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path, img_file))
        mask = cv2.imread(os.path.join(mask_path, img_file))

        if mask is None:
            print(f"== WARNING: Image {img_file} does not have corresponding mask in \"{mask_path}\", skipping. ==")
            continue

        yield (img, mask), meta_dict(img_file, 0)

        for idx in range(number):
            augm_res = augm(image=img, mask=mask)
            yield (augm_res["image"], augm_res["mask"]), meta_dict(img_file, idx+1)


if __name__ == "__main__":

    img_size = 400

    augm = alb.Compose([
        alb.RandomCrop(width=img_size, height=img_size),
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

    for (augm_img, augm_mask), meta_dict in generate_augm_set("augm_test", "image", "mask", augm=augm, number=10):
        cv2.imwrite("augm_test/augm_image/augm_image_{}.png".format(meta_dict["num"]), augm_img)
        cv2.imwrite("augm_test/augm_mask/augm_mask_{}.png".format(meta_dict["num"]), augm_mask)

    # from tqdm import tqdm
    # def test():
    #    for _,_ in tqdm(generate_augm_set(augm=augm, number=10)):
    #        pass
    
    # import timeit
    # print(timeit.timeit(test, number=1))

    # 10 iterations per image (1000 generations + 100 originals) : 84s