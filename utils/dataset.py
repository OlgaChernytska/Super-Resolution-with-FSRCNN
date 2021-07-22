import os
import numpy as np
import random
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import albumentations as A

from utils.constants import (
    IMAGE_FORMAT,
    DOWNSAMPLE_MODE,
    COLOR_CHANNELS,
    UPSCALING_FACTOR,
    HR_IMG_SIZE,
    LR_IMG_SIZE,
)


class DIV2K_Dataset(keras.utils.Sequence):
    def __init__(self, hr_image_folder: str, batch_size: int):
        self.batch_size = batch_size
        self.hr_image_folder = hr_image_folder
        self.image_fns = [
            x for x in os.listdir(hr_image_folder) if x.endswith(IMAGE_FORMAT)
        ]
        self.transform = A.Compose(
            [
                A.RandomCrop(width=HR_IMG_SIZE[0], height=HR_IMG_SIZE[1], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.8
                ),
            ]
        )
        self.to_float = A.ToFloat(max_value=255)

    def __len__(self):
        return len(self.image_fns) // self.batch_size

    def on_epoch_end(self):
        random.shuffle(self.image_fns)

    def __getitem__(self, idx):
        """Returns a batch of samples"""
        i = idx * self.batch_size
        batch_image_fns = self.image_fns[i : i + self.batch_size]
        batch_hr_images = np.zeros((self.batch_size,) + HR_IMG_SIZE + (COLOR_CHANNELS,))
        batch_lr_images = np.zeros((self.batch_size,) + LR_IMG_SIZE + (COLOR_CHANNELS,))

        for i, image_fn in enumerate(batch_image_fns):
            hr_image_pil = Image.open(os.path.join(self.hr_image_folder, image_fn))
            hr_image = np.array(hr_image_pil)

            hr_image_transform = self.transform(image=hr_image)["image"]
            #hr_image_transform = hr_image[:HR_IMG_SIZE[0], :HR_IMG_SIZE[1], :]
            # 
            hr_image_transform_pil = Image.fromarray(hr_image_transform)
            lr_image_transform_pil = hr_image_transform_pil.resize(
                LR_IMG_SIZE, resample=DOWNSAMPLE_MODE
            )
            lr_image_transform = np.array(lr_image_transform_pil)

            batch_hr_images[i] = self.to_float(image=hr_image_transform)["image"]
            batch_lr_images[i] = self.to_float(image=lr_image_transform)["image"]

        return (batch_lr_images, batch_hr_images)