# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/ylabbe/cosypose/blob/master/cosypose/datasets/augmentations.py
"""
from PIL import ImageEnhance, ImageFilter, Image
import numpy as np
import random
import logging
from torchvision.transforms import RandomResizedCrop, ToTensor


class PillowRGBAugmentation:
    def __init__(self, pillow_fn, p, factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        if random.random() <= self.p:
            factor = random.uniform(*self.factor_interval)
            for i in range(len(imgs)):
                if imgs[i].mode != "RGB":
                    logging.warning(
                        f"Error when apply data aug, image mode: {imgs[i].mode}"
                    )
                    imgs[i] = imgs[i].convert("RGB")
                    logging.warning(f"Success to change to {imgs[i].mode}")
                imgs[i] = (self._pillow_fn(imgs[i]).enhance(factor=factor)).convert(
                    "RGB"
                )
        return imgs


class PillowSharpness(PillowRGBAugmentation):
    def __init__(
        self,
        p=0.3,
        factor_interval=(0, 40.0),
    ):
        super().__init__(
            pillow_fn=ImageEnhance.Sharpness,
            p=p,
            factor_interval=factor_interval,
        )


class PillowContrast(PillowRGBAugmentation):
    def __init__(
        self,
        p=0.3,
        factor_interval=(0.5, 1.6),
    ):
        super().__init__(
            pillow_fn=ImageEnhance.Contrast,
            p=p,
            factor_interval=factor_interval,
        )


class PillowBrightness(PillowRGBAugmentation):
    def __init__(
        self,
        p=0.5,
        factor_interval=(0.5, 2.0),
    ):
        super().__init__(
            pillow_fn=ImageEnhance.Brightness,
            p=p,
            factor_interval=factor_interval,
        )


class PillowColor(PillowRGBAugmentation):
    def __init__(
        self,
        p=1,
        factor_interval=(0.0, 20.0),
    ):
        super().__init__(
            pillow_fn=ImageEnhance.Color,
            p=p,
            factor_interval=factor_interval,
        )


class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.k = random.randint(*factor_interval)

    def __call__(self, imgs):
        for i in range(len(imgs)):
            if random.random() <= self.p:
                imgs[i] = imgs[i].filter(ImageFilter.GaussianBlur(self.k))
        return imgs


class NumpyGaussianNoise:
    def __init__(self, p, factor_interval=(0.01, 0.3)):
        self.noise_ratio = random.uniform(*factor_interval)
        self.p = p

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        for i in range(len(imgs)):
            if random.random() <= self.p:
                img = np.copy(imgs[i])
                noisesigma = random.uniform(0, self.noise_ratio)
                gauss = np.random.normal(0, noisesigma, img.shape) * 255
                img = img + gauss

                img[img > 255] = 255
                img[img < 0] = 0
                imgs[i] = Image.fromarray(np.uint8(img))
        return imgs


class Augmentator:
    def __init__(self):
        self.sharpness = PillowSharpness(p=0.2, factor_interval=(0.5, 30.0))
        self.contrast = PillowContrast(p=0.2, factor_interval=(0.3, 3))
        self.brightness = PillowBrightness(p=0.2, factor_interval=(0.1, 1.5))
        self.color = PillowColor(p=0.2, factor_interval=(0.0, 2.0))
        self.blur = PillowBlur(p=0.2, factor_interval=(1, 2))
        self.gaussian_noise = NumpyGaussianNoise(p=0.2, factor_interval=(0.1, 0.04))

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        img_aug = self.sharpness(imgs)
        img_aug = self.contrast(img_aug)
        img_aug = self.brightness(img_aug)
        img_aug = self.color(img_aug)
        img_aug = self.blur(img_aug)
        img_aug = self.gaussian_noise(img_aug)
        return img_aug


class CenterCropRandomResizedCrop:
    def __init__(
        self,
        scale_range=[0.8, 1.0],
        ratio_range=[3.0 / 4, 4.0 / 3],
        translation_x=[-0.02, 0.02],
        translation_y=[-0.02, 0.02],
    ):
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.translation_x = translation_x
        self.translation_y = translation_y

    def transform_bbox(self, bbox, scale, aspect_ratio):
        # Calculate center point of bbox
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        # Scale the bbox around the center point
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        scaled_width = width * scale
        scaled_height = height * scale * aspect_ratio
        scaled_bbox = [
            cx - scaled_width / 2.0,
            cy - scaled_height / 2.0,
            cx + scaled_width / 2.0,
            cy + scaled_height / 2.0,
        ]
        return scaled_bbox

    def __call__(self, imgs, bboxes):
        scale = random.uniform(*self.scale_range)
        aspect_ratio = random.uniform(*self.ratio_range)
        # translation_x = random.uniform(*self.translation_x)
        # translation_y = random.uniform(*self.translation_y)

        if not isinstance(imgs, list):
            imgs = [imgs]
            bboxes = [bboxes]

        imgs_cropped_transformed = []
        for idx in range(len(imgs)):
            bbox_transformed = self.transform_bbox(
                bbox=bboxes[idx],
                scale=scale,
                aspect_ratio=aspect_ratio,
                # translation2d=[translation_x, translation_y],
            )
            # crop image with bbox_transfromed
            imgs_cropped_transformed.append(imgs[idx].crop(bbox_transformed))
        return imgs_cropped_transformed
