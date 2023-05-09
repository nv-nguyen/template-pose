# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/ylabbe/cosypose/blob/master/cosypose/datasets/augmentations.py
"""
from PIL import ImageEnhance, ImageFilter, Image
import numpy as np
import random
import logging


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
