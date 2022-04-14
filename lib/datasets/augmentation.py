# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/ylabbe/cosypose/blob/master/cosypose/datasets/augmentations.py
"""
from PIL import ImageEnhance, ImageFilter, Image
import numpy as np
import random


class PillowRGBAugmentation:
    def __init__(self, len_sequences, pillow_fn, p, factor_interval, delta_factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval
        self.delta_factor_interval = (delta_factor_interval[0], delta_factor_interval[1], len_sequences)

    def __call__(self, list_imgs):
        if random.random() <= self.p:
            init_factor = random.uniform(*self.factor_interval)
            delta_factor = np.random.uniform(*self.delta_factor_interval)
            for i in range(len(list_imgs)):
                if list_imgs[i].mode != "RGB":
                    print("Error when apply data aug, image mode:", list_imgs[i].mode)
                    list_imgs[i] = list_imgs[i].convert("RGB")
                    print("Success to change to {}".format(list_imgs[i].mode))
                list_imgs[i] = self._pillow_fn(list_imgs[i]).enhance(factor=init_factor *
                                                                            (1 + delta_factor[i])).convert("RGB")
        return list_imgs


class PillowSharpness(PillowRGBAugmentation):
    def __init__(self, len_sequences, p=0.3, factor_interval=(0, 40.),
                 delta_factor_interval=(0.0, 0.4)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness,
                         p=p,
                         factor_interval=factor_interval,
                         len_sequences=len_sequences,
                         delta_factor_interval=delta_factor_interval)


class PillowContrast(PillowRGBAugmentation):
    def __init__(self, len_sequences, p=0.3, factor_interval=(0.5, 1.6),
                 delta_factor_interval=(0.0, 0.4)):
        super().__init__(pillow_fn=ImageEnhance.Contrast,
                         p=p,
                         factor_interval=factor_interval,
                         len_sequences=len_sequences,
                         delta_factor_interval=delta_factor_interval)


class PillowBrightness(PillowRGBAugmentation):
    def __init__(self, len_sequences, p=0.5, factor_interval=(0.5, 2.0),
                 delta_factor_interval=(0.0, 0.4)):
        super().__init__(pillow_fn=ImageEnhance.Brightness,
                         p=p,
                         factor_interval=factor_interval,
                         len_sequences=len_sequences,
                         delta_factor_interval=delta_factor_interval)


class PillowColor(PillowRGBAugmentation):
    def __init__(self, len_sequences, p=1, factor_interval=(0.0, 20.0),
                 delta_factor_interval=(0.0, 0.4)):
        super().__init__(pillow_fn=ImageEnhance.Color,
                         p=p,
                         factor_interval=factor_interval,
                         len_sequences=len_sequences,
                         delta_factor_interval=delta_factor_interval)


class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.k = random.randint(*factor_interval)

    def __call__(self, list_imgs):
        for i in range(len(list_imgs)):
            if random.random() <= self.p:
                list_imgs[i] = list_imgs[i].filter(ImageFilter.GaussianBlur(self.k))
        return list_imgs


class NumpyGaussianNoise:
    def __init__(self, p, factor_interval=(0.01, 0.3)):
        self.noise_ratio = random.uniform(*factor_interval)
        self.p = p

    def __call__(self, list_imgs):
        for i in range(len(list_imgs)):
            if random.random() <= self.p:
                img = np.copy(list_imgs[i])
                noisesigma = random.uniform(0, self.noise_ratio)
                gauss = np.random.normal(0, noisesigma, img.shape) * 255
                img = img + gauss

                img[img > 255] = 255
                img[img < 0] = 0
                list_imgs[i] = Image.fromarray(np.uint8(img))
        return list_imgs


class RandomOcclusion:
    """
    randomly erasing holes
    ref: https://arxiv.org/abs/1708.04896
    """

    def __init__(self, p=0.5, low_size=0.02, high_size=0.2, low_ratio=0.3, high_ratio=1 / 0.3,
                 low_value=0, high_value=255, low_displacement=0.01, high_displacement=0.05):
        self.p = p
        self.low_size = low_size
        self.high_size = high_size
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.low_value = low_value
        self.high_value = high_value
        self.low_displacement = low_displacement
        self.high_displacement = high_displacement

    def __call__(self, list_imgs):
        img_h, img_w = list_imgs[0].size
        img_c = 3
        if np.random.rand() > self.p:
            return list_imgs
        # generating holes
        size = np.random.uniform(self.low_size, self.high_size) * img_h * img_w
        ratio = np.random.uniform(self.low_ratio, self.high_ratio)
        w = int(np.sqrt(size / ratio))
        h = int(np.sqrt(size * ratio))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        # generating displacement through sequences
        displacement = np.random.uniform(self.low_displacement, self.high_displacement, (len(list_imgs), 2))
        for idx_img in range(len(list_imgs)):
            y1, y2 = displacement[idx_img, 0] + top, displacement[idx_img, 0] + top + h
            x1, x2 = displacement[idx_img, 1] + left, displacement[idx_img, 1] + left + w
            # make sure that holes is inside the image
            [y1, y2] = np.clip(np.asarray([y1, y2]), 0, img_h).astype(int)
            [x1, x2] = np.clip(np.asarray([x1, x2]), 0, img_w).astype(int)
            c = np.random.uniform(self.low_value, self.high_value, (int(x2 - x1), int(y2 - y1), img_c))
            img = np.copy(list_imgs[idx_img].copy())
            img[x1:x2, y1:y2, :] = c
            list_imgs[idx_img] = Image.fromarray(img)
        return list_imgs


def apply_transform_query(img):
    sharpness = PillowSharpness(1, p=0.2, factor_interval=(0.5, 30.))
    contrast = PillowContrast(1, p=0.2, factor_interval=(0.3, 3))
    brightness = PillowBrightness(1, p=0.2, factor_interval=(0.1, 1.5))
    color = PillowColor(1, p=0.2, factor_interval=(0.0, 2.0))
    blur = PillowBlur(p=0.2, factor_interval=(1, 2))
    gaussian_noise = NumpyGaussianNoise(p=0.2, factor_interval=(0.1, 0.04))
    img_aug = sharpness([img])
    img_aug = contrast(img_aug)
    img_aug = brightness(img_aug)
    img_aug = color(img_aug)
    img_aug = blur(img_aug)
    img_aug = gaussian_noise(img_aug)
    return img_aug[0]