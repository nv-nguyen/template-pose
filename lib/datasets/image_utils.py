import numpy as np
import os
from PIL import Image, ImageFilter, ImageFile
import torchvision.transforms as transforms
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True
# ImageNet stats
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def open_image(filepath):
    img = Image.open(filepath)
    return img


def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):  # imagenet stats
    img = np.array(img).astype(np.float32) / 255.0
    img = img - mean
    img = img / std
    return img.transpose(2, 0, 1)


def _denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], img_size=224):
    std_matrix = np.repeat(np.asarray(std)[:, np.newaxis], img_size, axis=1)
    std_matrix = np.repeat(std_matrix[:, np.newaxis], img_size, axis=1)
    mean_matrix = np.repeat(np.asarray(mean)[:, np.newaxis], img_size, axis=1)
    mean_matrix = np.repeat(mean_matrix[:, np.newaxis], img_size, axis=1)

    img = img * std_matrix
    img = img + mean_matrix
    img = img * 255
    return img.transpose(1, 2, 0)  # CHW->HWC


def resize_pad(img, dim):
    w, h = img.size
    img = transforms.functional.resize(img, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - img.size[0]) / 2))
    right = int(np.floor((dim - img.size[0]) / 2))
    top = int(np.ceil((dim - img.size[1]) / 2))
    bottom = int(np.floor((dim - img.size[1]) / 2))
    img = transforms.functional.pad(img, (left, top, right, bottom))
    return img


def process_mask_image(mask, mask_size):
    mask = mask.resize((mask_size, mask_size))
    mask = (np.asarray(mask) / 255. > 0) * 1
    mask = torch.from_numpy(mask).unsqueeze(0)
    return mask


def check_bbox_in_image(image, bbox):
    """
    Check bounding box is inside image
    """
    img_size = image.size
    check = np.asarray([bbox[0] >= 0, bbox[1] >= 0, bbox[2] <= img_size[0], bbox[3] <= img_size[1]])
    return (check == np.ones(4, dtype=np.bool)).all()


def crop_image(image, bbox, keep_aspect_ratio):
    if not keep_aspect_ratio:
        return image.crop(bbox)
    else:
        new_bbox = np.array(bbox)
        current_bbox_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        final_size = max(current_bbox_size[0], current_bbox_size[1])
        # Add padding into y axis
        displacement_y = int((final_size - current_bbox_size[1]) / 2)
        new_bbox[1] -= displacement_y
        new_bbox[3] += displacement_y
        # Add padding into x axis
        displacement_x = int((final_size - current_bbox_size[0]) / 2)
        new_bbox[0] -= displacement_x
        new_bbox[2] += displacement_x
        if check_bbox_in_image(image, new_bbox):
            return image.crop(new_bbox)
        else:
            cropped_image = image.crop(bbox)
            return resize_pad(cropped_image, final_size)
