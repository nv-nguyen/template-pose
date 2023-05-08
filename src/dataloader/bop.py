import os, random
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import json
from src.dataloader.base import BaseBOP
import logging
import cv2
import os.path as osp

# set level logging
logging.basicConfig(level=logging.INFO)


class BOPDataset(BaseBOP):
    def __init__(
        self,
        root_dir,
        template_dir,
        obj_ids,
        img_size,
        cropping_with_bbox=None,
        reset_metaData=False,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.template_dir = template_dir
        self.img_size = img_size
        self.mask_size = 25 if img_size == 64 else int(img_size // 8)

        self.cropping_with_bbox = cropping_with_bbox
        self.obj_ids = obj_ids
        self.load_template_poses(template_dir=template_dir)
        self.load_list_scene(obj_ids)
        self.load_metaData(
            reset_metaData=reset_metaData,
            mode="query",
        )
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        logging.info(f"Length of dataloader: {self.__len__()}")

    def load_template_poses(self, template_dir):
        self.templates_poses = np.load(osp.join(template_dir, "obj_poses.npy"))
