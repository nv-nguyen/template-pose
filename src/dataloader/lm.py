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
import os.path as osp
from src.dataloader.lm_utils import get_list_id_obj_from_split_name
import cv2
# set level logging
logging.basicConfig(level=logging.INFO)

class LINEMOD(BaseBOP):
    def __init__(
        self,
        root_dir,
        template_dir,
        mode,
        split,
        obj_ids,
        img_size,
        virtual_bbox_size=0.25,  # 0.2 in the paper
        reset_metaData=False,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.template_dir = template_dir
        self.mode = mode
        self.split = split
        self.img_size = img_size
        self.mask_size = 25 if img_size == 64 else int(img_size // 8)
        self.virtual_bbox_size = virtual_bbox_size
        self.obj_ids = obj_ids
        self.load_template_poses(template_dir=template_dir)
        self.load_list_scene(obj_ids)
        self.load_metaData(
            reset_metaData=reset_metaData,
            mode=self.mode,
        )
        if self.split == "train":
            # keep only 90% of the data for training for each object
            self.metaData = self.subsample(self.metaData, 90)
        elif self.split == "seen_test":
            self.metaData = self.subsample(self.metaData, 10)

        self.id_symmetry = [0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        logging.info(f"Length of dataloader: {self.__len__()}")

    def __len__(self):
        return len(self.metaData)

    def subsample(self, df, percentage):
        # subsample the data for training and seen object testing
        # keep only 90% of the data for training for each object, 10% for testing
        # make sure that training and testing are disjoint
        avail_obj_id = np.unique(df["obj_id"])
        selected_obj_id = [id + 1 for id in self.obj_ids]
        logging.info(f"Available {avail_obj_id}, selected {selected_obj_id} ")
        selected_index = []
        for obj_id in self.obj_ids:
            df_obj = df[df["obj_id"] == obj_id + 1]
            if percentage > 50:
                df_obj = df_obj[: int(percentage / 100 * len(df_obj))]
            else:
                df_obj = df_obj[int((1 - percentage / 100) * len(df_obj)) :]
            selected_index.extend(df_obj.index)
        df = df.loc[selected_index]
        logging.info(f"Subsampled to {len(df)} ({percentage}%) images")
        return df

    def load_template_poses(self, template_dir):
        self.templates_poses = np.load(osp.join(template_dir, "obj_poses.npy"))

    def load_list_scene(self, obj_ids):
        self.list_scenes = sorted(
            [osp.join(self.root_dir, f"test/{obj_id+1:06d}") for obj_id in obj_ids]
        )
        logging.info(f"Found {len(self.list_scenes)} scenes")

    def process_mask(self, mask):
        mask_size = self.mask_size
        mask = cv2.resize(mask, (mask_size, mask_size))
        mask = (np.asarray(mask) / 255.0 > 0) * 1
        mask = torch.from_numpy(mask).unsqueeze(0)
        return mask

    def load_image(self, idx, type_img):
        K = np.array(self.metaData.iloc[idx]["intrinsic"]).reshape(3, 3)
        virtual_bbox_size = self.virtual_bbox_size
        if type_img == "synth":
            template_path, idx_template = self.get_template_path(self.template_dir, idx)
            mask_path = template_path.replace(".png", "_mask.png")
            img = Image.open(template_path).convert("RGB")
            mask = Image.open(mask_path)
            pose = self.templates_poses[idx_template]
            if np.linalg.norm(pose[:3, 3]) > 100:
                pose[:3, 3] = pose[:3, 3] / 1000  # mm to m
            return self.crop_with_gt_pose(
                np.array(img), mask, pose, K, virtual_bbox_size
            )
        else:
            rgb_path = self.metaData.iloc[idx]["rgb_path"]
            img = Image.open(rgb_path).convert("RGB")
            pose = np.array(self.metaData.iloc[idx]["pose"]).reshape(4, 4)
            pose[:3, 3] = pose[:3, 3] / 1000  # mm to m
            return self.crop_with_gt_pose(
                np.array(img), None, pose, K, virtual_bbox_size
            )

    def __getitem__(self, idx):
        obj_id = self.metaData.iloc[idx]["obj_id"]
        id_symmetry = self.id_symmetry[obj_id - 1]

        if self.mode == "query":
            # real image
            obj_pose = self.metaData.iloc[idx]["pose"]
            query = self.load_image(idx, type_img="real")
            query = self.im_transform(query)

            # synth image
            template, template_mask = self.load_image(idx, type_img="synth")
            template = self.im_transform(template)
            template_mask = self.process_mask(template_mask)

            return {
                "query": query,
                "template": template,
                "template_mask": template_mask,
                "obj_id": obj_id,
                "obj_pose": obj_pose,
                "id_symmetry": id_symmetry,
            }

        elif self.mode == "template":
            template_path = osp.join(
                self.template_dir, f"obj_{obj_id:06d}/{idx:06d}.png"
            )
            mask_path = template_path.replace(".png", "_mask.png")

            template = Image.open(template_path).convert("RGB")
            template = self.im_transform(template)

            template_mask = Image.open(mask_path)
            template_mask = self.process_mask(template_mask)

            pose = self.templates_poses[idx]
            pose = torch.from_numpy(pose).float()
            return {"template": template, "template_mask": template_mask, "pose": pose}


if __name__ == "__main__":
    from torchvision import utils

    logging.basicConfig(level=logging.INFO)
    root_dir = "/home/nguyen/Documents/datasets/templateV2/lm/"
    template_dir = "/home/nguyen/Documents/datasets/templateV2/templates/query"
    reset_metaData = True
    split_metaData = get_list_id_obj_from_split_name("split1")
    transform_inverse = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )
    for mode in ["query"]:
        dataset = LINEMOD(
            root_dir=root_dir,
            template_dir=template_dir,
            mode=mode,
            obj_ids=split_metaData[0],
            img_size=256,
            reset_metaData=True,
        )
        save_dir = f"./tmp/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        for idx in range(len(dataset)):
            batch = dataset[idx]
            utils.save_image(
                transform_inverse(batch["query"]).cuda(),
                f"{save_dir}/{idx}_query.png",
                nrow=1,
            )
            print(f"{save_dir}/{idx}_query.png")
            utils.save_image(
                transform_inverse(batch["template"]).cuda(),
                f"{save_dir}/{idx}_ref.png",
                nrow=4,
            )
            print(f"{save_dir}/{idx}_ref.png")
            print("----")
            if idx >= 5:
                break
