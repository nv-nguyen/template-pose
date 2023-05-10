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
from src.utils.augmentation import Augmentator, CenterCropRandomResizedCrop
from tqdm import tqdm
from src.poses.utils import (
    get_obj_poses_from_template_level,
    load_index_level0_in_level2,
    crop_frame,
)

# set level logging
logging.basicConfig(level=logging.INFO)


class BOPDataset(BaseBOP):
    def __init__(
        self,
        root_dir,
        template_dir,
        split,
        obj_ids,
        img_size,
        use_augmentation=False,
        use_random_geometric=False,
        cropping_with_bbox=True,
        reset_metaData=False,
        isTesting=False,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.template_dir = template_dir
        self.split = split

        self.img_size = img_size
        self.mask_size = 25 if img_size == 64 else int(img_size // 8)
        self.cropping_with_bbox = cropping_with_bbox
        self.use_augmentation = use_augmentation
        self.use_random_geometric = use_random_geometric
        self.augmentator = Augmentator()
        self.random_cropper = CenterCropRandomResizedCrop()

        self.load_template_poses(template_dir=template_dir)
        self.load_testing_indexes()
        if isinstance(obj_ids, str):
            obj_ids = [int(obj_id) for obj_id in obj_ids.split(",")]
            logging.info(f"ATTENTION: Loading {len(obj_ids)} objects!")
        self.load_list_scene(split=split)
        self.load_metaData(
            reset_metaData=reset_metaData,
            mode="query",
        )
        self.obj_ids = (
            obj_ids
            if obj_ids is not None
            else np.unique(self.metaData["obj_id"]).tolist()
        )
        if (
            self.split.startswith("train") or self.split.startswith("val")
        ) and not isTesting:
            # keep only 90% of the data for training for each object
            self.metaData = self.subsample(self.metaData, 90)
            self.isTesting = False
        elif self.split.startswith("test") or isTesting:
            self.metaData = self.subsample(self.metaData, 10)
            self.isTesting = True
            self.use_augmentation = False
            self.use_random_geometric = False
        else:
            logging.warning(f"Split {split} and mode {isTesting} not recognized")
            raise NotImplementedError
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((self.mask_size, self.mask_size)),
                transforms.Lambda(lambda mask: (np.asarray(mask) / 255.0 > 0) * 1),
                transforms.Lambda(lambda mask: torch.from_numpy(mask).unsqueeze(0)),
            ]
        )
        logging.info(
            f"Length of dataloader: {self.__len__()} with mode {self.isTesting} containing objects {np.unique(self.metaData['obj_id'])}"
        )

    def load_template_poses(self, template_dir):
        self.templates_poses = np.load(osp.join(template_dir, "obj_poses.npy"))

    def subsample(self, df, percentage):
        # subsample the data for training and validation
        avail_obj_id = np.unique(df["obj_id"])
        selected_obj_id = [id for id in self.obj_ids]
        logging.info(f"Available {avail_obj_id}, selected {selected_obj_id} ")
        selected_index = []
        index_dataframe = np.arange(0, len(df))
        for obj_id in selected_obj_id:
            selected_index_obj = index_dataframe[  # df["obj_id"] == obj_id]
                np.logical_and(df["obj_id"] == obj_id, df["visib_fract"] >= 0.5)
            ]
            if percentage > 50:
                selected_index_obj = selected_index_obj[
                    : int(percentage / 100 * len(selected_index_obj))
                ]  # keep first
            else:
                selected_index_obj = selected_index_obj[
                    int((1 - percentage / 100) * len(selected_index_obj)) :
                ]  # keep last
            selected_index.extend(selected_index_obj.tolist())
        df = df.iloc[selected_index]
        logging.info(
            f"Subsampled from {len(index_dataframe)} to {len(df)} ({percentage}%) images"
        )
        return df

    def __len__(self):
        return len(self.metaData)

    def load_image(self, idx, type_img):
        if type_img == "synth":
            template_path, _ = self.get_template_path(self.template_dir, idx)
            inplane = self.metaData.iloc[idx]["inplane"]
            rgb = Image.open(template_path)
            rgb = rgb.rotate(inplane)
            return rgb
        else:
            rgb_path = self.metaData.iloc[idx]["rgb_path"]
            rgb = Image.open(rgb_path).convert("RGB")
            if self.use_augmentation:
                rgb = self.augmentator([rgb])[0]
            return rgb

    def make_bbox_square(self, old_bbox):
        size_to_fit = np.max([old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]])
        new_bbox = np.array(old_bbox)
        old_bbox_size = [old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]]
        # Add padding into y axis
        displacement = int((size_to_fit - old_bbox_size[1]) / 2)
        new_bbox[1] = old_bbox[1] - displacement
        new_bbox[3] = old_bbox[3] + displacement
        # Add padding into x axis
        displacement = int((size_to_fit - old_bbox_size[0]) / 2)
        new_bbox[0] = old_bbox[0] - displacement
        new_bbox[2] = old_bbox[2] + displacement
        return new_bbox

    def get_bbox(self, img, idx=None):
        if idx is not None:
            mask_path = self.metaData.iloc[idx]["mask_path"]
            bbox = self.make_bbox_square(Image.open(mask_path).getbbox())
        else:
            bbox = self.make_bbox_square(img.getbbox())
        return bbox

    def crop(self, imgs, bboxes):
        if self.cropping_with_bbox:
            if self.use_random_geometric and not self.isTesting:
                imgs_cropped = self.random_cropper(imgs, bboxes)
            else:
                imgs_cropped = []
                for i in range(len(imgs)):
                    imgs_cropped.append(imgs[i].crop(bboxes[i]))
            return imgs_cropped

    def load_testing_indexes(self):
        self.testing_indexes = load_index_level0_in_level2("all")

    def __getitem__(self, idx):
        if not self.isTesting:
            query = self.load_image(idx, type_img="real")
            template = self.load_image(idx, type_img="synth")
            bboxes = [self.get_bbox(None, idx=idx), self.get_bbox(template)]

            [query, template] = self.crop([query, template], bboxes)
            template_mask = template.getchannel("A")
            template = template.convert("RGB")

            query = self.rgb_transform(query)
            template = self.rgb_transform(template)
            template_mask = self.mask_transform(template_mask)

            # generate a random resized crop parameters
            return {
                "query": query,
                "template": template,
                "template_mask": template_mask,
            }
        else:
            query_pose = self.metaData.iloc[idx]["pose"]
            obj_id = self.metaData.iloc[idx]["obj_id"]
            query = self.load_image(idx, type_img="real")
            query_bbox = self.get_bbox(None, idx=idx)
            imgs, bboxes = [query], [query_bbox]

            # load all templates
            for idx in self.testing_indexes:
                tmp = Image.open(f"{self.template_dir}/obj_{obj_id:06d}/{idx:06d}.png")
                imgs.append(tmp)
                bboxes.append(self.get_bbox(tmp))
            # crop and normalize image
            imgs = self.crop(imgs, bboxes)
            query = self.rgb_transform(imgs[0])
            templates = [
                self.rgb_transform(imgs[i].convert("RGB")) for i in range(1, len(imgs))
            ]
            template_masks = [
                self.mask_transform(imgs[i].getchannel("A"))
                for i in range(1, len(imgs))
            ]
            templates = torch.stack(templates, dim=0)
            template_masks = torch.stack(template_masks, dim=0)

            # loading poses
            query_pose = torch.from_numpy(np.array(query_pose).reshape(4, 4)[:3, :3])
            template_poses = torch.from_numpy(
                self.templates_poses[self.testing_indexes]
            )[:, :3, :3]
            return {
                "query": query,
                "query_pose": query_pose,
                "templates": templates,
                "template_masks": template_masks,
                "template_poses": template_poses,
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from torch.utils.data import DataLoader

    root_dir = "/gpfsscratch/rech/xjd/uyb58rn/datasets/template-pose-released/datasets"
    dataset_names = [
        "hb",
        # "tudl",
        # "hope",
        # "icmi",
        # "icbin",
        # "ruapc",
    ]

    # tless is special
    # for dataset_name, split in zip(["tless/train"], ["train_primesense"]):
    #     dataset = BOPDataset(
    #         root_dir=os.path.join(root_dir, dataset_name),
    #         template_dir=os.path.join(root_dir, f"templates/tless"),
    #         split=split,
    #         obj_ids=None,
    #         img_size=256,
    #         use_augmentation=False,
    #         cropping_with_bbox=True,
    #         reset_metaData=True,
    #     )
    transform_inverse = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )
    os.makedirs("./tmp", exist_ok=True)
    for dataset_name in tqdm(dataset_names):
        splits = [
            split
            for split in os.listdir(os.path.join(root_dir, dataset_name))
            if os.path.isdir(os.path.join(root_dir, dataset_name, split))
        ]
        splits = [
            split
            for split in splits
            if split.startswith("train") or split.startswith("val")
        ]
        for split in splits:
            dataset = BOPDataset(
                root_dir=os.path.join(root_dir, dataset_name),
                template_dir=os.path.join(root_dir, f"templates/{dataset_name}"),
                split=split,
                obj_ids=None,
                img_size=256,
                cropping_with_bbox=True,
                reset_metaData=False,
                use_augmentation=True,
                use_random_geometric=True,
                isTesting=True,
            )
            # train_data = DataLoader(
            #     dataset, batch_size=16, shuffle=False, num_workers=10
            # )
            # train_size, train_loader = len(train_data), iter(train_data)
            # for idx in tqdm(range(train_size)):
            #     batch = next(train_loader)
            #     if idx >= 500:
            #         break
            # logging.info(f"{dataset_name} is running correctly!")
            for idx in range(len(dataset)):
                sample = dataset[idx]
                for k in sample:
                    print(k, sample[k].shape)
                break
                # query = transform_inverse(sample["query"])
                # template = transform_inverse(sample["template"])
                # query = query.permute(1, 2, 0).numpy()
                # query = Image.fromarray(np.uint8(query * 255))
                # query.save(f"./tmp/{dataset_name}_{split}_{idx}_query.png")
                # template = template.permute(1, 2, 0).numpy()
                # template = Image.fromarray(np.uint8(template * 255))
                # template.save(f"./tmp/{dataset_name}_{split}_{idx}_template.png")
                # if idx == 10:
                #     break
