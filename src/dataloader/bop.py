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
from src.utils.augmentation import (
    Augmentator,
    CenterCropRandomResizedCrop,
    RandomRotation,
)
from src.utils.inout import get_root_project, load_json
from tqdm import tqdm
from src.poses.utils import (
    get_obj_poses_from_template_level,
    load_index_level0_in_level2,
    crop_frame,
    adding_inplane_to_pose,
)

# set level logging
logging.basicConfig(level=logging.INFO)
import copy


class BOPDataset(BaseBOP):
    def __init__(
        self,
        root_dir,
        template_dir,
        split,
        obj_ids,
        img_size,
        use_augmentation=False,
        use_additional_negative_samples_for_training=False,
        use_random_rotation=False,
        use_random_scale_translation=False,
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
        self.use_random_rotation = use_random_rotation
        self.use_random_scale_translation = use_random_scale_translation
        self.use_additional_negative_samples_for_training = (
            use_additional_negative_samples_for_training
        )
        self.augmentator = Augmentator()
        self.random_cropper = CenterCropRandomResizedCrop()
        self.random_rotator = RandomRotation()

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
            self.use_random_rotation = False
            self.use_random_scale_translation = False
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
        self.random_rotation_transfrom = transforms.Compose(
            [transforms.RandomRotation(degrees=(-90, 90))]
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
        return df.reset_index(drop=True)

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
            if self.use_random_scale_translation and not self.isTesting:
                imgs_cropped = self.random_cropper(imgs, bboxes)
            else:
                imgs_cropped = []
                for i in range(len(imgs)):
                    imgs_cropped.append(imgs[i].crop(bboxes[i]))
            return imgs_cropped

    def load_testing_indexes(self):
        self.testing_indexes = load_index_level0_in_level2("all")

    def load_negative_sample_same_obj(self, idx):
        obj_id = self.metaData.iloc[idx]["obj_id"]
        selected_idx = np.random.choice(
            self.metaData[self.metaData.obj_id == obj_id].index
        )
        # loading one sample
        query = self.load_image(selected_idx, type_img="real")
        template = self.load_image(selected_idx, type_img="synth")
        bboxes = [
            self.get_bbox(None, idx=selected_idx),
            self.get_bbox(template),
        ]

        [query, template] = self.crop([query, template], bboxes)
        template_mask = template.getchannel("A")
        template = template.convert("RGB")

        query = self.rgb_transform(query)
        template = self.rgb_transform(template)
        template_mask = self.mask_transform(template_mask)
        return query, template, template_mask

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

            if self.use_random_rotation:
                [query, template, template_mask] = self.random_rotator(
                    [query, template, template_mask]
                )
            sample = {
                "query": query,
                "template": template,
                "template_mask": template_mask,
            }
            if self.use_additional_negative_samples_for_training:
                # additional trick for contrast learning: adding samples of same objects (different poses)
                neg_query, neg_template, neg_mask = self.load_negative_sample_same_obj(
                    idx
                )
                sample["neg_query"] = neg_query
                sample["neg_template"] = neg_template
                sample["neg_template_mask"] = neg_mask
            return sample
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


class BOPDatasetTest(BOPDataset):
    def __init__(
        self,
        root_dir,
        template_dir,
        split,
        img_size,
        obj_id,
        mode,
        linemod_setting=False,
        reset_metaData=False,
        batch_size=None,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.template_dir = template_dir
        self.split = split
        self.obj_id = obj_id
        self.linemod_setting = linemod_setting
        self.mode = mode  # query: load only query image, template:load only templates
        self.img_size = img_size
        self.mask_size = 25 if img_size == 64 else int(img_size // 8)
        self.cropping_with_bbox = True
        self.batch_size = batch_size

        self.load_template_poses(template_dir=template_dir)
        self.load_testing_indexes()
        if self.mode == "query":
            self.load_list_scene(split=split)
            self.load_metaData(
                reset_metaData=reset_metaData,
                mode="query",
            )
            self.metaData = self.metaData[self.metaData.obj_id == obj_id]
            self.metaData.reset_index(inplace=True)
            if not self.linemod_setting:
                init_size = len(self.metaData)
                root_project = get_root_project()
                # for tless setting, we subsample the dataset by taking only images from metaData
                with open(
                    f"{root_project}/src/dataloader/tless_bop19_test_list.json"
                ) as json_file:
                    bop19_test_list = json.load(json_file)
                bop19_test_list = pd.DataFrame.from_dict(
                    bop19_test_list, orient="index"
                )
                bop19_test_list = bop19_test_list.transpose()
                selected_frames = np.zeros(len(self.metaData), dtype=bool)
                for i in tqdm(
                    range(len(self.metaData)), desc="Subsampling MetaData..."
                ):
                    idx_data = [
                        int(self.metaData.iloc[i]["scene_id"]),
                        self.metaData.iloc[i]["frame_id"],
                    ]
                    selected_frames[i] = (bop19_test_list == idx_data).all(1).any()
                self.metaData = self.metaData[selected_frames]
                self.metaData.reset_index(inplace=True)
                logging.info(
                    f"Subsampling from size {init_size} to size {len(self.metaData)} by taking only images of BOP"
                )

        else:
            self.metaData = {
                "inplane": np.arange(0, 360, 10).tolist()
                * self.testing_indexes.shape[0],
                "idx_template": self.testing_indexes.tolist() * 36,
            }
            self.metaData = pd.DataFrame.from_dict(self.metaData, orient="index")
            self.metaData = self.metaData.transpose()
        logging.info(f"Length of dataset: {self.__len__()}")

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
        self.use_augmentation = False
        self.use_random_scale_translation = False

    def __len__(self):
        if self.batch_size == None:
            return len(self.metaData)
        else:
            size = len(self.metaData)
            size = size - size % self.batch_size
            return size

    def load_testing_indexes(self):
        if self.linemod_setting:
            self.testing_indexes, _ = get_obj_poses_from_template_level(
                2, "upper", return_index=True
            )
        else:
            self.testing_indexes, _ = get_obj_poses_from_template_level(
                2, "all", return_index=True
            )

    def __getitem__(self, idx):
        if self.mode == "query":
            query = self.load_image(idx, type_img="real")
            bboxes = [self.get_bbox(None, idx=idx)]
            [query] = self.crop([query], bboxes)

            query = self.rgb_transform(query)
            pose = self.metaData.iloc[idx]["pose"]
            query_pose = torch.from_numpy(np.array(pose).reshape(4, 4)[:3, :3])
            sample = {
                "query": query,
                "query_pose": query_pose,
            }
            if not self.linemod_setting:
                intrinsic = np.array(self.metaData.iloc[idx]["intrinsic"]).reshape(3, 3)
                depth_path = self.metaData.iloc[idx]["depth_path"]
                query_translation = np.array(pose).reshape(4, 4)[:3, 3]
                sample["intrinsic"] = torch.from_numpy(intrinsic)
                sample["depth_path"] = depth_path
                sample["query_translation"] = torch.from_numpy(query_translation)
        else:
            idx_template = self.metaData.iloc[idx]["idx_template"]
            inplane = self.metaData.iloc[idx]["inplane"]

            template_path = osp.join(
                self.template_dir, f"obj_{self.obj_id:06d}/{idx_template:06d}.png"
            )
            template = Image.open(template_path).rotate(inplane)
            template_bbox = template.getbbox()
            template = template.crop(self.make_bbox_square(template.getbbox()))
            template_mask = template.getchannel("A")
            template = template.convert("RGB")

            template = self.rgb_transform(template)
            template_mask = self.mask_transform(template_mask)

            template_pose = self.templates_poses[idx_template][:3, :3]
            template_pose = adding_inplane_to_pose(pose=template_pose, inplane=inplane)
            template_pose = torch.from_numpy(template_pose)
            sample = {
                "template": template,
                "template_mask": template_mask,
                "template_pose": template_pose,
                # "template_bbox": torch.from_numpy(template_bbox),
            }
        return sample


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from torch.utils.data import DataLoader
    from src.model.loss import GeodesicError
    from src.dataloader.lm_utils import query_real_ids
    from torchvision.utils import make_grid, save_image

    root_dir = "/home/nguyen/Documents/datasets/template-pose-released/datasets/"
    transform_inverse = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )
    root_dirs = [
        os.path.join(root_dir, dataset_name) for dataset_name in ["tless/test"]
    ]
    os.makedirs("./tmp", exist_ok=True)
    for idx_dataset in range(len(root_dirs)):
        for obj_id in [21]:
            for mode in ["query"]:
                dataset = BOPDatasetTest(
                    root_dir=root_dirs[idx_dataset],
                    template_dir=os.path.join(root_dir, f"templates_pyrender/tless"),
                    split="test_primesense",
                    obj_id=obj_id,
                    img_size=256,
                    reset_metaData=True,
                    linemod_setting=True,
                    mode=mode,
                )

                # train_data = DataLoader(
                #     dataset, batch_size=36, shuffle=True, num_workers=8
                # )
                # train_size, train_loader = len(train_data), iter(train_data)
                # logging.info(f"object {obj_id}, mode {mode}, length {train_size}")
                for idx in tqdm(range(len(dataset))):
                    # batch = next(train_loader)
                    save_image_path = os.path.join(
                        f"./media/demo/tless_{obj_id:02d}/query_{idx}.png"
                    )
                    sample = dataset[idx]
                    rgb = sample["query"]
                    save_image(
                        transform_inverse(rgb),
                        save_image_path,
                        nrow=1,
                    )
                    print(save_image_path)
                    if idx == 5:
                        break
