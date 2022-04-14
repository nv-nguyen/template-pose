import os
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import json
import random

from lib.datasets import image_utils
from lib.datasets.tless import inout
from lib.datasets import augmentation, dataloader_utils

np.random.seed(2021)
random.seed(2021)


def get_mask_size(image_size):
    list_img_size = np.asarray([64, 96, 128, 160, 192, 224, 256])
    list_mask_size = np.asarray([25, 12, 16, 20, 24, 28, 32])
    mask_size = list_mask_size[np.where(list_img_size == image_size)[0]][0]
    return mask_size


class Tless(data.Dataset):
    def __init__(self, root_dir, split, list_id_obj, use_augmentation, image_size, save_path, is_master):
        self.root_dir = root_dir
        self.list_id_obj = list(list_id_obj)
        self.list_background_img = dataloader_utils.get_list_background_img_from_dir(
            os.path.join(self.root_dir, "SUN397"))
        self.use_augmentation = use_augmentation
        self.split = split
        self.image_size = image_size
        self.mask_size = get_mask_size(image_size)
        self.save_path = save_path
        self.is_master = is_master
        self.query_data = self.get_data()
        self.im_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std=[0.229, 0.224, 0.225])])
        print("Length of the dataset: {}".format(self.__len__()))
        if self.is_master:
            self.save_random_sequences()

    def __len__(self):
        return len(self.query_data)

    def get_data(self):
        # load the query frame
        list_files = os.path.join(self.root_dir, "tless_{}.json".format(self.split))
        with open(list_files) as json_file:
            query_frame = json.load(json_file)
        query_frame = pd.DataFrame.from_dict(query_frame, orient='index')
        query_frame = query_frame.transpose()

        bop19_test_list = "./lib/datasets/tless/tless_bop19_test_list.json"
        with open(bop19_test_list) as json_file:
            bop19_test_list = json.load(json_file)
        bop19_test_list = pd.DataFrame.from_dict(bop19_test_list, orient='index')
        bop19_test_list = bop19_test_list.transpose()
        print("Size of BOP19 test list", len(bop19_test_list))

        print("Taking only objects {}".format(self.list_id_obj))
        query_frame = query_frame[query_frame.id_obj.isin(self.list_id_obj)]
        if self.split == "test":
            print("Taking only images from BOP19 test list!!!")
            initial_size = len(query_frame)
            idx_bop_challenges = np.zeros(initial_size, dtype=bool)
            for idx in range(initial_size):
                idx_data = [query_frame.iloc[idx]['id_scene'], query_frame.iloc[idx]['id_frame']]
                idx_bop_challenges[idx] = (bop19_test_list == idx_data).all(1).any()
            query_frame = query_frame[idx_bop_challenges]
            print("Initial size of test set: {}, BOP19 size: {}".format(initial_size, len(query_frame)))
        # shuffle data
        query_frame = query_frame.sample(frac=1, random_state=2021).reset_index(drop=True)
        return query_frame

    def process_background(self, query, query_mask):
        if self.split == "train":
            # if there is gt_mask available for query image, no need to add random background
            index_bkg = np.random.randint(0, len(self.list_background_img))
            img_path = os.path.join(self.root_dir, "SUN397", self.list_background_img[index_bkg])
            bkg_img = Image.open(img_path).resize(query.size)
            query = Image.composite(query, bkg_img, query_mask).convert("RGB")
        return query

    def _sample(self, idx, query):
        if query:
            id_scene = self.query_data.iloc[idx]['id_scene']
            id_frame = self.query_data.iloc[idx]['id_frame']
            idx_frame = self.query_data.iloc[idx]['idx_frame']
            idx_obj_in_scene = self.query_data.iloc[idx]['idx_obj_in_scene']
            id_obj = self.query_data.iloc[idx]['id_obj']
            visib_fract = self.query_data.iloc[idx]['visib_fract']
            # get query image
            query = inout.open_real_image_tless(root_path=self.root_dir, split=self.split,
                                                id_scene=id_scene, id_frame=id_frame,
                                                idx_obj_in_scene=idx_obj_in_scene, image_type="rgb")
            query_mask = inout.open_real_image_tless(root_path=self.root_dir, split=self.split,
                                                     id_scene=id_scene, id_frame=id_frame,
                                                     idx_obj_in_scene=idx_obj_in_scene, image_type="mask")
            query = image_utils.crop_image(query, bbox=query_mask.getbbox(), keep_aspect_ratio=True)
            query_mask = image_utils.crop_image(query_mask, bbox=query_mask.getbbox(), keep_aspect_ratio=True)

            if self.split == "train" and self.use_augmentation:
                query = augmentation.apply_transform_query(query)
            if self.split == "train":  # we process background except when training
                query = self.process_background(query=query, query_mask=query_mask)
            query = query.resize((self.image_size, self.image_size))

            return query, id_scene, id_frame, idx_frame, id_obj, idx_obj_in_scene, visib_fract
        else:
            id_obj = self.query_data.iloc[idx]['id_obj']
            index_nearest_template = self.query_data.iloc[idx]['index_nearest_template']
            gt_inplane = self.query_data.iloc[idx]['gt_inplane']
            gt_template = inout.open_template_tless(root_path=self.root_dir, id_obj=id_obj,
                                                    idx_template=index_nearest_template,
                                                    image_type="rgb", inplane=gt_inplane)
            gt_mask = inout.open_template_tless(root_path=self.root_dir, id_obj=id_obj,
                                                idx_template=index_nearest_template,
                                                image_type="mask", inplane=gt_inplane)
            gt_template = image_utils.crop_image(gt_template, bbox=gt_mask.getbbox(), keep_aspect_ratio=True)
            gt_template = gt_template.resize((self.image_size, self.image_size))

            gt_mask = image_utils.crop_image(gt_mask, bbox=gt_mask.getbbox(), keep_aspect_ratio=True)
            gt_mask = gt_mask.resize((self.image_size, self.image_size))
            return gt_template, gt_mask

    def __getitem__(self, idx):
        query, id_scene, id_frame, idx_frame, id_obj, idx_obj_in_scene, visib_fract = self._sample(idx, query=True)
        query = self.im_transform(query)
        if self.split == "test":
            obj_pose = np.zeros((4, 4))
            obj_pose[3, 3] = 1
            obj_pose[:3, :3] = np.asarray(self.query_data.iloc[idx]['cam_R_m2c']).reshape(3, 3)
            obj_pose[:3, 3] = np.asarray(self.query_data.iloc[idx]['cam_t_m2c']).reshape(3)
            obj_pose = torch.from_numpy(obj_pose)

            gt_idx_template = self.query_data.iloc[idx]['index_nearest_template']
            gt_inplane = self.query_data.iloc[idx]['gt_inplane']
            return dict(query=query, obj_pose=obj_pose, idx_obj_in_scene=idx_obj_in_scene,
                        id_obj=id_obj, id_scene=id_scene, id_frame=id_frame, idx_frame=idx_frame,
                        gt_idx_template=gt_idx_template, gt_inplane=gt_inplane, visib_fract=visib_fract)
        else:
            template, mask = self._sample(idx, query=False)
            template = self.im_transform(template)
            mask = image_utils.process_mask_image(mask, mask_size=self.mask_size)
            return dict(query=query, template=template, mask=mask,
                        id_obj=id_obj, id_scene=id_scene, id_frame=id_frame, idx_frame=idx_frame)

    def save_random_sequences(self):
        len_data = self.__len__()
        list_index = np.unique(np.random.randint(0, len_data, 10))
        print("Saving samples at {}".format(self.save_path))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for idx in list_index:
            save_path = os.path.join(self.save_path, "{:06d}".format(idx))
            query, _, _, _, _, _, _ = self._sample(idx, query=True)
            query.save(save_path + "_query.png")


if __name__ == '__main__':
    from lib.utils.config import Config

    config_global = Config(config_file="./config.json").get_config()
    save_dir = "./draft/TLess"
    for id_obj in range(1, 18):
        #     Tless(root_dir=config_global.root_path, split="test",
        #           use_augmentation=False, list_id_obj=[id_obj],
        #           image_size=224, save_path=os.path.join(save_dir, "testing"), is_master=True)
        Tless(root_dir=config_global.root_path, split="train",
              use_augmentation=True, list_id_obj=[id_obj],
              image_size=224, save_path=os.path.join(save_dir, "training"), is_master=True)
