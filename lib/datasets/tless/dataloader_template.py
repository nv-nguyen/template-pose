import os, sys, random

import numpy as np
import torch

np.random.seed(2021)
random.seed(2021)
from PIL import Image, ImageFilter
import pandas as pd

import torchvision.transforms as transforms
import torch.utils.data as data

from lib.datasets import image_utils
from lib.datasets.tless import inout


def get_mask_size(image_size):
    list_img_size = np.asarray([64, 96, 128, 160, 192, 224, 256])
    list_mask_size = np.asarray([25, 12, 16, 20, 24, 28, 32])
    mask_size = list_mask_size[np.where(list_img_size == image_size)[0]][0]
    return mask_size


class TemplatesTless(data.Dataset):
    def __init__(self, root_dir, id_obj, image_size, save_path, is_master, dense=False):
        self.root_dir = root_dir
        self.id_obj = id_obj
        self.image_size = image_size
        self.mask_size = get_mask_size(image_size)
        self.save_path = save_path
        self.is_master = is_master
        self.dense = dense
        self.template_data = self.get_data()
        self.im_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std=[0.229, 0.224, 0.225])])
        print("Length of the dataset: {}".format(self.__len__()))
        if self.is_master:
            self.save_random_sequences()

    def __len__(self):
        return len(self.template_data)

    def get_data(self):
        idx_templates, inplanes, poses = [], [], []
        template_opengl_poses = inout.read_template_poses(opengl_camera=True, dense=self.dense)
        inplane = np.arange(0, 360, 10)
        for idx_template in range(len(template_opengl_poses)):
            idx_templates.extend([idx_template for _ in range(len(inplane))])
            inplanes.extend([inplane[i] for i in range(len(inplane))])
            poses.extend([template_opengl_poses[idx_template] for _ in range(len(inplane))])
        all_data = {"idx_template": idx_templates, "inplane": inplanes, "poses": poses}
        template_frame = pd.DataFrame.from_dict(all_data, orient='index')
        template_frame = template_frame.transpose()
        return template_frame

    def _sample_template(self, idx):
        idx_template = self.template_data.iloc[idx]['idx_template']
        inplane = self.template_data.iloc[idx]['inplane']
        poses = self.template_data.iloc[idx]['poses']

        template = inout.open_template_tless(root_path=self.root_dir, id_obj=self.id_obj, idx_template=idx_template,
                                             image_type="rgb", inplane=inplane, dense=self.dense)
        mask = inout.open_template_tless(root_path=self.root_dir, id_obj=self.id_obj, idx_template=idx_template,
                                         image_type="mask", inplane=inplane, dense=self.dense)

        template = image_utils.crop_image(template, bbox=mask.getbbox(), keep_aspect_ratio=True)
        template = template.resize((self.image_size, self.image_size))

        mask = image_utils.crop_image(mask, bbox=mask.getbbox(), keep_aspect_ratio=True)
        mask = mask.resize((self.image_size, self.image_size))
        return template, mask, poses, inplane, idx_template

    def __getitem__(self, idx):
        template, mask, pose, inplane, idx_template = self._sample_template(idx)
        template = self.im_transform(template)
        mask = image_utils.process_mask_image(mask, mask_size=self.mask_size)
        obj_pose = torch.from_numpy(pose)
        return dict(template=template, mask=mask, obj_pose=obj_pose,
                    inplane=inplane, idx_template=idx_template)

    def save_random_sequences(self):
        len_data = self.__len__()
        list_index = np.unique(np.random.randint(0, len_data, 10))
        print("Saving samples at {}".format(self.save_path))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for idx in list_index:
            save_path = os.path.join(self.save_path, "{:06d}".format(idx))
            template, mask, _, _, _ = self._sample_template(idx)
            template.save(save_path + "_template.png")
            mask.save(save_path + "_mask.png")


if __name__ == '__main__':
    from lib.utils.config import Config

    config_global = Config(config_file="./config.json").get_config()
    save_dir = "./draft/TLess"
    TemplatesTless(root_dir=config_global.root_path, id_obj=22, image_size=224,
                   save_path=os.path.join(save_dir, "templates"), is_master=True)
