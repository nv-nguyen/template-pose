import os, random
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import json
from lib.poses import utils
from lib.datasets import image_utils, dataloader_utils
from lib.datasets.linemod import inout

np.random.seed(2022)
random.seed(2022)
number_train_template = 1542
number_test_template = 301


def get_mask_size(image_size):
    list_img_size = np.asarray([64, 96, 128, 160, 192, 224, 256])
    list_mask_size = np.asarray([25, 12, 16, 20, 24, 28, 32])
    mask_size = list_mask_size[np.where(list_img_size == image_size)[0]][0]
    return mask_size


class LINEMOD(data.Dataset):
    def __init__(self, root_dir, dataset, list_id_obj, split, image_size, save_path, is_master):
        self.root_dir = root_dir
        self.dataset_name = dataset
        self.list_id_obj = list(list_id_obj)
        self.split = split
        self.image_size = image_size
        self.mask_size = get_mask_size(image_size)
        self.save_path = save_path
        self.is_master = is_master
        self.query_data, self.template_data = self.get_data()
        self.im_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std=[0.229, 0.224, 0.225])])
        print("Length of the dataset: {}".format(self.__len__()))
        if self.is_master:
            self.save_random_sequences()

    def __len__(self):
        return len(self.query_data)

    def get_data(self):
        # load the query frame
        list_files = os.path.join(self.root_dir, self.dataset_name + ".json")
        print(list_files)
        with open(list_files) as json_file:
            query_frame = json.load(json_file)
        query_frame = pd.DataFrame.from_dict(query_frame, orient='index')
        query_frame = query_frame.transpose()
        print(len(query_frame))
        print("Id object available {}".format(sorted(query_frame['id_obj'].unique())))
        print("Taking only objects {}".format(self.list_id_obj))
        query_frame = query_frame[query_frame.id_obj.isin(self.list_id_obj)]
        query_frame = query_frame.sample(frac=1, random_state=2022).reset_index(drop=True)

        if "test" in self.split:
            if self.split == "seen_test":
                # take 10% to evaluate on seen objects (unseen poses)
                index_test = query_frame.groupby('id_obj').apply(dataloader_utils.sampling_k_samples
                                                                 ).index.get_level_values(1)
                index_test = np.asarray(index_test)
                query_frame = query_frame.iloc[index_test]
                query_frame = query_frame.sample(frac=1, random_state=2022).reset_index(drop=True)
                print("Split test seen: ", len(query_frame))
                return query_frame, None
            else:
                return query_frame, None
        else:
            index_test = query_frame.groupby('id_obj').apply(dataloader_utils.sampling_k_samples
                                                             ).index.get_level_values(1)
            index_test = np.asarray(index_test)
            query_frame = query_frame.drop(index_test)
            query_frame = query_frame.sample(frac=1, random_state=2022).reset_index(drop=True)
            query_frame["synthetic_path"] = query_frame["train_template_path"]
            query_frame["synthetic_location"] = query_frame["train_template_location"]

            # load the training template frame
            list_path, list_poses, ids_obj, id_symmetry = [], [], [], []
            if os.path.exists("./lib/poses/predefined_poses/half_sphere_level2_and_level3.npy"):
                obj_poses = np.load("./lib/poses/predefined_poses/half_sphere_level2_and_level3.npy")
            else:
                obj_poses = np.load("../lib/poses/predefined_poses/half_sphere_level2_and_level3.npy")
            for id_obj in self.list_id_obj:
                if self.dataset_name == "occlusionLINEMOD":
                    obj_name = inout.occlusion_real_id_to_name[id_obj]
                else:
                    obj_name = inout.LINEMOD_real_id_to_name[id_obj]
                for id_frame in range(number_train_template):
                    list_path.append(os.path.join(obj_name, "{:06d}.png".format(id_frame)))
                    location = utils.opencv2opengl(np.asarray(obj_poses[id_frame]))[2, :3]
                    list_poses.append(location)
                    ids_obj.append(id_obj)
                    id_symmetry.append(inout.list_all_id_symmetry[id_obj])
            all_data = {"id_obj": ids_obj,
                        "id_symmetry": id_symmetry,
                        "obj_poses": list_poses,
                        "synthetic_path": list_path}
            template_frame = pd.DataFrame.from_dict(all_data, orient='index')
            template_frame = template_frame.transpose()

            # shuffle data
            template_frame = template_frame.sample(frac=1).reset_index(drop=True)
            print("Split seen training ", len(query_frame))
            return query_frame, template_frame

    def _sample_mask(self, idx):
        rgb_path = self.query_data.iloc[idx]['synthetic_path']
        mask_path = os.path.join(self.root_dir, "crop_image{}".format(self.image_size),
                                 rgb_path.replace(".png", "_mask.png"))
        mask = Image.open(mask_path)
        return mask

    def _sample(self, idx, isQuery, isPositive=None, isDiff_obj=None):
        """
        Sampling function given that whether
        1. Image is query (or real image),
        2. Image have same pose as idx,
        3. Image is same object
        """
        return_mask = False
        if isQuery:
            img_path = os.path.join(self.root_dir, "crop_image{}".format(self.image_size),
                                    self.dataset_name, self.query_data.iloc[idx]['real_path'])
        else:
            if isPositive:
                # print("split", self.split, self.dataset_name, self.query_data.keys())
                img_path = os.path.join(self.root_dir, "crop_image{}".format(self.image_size),
                                        self.query_data.iloc[idx]['synthetic_path'])
            else:
                id_first_obj = self.query_data.iloc[idx]['id_obj']
                id_symmetry = inout.list_all_id_symmetry[id_first_obj]
                if isDiff_obj:  # negative is of different obj, can be real or synthetic
                    list_id_second_obj = self.list_id_obj.copy()
                    list_id_second_obj.remove(id_first_obj)
                    id_second_obj = np.random.choice(list_id_second_obj)
                    new_frame = self.query_data[self.query_data.id_obj == id_second_obj]
                    idx_frame_second_obj = np.random.randint(0, len(new_frame))
                    if random.random() >= 0.0:
                        img_path = os.path.join(self.root_dir, "crop_image{}".format(self.image_size),
                                                new_frame.iloc[idx_frame_second_obj]['synthetic_path'])
                    else:
                        img_path = os.path.join(self.root_dir, "crop_image{}".format(self.image_size),
                                                self.dataset_name, new_frame.iloc[idx_frame_second_obj]['real_path'])
                else:  # negative is of same object but different pose and is real image
                    return_mask = True
                    new_frame = self.query_data
                    new_frame = new_frame[new_frame.id_obj == id_first_obj]
                    query_pose = self.query_data.iloc[idx]["real_location"]
                    if id_symmetry != 0:  # if objects are symmetry
                        template_poses = new_frame.real_location.values
                        template_poses = np.vstack(template_poses)
                        new_frame = new_frame[np.logical_or(np.abs(template_poses[:, 0]) != np.abs(query_pose[0]),
                                                            np.abs(template_poses[:, 1]) != np.abs(query_pose[1]))]
                    # select a template such that the angle is dissimilar (delta>10)
                    delta_degree = 0
                    while delta_degree < 10:
                        idx_frame_second_pose = np.random.randint(0, len(new_frame))
                        template_pose = np.asarray(new_frame.iloc[idx_frame_second_pose]['real_location'])
                        division_term = np.linalg.norm(query_pose) * np.linalg.norm(template_pose)
                        delta = np.clip(template_pose.dot(query_pose) / division_term, a_min=0, a_max=1)
                        delta_degree = np.rad2deg(np.arccos(delta))

                    img_path = os.path.join(self.root_dir, "crop_image{}".format(self.image_size),
                                            new_frame.iloc[idx_frame_second_pose]['synthetic_path'])
                    mask_path = img_path.replace(".png", "_mask.png")
                    mask = Image.open(mask_path)
        img = Image.open(img_path)
        img = image_utils.resize_pad(img, self.image_size)
        if return_mask:
            return delta, img, mask
        else:
            return img

    def _sample_triplet(self, idx, save_path=None):
        img1 = self._sample(idx, isQuery=True)
        img2 = self._sample(idx, isQuery=False, isPositive=True)
        mask2 = self._sample_mask(idx)
        if random.random() >= 0.5:
            img3 = self._sample(idx, isQuery=False, isPositive=False, isDiff_obj=True)
        else:
            _, img3, _ = self._sample(idx, isQuery=False, isPositive=False, isDiff_obj=False)

        # sample fourth image for regression
        if random.random() >= 0.5:  # different pose
            delta, img4, mask4 = self._sample(idx, isQuery=False, isPositive=False, isDiff_obj=False)
        else:  # same pose
            mask4 = self._sample_mask(idx)
            delta, img4 = 0, img2.copy()
        if save_path is None:
            mask2 = image_utils.process_mask_image(mask2, mask_size=self.mask_size)
            mask4 = image_utils.process_mask_image(mask4, mask_size=self.mask_size)
            return [self.im_transform(img1)], [self.im_transform(img2), mask2], \
                   [self.im_transform(img3)], [self.im_transform(img4), mask4, delta]
        else:
            for i, img in enumerate([img1, img2, mask2, img3, img4, mask4]):
                img.save(save_path + "_sample_{}.png".format(i))

    def __getitem__(self, idx):
        id_obj = self.query_data.iloc[idx]['id_obj']
        id_symmetry = inout.list_all_id_symmetry[id_obj]
        obj_pose = torch.from_numpy(np.asarray(self.query_data.iloc[idx]["real_location"]))
        if not self.split == "train":
            query = self._sample(idx, isQuery=True)
            query = self.im_transform(query)
            return dict(id_obj=id_obj, id_symmetry=id_symmetry, obj_pose=obj_pose, query=query)
        else:
            first_data, second_data, third_data, fourth_data = self._sample_triplet(idx)
            return dict(id_obj=id_obj, id_symmetry=id_symmetry, obj_pose=obj_pose, query=first_data[0],
                        template=second_data[0], mask=second_data[1], negative_random=third_data[0],
                        negative_same_obj=fourth_data[0], mask_negative_same_obj=fourth_data[1],
                        negative_same_obj_delta=fourth_data[2])

    def save_random_sequences(self):
        len_data = self.__len__()
        list_index = np.unique(np.random.randint(0, len_data, 10))
        print("Saving samples at {}".format(self.save_path))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for idx in list_index:
            save_path = os.path.join(self.save_path, "{:06d}".format(idx))
            if self.split == "train":
                self._sample_triplet(idx, save_path)
            else:
                query = self._sample(idx, isQuery=True)
                query.save(save_path + "_test.png")


if __name__ == '__main__':
    from lib.utils.config import Config

    config_global = Config(config_file="./config.json").get_config()
    # save_dir = "./draft/LINEMOD"
    # LINEMOD(root_dir=config_global.root_path, dataset="LINEMOD",
    #         list_id_obj=list(range(0, 9)), split="train", image_size=224, save_path=save_dir, is_master=True)

    save_dir = "./draft/occlusionLINEMOD"
    LINEMOD(root_dir=config_global.root_path, dataset="occlusionLINEMOD",
            list_id_obj=[0, 3], split="train", image_size=224, save_path=save_dir, is_master=True)
    # to test: python -m lib.datasets.linemod.dataloader_query
