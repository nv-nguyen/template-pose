import os
import argparse
from tqdm import tqdm
import numpy as np
import glob
import time
from functools import partial
import multiprocessing
import json
from lib.utils.config import Config
from lib.datasets.linemod import inout, visualization
from lib.datasets.linemod.processing_utils import read_template_poses, find_best_template
from lib.poses import utils

names = ["id_obj", "id_symmetry",
         "real_path", "real_location",
         "test_template_path", "test_template_location", "best_test_error",
         "train_template_path", "train_template_location", "best_train_error"]
test_template_poses = read_template_poses(split="test", opengl_camera=True)
train_template_poses = read_template_poses(split="train", opengl_camera=True)


def search_nearest_neighbors(idx_obj, dataset, config, crop_dir):
    data = [[] for _ in range(len(names))]
    if dataset == "LINEMOD":
        obj_name = inout.LINEMOD_names[idx_obj]
        id_obj = int(inout.LINEMOD_real_ids[idx_obj])
        dataset_path = os.path.join("LINEMOD/objects", obj_name, "rgb/")
        num_real_frames = len(glob.glob(os.path.join(config.root_path, "linemod", dataset_path, "*.jpg")))
    else:
        id_obj = int(inout.occlusion_real_ids[idx_obj])
        obj_name = inout.occlusion_LINEMOD_names[idx_obj]
        dataset_path = os.path.join("occlusionLINEMOD/RGB-D/rgb_noseg/")
        num_real_frames = len(glob.glob(os.path.join(config.root_path, "linemod", dataset_path, "*.png")))
    test_template_dir = os.path.join("templatesLINEMOD", "test", obj_name)
    train_template_dir = os.path.join("templatesLINEMOD", "train", obj_name)
    print(dataset, id_obj, obj_name)
    for id_frame in tqdm(range(num_real_frames)):
        pose = inout.read_opencv_pose_linemod(root_dir=config.root_path,
                                              dataset=dataset,
                                              idx_obj=idx_obj, id_frame=id_frame)
        if pose is not None:  # in occlusionLINEMOD, some objects are heavy occluded
            pose = utils.remove_inplane_rotation(pose)
            pose = utils.opencv2opengl(pose)
            idx_test_template, test_error = find_best_template(query_pose=np.copy(pose),
                                                               template_poses=np.copy(test_template_poses),
                                                               id_symmetry=inout.list_id_symmetry[idx_obj],
                                                               base_on_angle=False)

            idx_train_template, train_error = find_best_template(query_pose=np.copy(pose),
                                                                 template_poses=np.copy(train_template_poses),
                                                                 id_symmetry=inout.list_id_symmetry[idx_obj],
                                                                 base_on_angle=False)

            real_path = os.path.join(obj_name, "{:06d}.png".format(id_frame))
            real_location = pose[2, :3].tolist()

            test_template_path = os.path.join(test_template_dir, "{:06d}.png".format(idx_test_template))
            test_template_location = test_template_poses[idx_test_template, 2, :3].tolist()

            train_template_path = os.path.join(train_template_dir, "{:06d}.png".format(idx_train_template))
            train_template_location = train_template_poses[idx_train_template, 2, :3].tolist()

            row = [id_obj, inout.list_id_symmetry[idx_obj],
                   real_path, real_location,
                   test_template_path, test_template_location, test_error,
                   train_template_path, train_template_location, train_error]
            for i in range(len(names)):
                data[i].append(row[i])
            if id_frame % 50 == 0:
                visualization.visualize_gt_templates(crop_dir=crop_dir, dataset=dataset,
                                                     obj_name=obj_name, idx_frame=id_frame,
                                                     idx_train_template=idx_train_template,
                                                     idx_test_template=idx_test_template,
                                                     train_error=train_error, test_error=test_error)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to create dataframe of Contrast Learning for Template Matching')
    parser.add_argument('--config', type=str, default="./config.json")
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    config = Config(config_file=args.config).get_config()

    pool = multiprocessing.Pool(processes=args.num_workers)
    crop_dir = os.path.join(config.root_path, "crop_image224")

    for dataset in ["LINEMOD", "occlusionLINEMOD"]:
        if dataset == "LINEMOD":
            list_index = range(13)
        elif dataset == "occlusionLINEMOD":
            list_index = range(8)
        search_nearest_neighbors_with_index = partial(search_nearest_neighbors, dataset=dataset, config=config,
                                                      crop_dir=crop_dir)

        start_time = time.time()
        mapped_values = list(
            tqdm(pool.imap_unordered(search_nearest_neighbors_with_index, list_index), total=len(list_index)))
        finish_time = time.time()
        print("Total time to create dataframe for {}:".format(dataset), finish_time - start_time)

        nearest_neigh_data = [[] for _ in range(len(names))]
        for data in mapped_values:
            for idx_frame in range(len(data[0])):
                for i in range(len(names)):
                    nearest_neigh_data[i].append(data[i][idx_frame])

        data_frame = {names[i]: nearest_neigh_data[i] for i in range(len(names))}
        with open(os.path.join(config.root_path, dataset + ".json"), 'w') as f:
            json.dump(data_frame, f, indent=4)
