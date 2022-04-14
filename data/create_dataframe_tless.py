from lib.utils.config import Config
import os
import argparse
from tqdm import tqdm
import numpy as np
import time
from functools import partial
import multiprocessing
import json
from lib.datasets.tless import inout, processing_utils, visualization

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

template_poses = inout.read_template_poses(opengl_camera=True)
names = ['split', 'id_obj', 'id_scene', 'id_frame', 'idx_frame', 'rgb_path', 'depth_path', "visib_fract",
         'idx_obj_in_scene', 'cam_R_m2c', 'cam_t_m2c', 'cam_K', 'depth_scale', 'gt_inplane', 'nearest_inplane',
         'index_nearest_template', 'best_error']


def search_nearest_neighbors(id_obj, split, config):
    data = [[] for _ in range(len(names))]
    pose_path = os.path.join(config.root_path, "tless/opencv_pose/", "{:02d}_{}.json".format(id_obj, split))
    with open(pose_path, 'r') as f:
        gt_data = yaml.load(f, Loader=yaml.CLoader)

    num_real_frames = len(gt_data)
    for idx_frame in tqdm(range(num_real_frames)):
        [id_scene, id_frame, rgb_path, depth_path, visib_fract, idx_obj_in_scene, cam_R_m2c, cam_t_m2c, \
         cam_K, depth_scale], pose = inout.read_opencv_pose_tless(root_dir=config.root_path, dataset="tless",
                                                                  split=split, id_obj=id_obj,
                                                                  idx_frame=idx_frame, all=True)

        gt_inplane, index_nearest_template, best_error = processing_utils.find_best_template(query_opencv=np.copy(pose),
                                                                                             templates_opengl=template_poses)
        nearest_inplane = processing_utils.find_nearest_inplane(gt_inplane, bin_size=10)

        row = [split, id_obj, id_scene, id_frame, idx_frame, rgb_path, depth_path, visib_fract,
               idx_obj_in_scene, cam_R_m2c, cam_t_m2c, cam_K, depth_scale, gt_inplane.tolist(),
               nearest_inplane.tolist(),
               index_nearest_template, best_error.tolist()]

        for i in range(len(names)):
            data[i].append(row[i])

        if id_frame % 50 == 0:
            visualization.visualization_gt_templates(root_path=config.root_path,
                                                     split=split,
                                                     id_scene=id_scene,
                                                     id_frame=id_frame,
                                                     id_obj=id_obj,
                                                     idx_obj_in_scene=idx_obj_in_scene,
                                                     index_nearest_template=index_nearest_template,
                                                     gt_inplane=gt_inplane,
                                                     nearest_inplane=nearest_inplane,
                                                     best_error=best_error)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to create dataframe of Contrast Learning for Template Matching')
    parser.add_argument('--config', type=str, default="./config.json")
    parser.add_argument('--split', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    config = Config(config_file=args.config).get_config()
    pool = multiprocessing.Pool(processes=args.num_workers)
    list_index = range(1, 31)

    search_nearest_neighbors_with_index = partial(search_nearest_neighbors, split=args.split, config=config)

    start_time = time.time()
    mapped_values = list(
        tqdm(pool.imap_unordered(search_nearest_neighbors_with_index, list_index), total=len(list_index)))
    finish_time = time.time()
    print("Total time to create dataframe for T-LESS", finish_time - start_time)

    nearest_neigh_data = [[] for _ in range(len(names))]
    for data in mapped_values:
        for idx_frame in range(len(data[0])):
            for idx_name in range(len(names)):
                nearest_neigh_data[idx_name].append(data[idx_name][idx_frame])

    data_frame = {names[i]: nearest_neigh_data[i] for i in range(len(names))}
    with open(os.path.join(config.root_path, "tless_{}.json".format(args.split)), 'w') as f:
        json.dump(data_frame, f, indent=4)
