import glob
import numpy as np
import sys
import json
from tqdm import tqdm
import os
import argparse
from lib.utils.config import Config
from lib.poses import utils
from lib.datasets.linemod import inout, processing_utils
from lib.utils.inout_BOPformat import save_info


def process_linemod(linemod_path, save_dir):
    if not os.path.exists(os.path.join(save_dir, "LINEMOD")):
        os.makedirs(os.path.join(save_dir, "LINEMOD"))
    for id_obj in tqdm(range(len(inout.LINEMOD_names))):
        all_poses = {}
        pose_obj_path = os.path.join(linemod_path, "objects", inout.LINEMOD_names[id_obj], "pose")
        num_pose = len(glob.glob(os.path.join(pose_obj_path, "*.txt")))
        for id_frame in tqdm(range(num_pose)):
            pose = inout.read_original_pose_linemod(linemod_dir=linemod_path,
                                                    id_obj=id_obj,
                                                    id_frame=id_frame)
            # we have transformed from BOP to current mesh so we need to inverse it
            offset = utils.inverse_matrix_world(offset_bop_linemod[id_obj])
            pose = pose.dot(offset)  # inverse transform of mesh
            all_poses[id_frame] = {'cam_R_w2c': pose[:3, :3], 'cam_t_w2c': pose[:3, 3]}
        save_info(os.path.join(save_dir, "LINEMOD", "{}.json".format(inout.LINEMOD_names[id_obj])), all_poses)


def process_occlusion_linemod(linemod_occlusion_path, save_dir):
    occlusion_name_to_id = {
        'Ape': 0,
        'Can': 3,
        'Cat': 4,
        'Driller': 5,
        'Duck': 6,
        'Eggbox': 7,
        'Glue': 8,
        'Holepuncher': 9}
    occlusion_real_id_to_Name = {v: k for k, v in occlusion_name_to_id.items()}

    if not os.path.exists(os.path.join(save_dir, "occlusionLINEMOD")):
        os.makedirs(os.path.join(save_dir, "occlusionLINEMOD"))
    for idx_obj in tqdm(range(len(inout.occlusion_real_ids))):
        id_obj = inout.occlusion_real_ids[idx_obj]
        obj_path = os.path.join(linemod_occlusion_path, "poses", occlusion_real_id_to_Name[id_obj])
        num_pose = len(glob.glob(os.path.join(obj_path, "*.txt")))
        all_poses = {}
        for id_frame in tqdm(range(num_pose)):
            pose = inout.read_original_pose_occlusion_linemod(occlusion_dir=linemod_occlusion_path,
                                                              id_obj=id_obj,
                                                              id_frame=id_frame)
            if pose is not None:
                pose = pose.dot(utils.inverse_matrix_world(
                    offset_bop_occlusionLinemod[idx_obj]))  # inverse transform of mesh
                pose = utils.opencv2opengl(pose)  # change opengl to opencv
                all_poses[id_frame] = {'cam_R_w2c': pose[:3, :3], 'cam_t_w2c': pose[:3, 3]}
        save_info(os.path.join(save_dir, "occlusionLINEMOD", "{}.json".format(inout.LINEMOD_names[id_obj])), all_poses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert pose convention of LINEMOD and OcclusionLINEMOD')
    parser.add_argument('--config', type=str, default="./config.json")
    args = parser.parse_args()

    config = Config(config_file=args.config).get_config()
    save_directory = os.path.join(config.root_path, "linemod/opencv_pose")

    offset_bop_linemod = np.zeros((13, 4, 4))
    for idx, id_obj in enumerate(inout.LINEMOD_real_ids):
        offset_bop_linemod[idx] = processing_utils.get_transformation_LINEMOD(
            bop_path=os.path.join(config.root_path, config.LINEMOD.cad_path, "models"),
            linemod_path=os.path.join(config.root_path, config.LINEMOD.local_path, "models"), id_obj=id_obj)
    process_linemod(os.path.join(config.root_path, config.LINEMOD.local_path), save_directory)

    offset_bop_occlusionLinemod = np.zeros((8, 4, 4))
    for idx, id_obj in enumerate(inout.occlusion_real_ids):
        offset_bop_occlusionLinemod[idx] = processing_utils.get_transformation_occlusionLINEMOD(idx_obj=idx)
    process_occlusion_linemod(os.path.join(config.root_path, config.occlusionLINEMOD.local_path), save_directory)