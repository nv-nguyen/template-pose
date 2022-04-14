import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
from functools import partial
import multiprocessing
import argparse
from lib.utils.config import Config
from lib.datasets.linemod.inout import LINEMOD_name_to_real_id


def call_blender_proc(id_obj, list_cad_path, list_output_dir, list_obj_poses, disable_output):
    output_dir = list_output_dir[id_obj]
    cad_path = list_cad_path[id_obj]
    obj_poses = list_obj_poses[id_obj]
    if os.path.exists(output_dir):  # remove first to avoid the overlapping folder of blender proc
        os.system("rm -r {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pose_path = os.path.join(output_dir, "{:06d}".format(id_obj))
    np.save(pose_path, obj_poses)

    command = "blenderproc run ./lib/renderer/blenderproc.py {} {} {}".format(cad_path,
                                                                              pose_path + ".npy", output_dir)
    if disable_output:
        command += " true"
    else:
        command += " false"
    os.system(command)
    # remove all .npy files in output folder
    os.system("rm {}".format(output_dir + '/*.npy'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Template Matching render template scripts')
    parser.add_argument('--config', type=str, default="./config.json")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--disable_output', action='store_true', help="Disable output of blender")
    args = parser.parse_args()

    config = Config(config_file=args.config).get_config()
    pool = multiprocessing.Pool(processes=args.num_workers)
    if args.dataset == "linemod":
        list_obj = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                    'iron', 'lamp', 'phone']
        for split in ["train", "test"]:
            if split == "test":
                obj_pose = np.load("./lib/poses/predefined_poses/half_sphere_level2.npy")
            else:
                obj_pose = np.load("./lib/poses/predefined_poses/half_sphere_level2_and_level3.npy")
            cad_paths = []
            output_dirs = []
            poses = []
            for obj in list_obj:
                id_object = LINEMOD_name_to_real_id[obj]
                cad_paths.append(os.path.join(config.root_path,
                                              "linemod/models/models/obj_{:06d}.ply".format(id_object + 1)))
                output_dirs.append(os.path.join(config.root_path, "templates/linemod/{}/{}".format(split, obj)))
                poses.append(obj_pose)

            start_time = time.time()
            call_blender_proc_with_index = partial(call_blender_proc,
                                                   list_cad_path=cad_paths,
                                                   list_output_dir=output_dirs,
                                                   list_obj_poses=poses, disable_output=args.disable_output)
            mapped_values = list(
                tqdm(pool.imap_unordered(call_blender_proc_with_index, range(len(list_obj))), total=len(list_obj)))
            finish_time = time.time()
            print("Total time to render templates for LINEMOD:", finish_time - start_time)

    elif args.dataset == "tless":
        obj_pose = np.load("./lib/poses/predefined_poses/sphere_level2.npy")
        # to use level 3: obj_pose = np.load("../lib/poses/predefined_poses/sphere_level3.npy")
        cad_paths = []
        output_dirs = []
        poses = []
        list_obj = range(1, 31)
        for id_object in list_obj:
            cad_paths.append(os.path.join(config.root_path,
                                          "tless/models/models_cad/obj_{:02d}.ply".format(id_object)))
            output_dirs.append(os.path.join(config.root_path, "templates/tless/{:02d}".format(id_object)))
            poses.append(obj_pose)

        start_time = time.time()
        call_blender_proc_with_index = partial(call_blender_proc,
                                               list_cad_path=cad_paths,
                                               list_output_dir=output_dirs,
                                               list_obj_poses=poses, disable_output=args.disable_output)
        mapped_values = list(
            tqdm(pool.imap_unordered(call_blender_proc_with_index, range(len(list_obj))), total=len(list_obj)))
        finish_time = time.time()
        print("Total time to render templates for T-Less:", finish_time - start_time)
