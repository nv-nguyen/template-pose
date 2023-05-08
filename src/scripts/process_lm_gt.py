import logging
import os, sys
import os.path as osp

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from src.dataloader.process_lm_utils import (
    process_poses,
    process_folder,
    process_occlusionLM,
)
from src.utils.inout import get_root_project
from functools import partial
import multiprocessing
from tqdm import tqdm
import time
from src.dataloader.lm_utils import query_real_ids


def extract_openCV_pose(save_path):
    root_repo = get_root_project()
    tmp_path = f"{root_repo}/src/dataloader/opencv_pose.zip"
    logging.info(f"Download {tmp_path} to {save_path}")
    os.system(f"unzip {tmp_path} -d {save_path}")


def process_query_with_idx(
    idx, list_obj_root_dir, list_obj_pose_path, list_obj_save_path
):
    lm_real_ids = np.asarray([0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14])
    obj_root_dir = list_obj_root_dir[idx]
    obj_pose_path = list_obj_pose_path[idx]
    obj_save_path = list_obj_save_path[idx]
    os.makedirs(obj_root_dir, exist_ok=True)
    poses = process_poses(obj_pose_path, id_obj=lm_real_ids[idx])
    process_folder(input_dir=obj_root_dir, save_dir=obj_save_path, poses=poses)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="render",
)
def process_gt(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    pool = multiprocessing.Pool(processes=8)

    query_names = cfg.data.lm.obj_names.split(", ")
    occlusion_query_names = cfg.data.olm.obj_names.split(", ")
    # process query
    dataset_pose_path = osp.join(cfg.data.lm.root_dir, "opencv_pose", "LINEMOD")
    if not osp.exists(dataset_pose_path):
        extract_openCV_pose(cfg.data.lm.root_dir)
    cfg.data.lm.test_dir = osp.join(cfg.data.lm.root_dir, "test")

    list_obj_root_dir, list_obj_pose_path, list_obj_save_path = [], [], []
    for obj_id, obj_name in zip(query_real_ids, query_names):
        obj_root_dir = osp.join(cfg.data.lm.root_dir, "objects", obj_name)
        list_obj_root_dir.append(obj_root_dir)
        obj_pose_path = osp.join(dataset_pose_path, f"{obj_name}.json")
        list_obj_pose_path.append(obj_pose_path)
        obj_save_path = osp.join(cfg.data.lm.test_dir, f"{obj_id+1:06d}")
        list_obj_save_path.append(obj_save_path)
    start_time = time.time()
    process_query_with_idx_p = partial(
        process_query_with_idx,
        list_obj_root_dir=list_obj_root_dir,
        list_obj_pose_path=list_obj_pose_path,
        list_obj_save_path=list_obj_save_path,
    )
    mapped_values = list(
        tqdm(
            pool.imap_unordered(process_query_with_idx_p, range(len(query_names))),
            total=len(query_names),
        )
    )
    finish_time = time.time()
    logging.info(f"Total time for query: {finish_time - start_time}")

    # process occlusion query
    dataset_pose_path = osp.join(
        cfg.data.lm.root_dir, "opencv_pose", "occlusionLINEMOD"
    )
    start_time = time.time()
    os.makedirs(osp.join(cfg.data.olm.root_dir, "test"), exist_ok=True)
    process_occlusionLM(
        cfg.data.olm.root_dir,
        dataset_pose_path,
        osp.join(cfg.data.olm.root_dir, "test/000000"),
        query_names,
        occlusion_query_names,
    )
    finish_time = time.time()
    logging.info(f"Total time for occlusionquery: {finish_time - start_time}")


if __name__ == "__main__":
    process_gt()
