import os
import argparse
from tqdm import tqdm
import numpy as np
import glob
import time
from functools import partial
import multiprocessing
import logging

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.inout import get_root_project
from functools import partial
import multiprocessing
from tqdm import tqdm
import time

import os.path as osp
from src.poses.utils import NearestTemplateFinder, combine_R_and_T
from src.dataloader.base import BaseBOP
from src.utils.inout import save_json


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="render",
)
def compute_neighbors(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    for data_cfg in cfg.data.values():
        logging.info(f"Compute neighbors template for {data_cfg.dataset_name}")
        splits = os.listdir(data_cfg.root_dir)
        splits = [
            split
            for split in splits
            if split.startswith("train")
            or split.startswith("val")
            or split.startswith("test")
        ]
        for split in splits:
            start_time = time.time()
            finder = NearestTemplateFinder(
                level_templates=2,
                pose_distribution="all",
                return_inplane=True,
            )
            bop_dataset = BaseBOP(root_dir=data_cfg.root_dir, split=split)
            bop_dataset.load_list_scene(split=split)
            for scene_path in tqdm(bop_dataset.list_scenes):
                templates_infos = {}
                scene_data = bop_dataset.load_scene(scene_path)
                save_template_path = osp.join(scene_path, f"template_level2.json")
                rgbs_path = scene_data["rgb_paths"]
                for idx_frame in range(len(rgbs_path)):
                    rgb_path = scene_data["rgb_paths"][idx_frame]
                    id_frame = int(str(rgb_path).split("/")[-1].split(".")[0])
                    frame_poses = (
                        scene_data["scene_gt"][f"{id_frame}"]
                        if f"{id_frame}" in scene_data["scene_gt"]
                        else scene_data["scene_gt"][id_frame]
                    )
                    frame_poses = (
                        frame_poses if isinstance(frame_poses, list) else [frame_poses]
                    )
                    cad_ids = [x["obj_id"] for x in frame_poses]
                    cad_poses = np.array(
                        [
                            combine_R_and_T(x["cam_R_m2c"], x["cam_t_m2c"])
                            for x in frame_poses
                        ]
                    )
                    idx_templates, inplanes = finder.search_nearest_template(cad_poses)
                    templates_infos[f"{id_frame}"] = [
                        {
                            "obj_id": cad_ids[idx_obj],
                            "idx_template": int(idx_templates[idx_obj]),
                            "inplane": float(inplanes[idx_obj]),
                        }
                        for idx_obj in range(len(cad_ids))
                    ]
                save_json(save_template_path, templates_infos)
            logging.info(f"Time to compute neighbors: {time.time() - start_time}")


if __name__ == "__main__":
    compute_neighbors()
