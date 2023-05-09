import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
from functools import partial
import multiprocessing
import logging
import os, sys
import os.path as osp

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from src.scripts.render_template import call_blender_proc


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="render",
)
def render(cfg: DictConfig) -> None:
    
    OmegaConf.set_struct(cfg, False)
    save_dir = osp.join(osp.dirname(cfg.data.lm.root_dir), "templates")
    dataset_name = cfg.dataset_to_render
    if dataset_name == "tless":
        data_cfg = cfg.data["tless_test"]
    else:
        data_cfg = cfg.data[dataset_name]
    obj_pose_path = f"{save_dir}/{dataset_name}/obj_poses.npy"
    
    logging.info(f"Checking {data_cfg.dataset_name} ...")
    if dataset_name in ["tless"]:
        cad_dir = os.path.join(data_cfg.root_dir, "models/models_cad")
    else:
        cad_dir = os.path.join(data_cfg.root_dir, "models/models")
    object_ids = sorted(
        [
            int(name[4:][:-4])
            for name in os.listdir(cad_dir)
            if name.endswith(".ply") and not name.endswith("old.ply")
        ]
    )
    cad_paths, template_dirs = [], []
    for object_id in object_ids:
        cad_path = os.path.join(
            cad_dir,
            "obj_{:06d}.ply".format(object_id),
        )
        template_dir = os.path.join(save_dir, f"{dataset_name}/obj_{object_id:06d}")
        num_templates = len([file for file in os.listdir(template_dir) if file.endswith(".png")])
        if num_templates != 642:
            logging.info(
                f"Dataset {data_cfg.dataset_name}, obj {object_id} failed, found only {num_templates}"
            )
            cad_paths.append(cad_path)
            template_dirs.append(template_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    
    start_time = time.time()
    pool = multiprocessing.Pool(processes=int(cfg.num_workers))
    call_blender_proc_with_index = partial(
        call_blender_proc,
        list_cad_path=cad_paths,
        list_output_dir=template_dirs,
        obj_pose_path=obj_pose_path,
        disable_output=cfg.disable_output,
        gpus_devices=cfg.gpus,
    )
    mapped_values = list(
        tqdm(
            pool.imap_unordered(
                call_blender_proc_with_index, range(len(template_dirs))
            ),
            total=len(template_dirs),
        )
    )
    finish_time = time.time()
    print(f"Total time to render templates for query: {finish_time - start_time}")


if __name__ == "__main__":
    render()
