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
from src.poses.utils import get_obj_poses_from_template_level, get_root_project
from src.dataloader.lm_utils import query_name_to_real_id

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np


def call_blender_proc(
    id_obj,
    list_cad_path,
    list_output_dir,
    obj_pose_path,
    disable_output,
    gpus_devices,
    custom_blender_path,
):
    output_dir = list_output_dir[id_obj]
    cad_path = list_cad_path[id_obj]
    if os.path.exists(
        output_dir
    ):  # remove first to avoid the overlapping folder of blender proc
        os.system("rm -r {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    command = f"blenderproc run ./src/poses/blenderproc.py {cad_path} {obj_pose_path} {output_dir} {gpus_devices}"
    if custom_blender_path is not None:
        command += f" --custom-blender-path {custom_blender_path}"
    if disable_output:
        command += " true"
    else:
        command += " false"
    os.system(command)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="render",
)
def render(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    save_dir = osp.join(osp.dirname(cfg.data.lm.root_dir), "templates")

    # query
    if cfg.dataset_to_render in ["lm", "all"]:
        template_poses = get_obj_poses_from_template_level(
            level=2, pose_distribution="upper"
        )
        template_poses[:, :3, 3] *= 0.4  # zoom to object
        os.makedirs(f"{save_dir}/query", exist_ok=True)
        obj_pose_path = f"{save_dir}/query/obj_poses.npy"
        np.save(obj_pose_path, template_poses)

        cad_paths = []
        output_dirs = []
        query_names = cfg.data.lm.obj_names.split(", ")
        for obj in query_names:
            id_object = query_name_to_real_id[obj]
            cad_paths.append(
                os.path.join(
                    cfg.data.lm.root_dir,
                    "models/models/obj_{:06d}.ply".format(id_object + 1),
                )
            )
            output_dirs.append(
                os.path.join(
                    osp.dirname(cfg.data.lm.root_dir),
                    "templates/query/obj_{:06d}".format(id_object + 1),
                )
            )
            os.makedirs(osp.dirname(output_dirs[-1]), exist_ok=True)

        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
        start_time = time.time()
        pool = multiprocessing.Pool(processes=int(cfg.num_workers))
        call_blender_proc_with_index = partial(
            call_blender_proc,
            list_cad_path=cad_paths,
            list_output_dir=output_dirs,
            obj_pose_path=obj_pose_path,
            disable_output=cfg.disable_output,
            gpus_devices=cfg.gpus,
            custom_blender_path=cfg.custom_blender_path,
        )
        mapped_values = list(
            tqdm(
                pool.imap_unordered(
                    call_blender_proc_with_index, range(len(query_names))
                ),
                total=len(query_names),
            )
        )
        finish_time = time.time()
        print("Total time to render templates for query:", finish_time - start_time)

    # TLESS
    if cfg.dataset_to_render in ["tless", "all"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
        start_time = time.time()
        pool = multiprocessing.Pool(processes=cfg.num_workers)
        template_poses = get_obj_poses_from_template_level(
            level=2, pose_distribution="all"
        )
        template_poses[:, :3, 3] *= 0.4  # zoom to object
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/tless", exist_ok=True)
        obj_pose_path = f"{save_dir}/tless/obj_poses.npy"
        np.save(obj_pose_path, template_poses)

        cad_paths = []
        output_dirs = []
        list_obj = range(1, 31)
        for id_object in list_obj:
            cad_paths.append(
                osp.join(
                    cfg.data.tless_test.root_dir,
                    f"models/models_cad/obj_{id_object:06d}.ply",
                )
            )
            output_dirs.append(
                osp.join(
                    osp.dirname(cfg.data.lm.root_dir),
                    f"templates/tless/obj_{id_object:06d}",
                )
            )

        start_time = time.time()
        call_blender_proc_with_index = partial(
            call_blender_proc,
            list_cad_path=cad_paths,
            list_output_dir=output_dirs,
            obj_pose_path=obj_pose_path,
            disable_output=cfg.disable_output,
            gpus_devices=cfg.gpus,
            custom_blender_path=cfg.custom_blender_path,
        )
        mapped_values = list(
            tqdm(
                pool.imap_unordered(call_blender_proc_with_index, range(len(list_obj))),
                total=len(list_obj),
            )
        )
        finish_time = time.time()
        print("Total time to render templates for T-Less:", finish_time - start_time)


if __name__ == "__main__":
    render()
