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
from src.poses.utils import get_obj_poses_from_template_level

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
):
    output_dir = list_output_dir[id_obj]
    cad_path = list_cad_path[id_obj]
    if os.path.exists(
        output_dir
    ):  # remove first to avoid the overlapping folder of blender proc
        os.system("rm -r {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # command = f"blenderproc run ./src/poses/blenderproc.py {cad_path} {obj_pose_path} {output_dir} {gpus_devices}"
    command = f"python -m src.poses.pyrender {cad_path} {obj_pose_path} {output_dir} {gpus_devices}"
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
    save_dir = osp.join(osp.dirname(cfg.data.lm.root_dir), "templates_pyrender")

    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4  # zoom to object
    os.makedirs(f"{save_dir}/{cfg.dataset_to_render}", exist_ok=True)
    obj_pose_path = f"{save_dir}/{cfg.dataset_to_render}/obj_poses.npy"
    np.save(obj_pose_path, template_poses)

    if cfg.dataset_to_render in ["tless"]:
        cad_dir = os.path.join(cfg.data["tless_test"].root_dir, "models/models_cad")
    else:
        cad_dir = os.path.join(
            cfg.data[cfg.dataset_to_render].root_dir, "models/models"
        )
    cad_paths = []
    output_dirs = []
    object_ids = sorted(
        [
            int(name[4:][:-4])
            for name in os.listdir(cad_dir)
            if name.endswith(".ply") and not name.endswith("old.ply")
        ]
    )
    for object_id in object_ids:
        cad_paths.append(
            os.path.join(
                cad_dir,
                "obj_{:06d}.ply".format(object_id),
            )
        )
        # hope and ycbv cad format is different which make the render is always black
        # use obj format instead (make sure you have used python -m src.scripts.process_mesh to convert ply to obj)
        # if cfg.dataset_to_render in ["hope", "ycbv"]:
        #     cad_paths[-1] = cad_paths[-1].replace(".ply", ".obj")

        output_dirs.append(
            os.path.join(
                save_dir,
                f"{cfg.dataset_to_render}/obj_{object_id:06d}",
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
    )
    mapped_values = list(
        tqdm(
            pool.imap_unordered(call_blender_proc_with_index, range(len(object_ids))),
            total=len(object_ids),
        )
    )
    finish_time = time.time()
    print("Total time to render templates for query:", finish_time - start_time)


if __name__ == "__main__":
    render()
