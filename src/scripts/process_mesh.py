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
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from src.utils.trimesh_utils import load_mesh
import trimesh


def convert_ply_to_obj(idx, ply_paths, obj_paths):
    ply_path = ply_paths[idx]
    obj_path = obj_paths[idx]

    # open mesh
    mesh = load_mesh(ply_path)
    texture = mesh.visual.material.image
    vertex_colors = trimesh.visual.uv_to_color(mesh.visual.uv, texture)
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors
    )
    new_mesh.export(obj_path)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="render",
)
def render(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    start_time = time.time()
    # convert mesh format of ycbv or hope from textured CAD to vertex color CAD
    for dataset_name in ["hope"]:
        cad_dir = os.path.join(cfg.data[dataset_name].root_dir, "models/models")
        cad_names = sorted(
            [name for name in os.listdir(cad_dir) if name.endswith(".ply")]
        )
        save_paths = [
            osp.join(cad_dir, f"{cad_path[:-4]}.obj") for cad_path in cad_names
        ]
        cad_paths = [osp.join(cad_dir, name) for name in cad_names]

        convert_ply_to_obj_with_index = partial(
            convert_ply_to_obj, ply_paths=cad_paths, obj_paths=save_paths
        )
        pool = multiprocessing.Pool(processes=1)
        mapped_values = list(
            tqdm(
                pool.imap_unordered(
                    convert_ply_to_obj_with_index, range(len(cad_paths))
                ),
                total=len(cad_paths),
            )
        )
        finish_time = time.time()
        print("Total time to convert from .ply to .obj :", finish_time - start_time)


if __name__ == "__main__":
    # set level logging
    logging.basicConfig(level=logging.INFO)
    render()
