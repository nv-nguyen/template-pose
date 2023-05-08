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
from src.utils.inout import write_txt


def manual_formatting(save_path, vertices, faces, vertex_colors):
    new_texts = [
        "ply",
        "format ascii 1.0",
        "comment Created by Blender 2.77 (sub 0) - www.blender.org, source file: ''",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        # "property float nx",
        # "property float ny",
        # "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar uint vertex_indices",
        "end_header",
    ]
    assert len(vertices) == len(vertex_colors)
    for i in range(len(vertices)):
        new_texts.append(
            f"{vertices[i][0]} {vertices[i][1]} {vertices[i][2]} {vertex_colors[i][0]} {vertex_colors[i][1]} {vertex_colors[i][2]}"
        )
    for i in range(len(faces)):
        new_texts.append(f"3 {faces[i][0]} {faces[i][1]} {faces[i][2]}")
    write_txt(save_path, new_texts)
    print(f"Finish formatting {save_path}")


def convert_ply_to_obj(idx, ply_paths):
    ply_path = ply_paths[idx]

    # open mesh
    mesh = load_mesh(ply_path)
    texture = mesh.visual.material.image
    vertex_colors = trimesh.visual.uv_to_color(mesh.visual.uv, texture)
    # rename the older one and the new_one
    os.rename(ply_path, ply_path.replace(".ply", "_old.ply"))
    manual_formatting(ply_path, mesh.vertices, mesh.faces, vertex_colors=vertex_colors)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="render",
)
def render(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    start_time = time.time()
    # convert mesh format of ycbv or hope from textured CAD to vertex color CAD
    for dataset_name in ["hope", "ruapc"]:
        cad_dir = os.path.join(cfg.data[dataset_name].root_dir, "models/models")
        cad_names = sorted(
            [name for name in os.listdir(cad_dir) if name.endswith(".ply") and not name.endswith("_old.ply")]
        )
        cad_paths = [osp.join(cad_dir, name) for name in cad_names]

        convert_ply_to_obj_with_index = partial(convert_ply_to_obj, ply_paths=cad_paths)
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
