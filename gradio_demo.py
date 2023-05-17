import fire
import gradio as gr
from functools import partial
from PIL import Image
import numpy as np
import os
import glob
import argparse
from omegaconf import DictConfig, OmegaConf

WEBSITE = """
<h1 style='text-align: center'>Templates for 3D Object Pose Estimation Revisited: <br>
                Generalization to New Objects and Robustness to Occlusions</br> </h1>

<h3 style='text-align: center'>
<a href="https://nv-nguyen.github.io/" target="_blank"><nobr>Van Nguyen Nguyen</nobr></a> &emsp;
<a href="https://yinlinhu.github.io/" target="_blank"><nobr>Yinlin Hu</nobr></a> &emsp;
<a href="https://youngxiao13.github.io/" target="_blank"><nobr>Yang Xiao</nobr></a> &emsp;
<a href="https://people.epfl.ch/mathieu.salzmann" target="_blank"><nobr>Mathieu Salzmann</nobr></a> &emsp;
<a href="https://vincentlepetit.github.io/" target="_blank"><nobr>Vincent Lepetit</nobr></a>
</h3>

<h3 style='text-align: center'>
<nobr>CVPR 2022</nobr>
</h3>

<h3 style="text-align:center;">
<a target="_blank" href="https://arxiv.org/abs/2203.17234"> <button type="button" class="btn btn-primary btn-lg"> Paper </button></a>
<a target="_blank" href="https://github.com/nv-nguyen/template-pose"> <button type="button" class="btn btn-primary btn-lg"> Github </button></a>
<a target="_blank" href="https://nv-nguyen.github.io/template-pose"> <button type="button" class="btn btn-primary btn-lg"> Webpage </button></a>
</h3>

<h3> Description 
<p>
This space illustrates <a href='https://nv-nguyen.github.io/template-pose' target='_blank'><b>Template-Pose</b></a>, a method for novel object pose estimation from CAD.
</p>
</h3>
"""


def get_examples(dir):
    name_example = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if os.path.isdir(os.path.join(dir, f))
    ]
    examples = []  # query, cad
    for name in name_example:
        query_paths = glob.glob(os.path.join(name, "query*.png"))
        for query_path in query_paths:
            obj_id = int(os.path.basename(name).split("_")[-1])
            cad_path = os.path.join(name, f"obj_{obj_id:06d}.ply")
            examples.append([query_path, cad_path])
            break
    return examples


def call_pyrender(cad_model, is_top_sphere):
    from src.poses.pyrender import render
    # get template position on the sphere
    from src.poses.utils import get_obj_poses_from_template_level
    from src.utils.trimesh_utils import get_obj_diameter
    poses = get_obj_poses_from_template_level(
        level=2, pose_distribution="upper" if is_top_sphere else "all"
    )
    # normalize meshes
    cad_model = get_obj_diameter()
    render(
        mesh,
        output_dir,
        obj_poses,
        img_size,
        intrinsic,
        is_tless=False,
    )


def main(model, device, query_image, cad_model, is_top_sphere, num_neighbors):
    """
    The pipeline is:
    1. Rendering posed templates given CAD model
    2. Compute descriptors of these templates
    3. For each query image, compute its features and find nearest neighbors
    """
    print(query_image, cad_model, is_top_sphere, num_neighbors)
    # render images from CAD model
    templates = call_pyrender(cad_model, is_top_sphere)

    return templates


def run_demo():
    inputs = [
        gr.Image(label="cropped query image", type="pil", image_mode="RGB"),
        gr.Model3D(label="CAD model"),
        gr.inputs.Checkbox(label="Templates only on top sphere?", default=False),
        gr.Slider(0, 5, value=3, step=1, label="Number of neighbors to show"),
    ]
    output = gr.Gallery(label="Nearest neighbors")
    output.style(grid=5)

    fn_with_model = partial(main, None, None)
    fn_with_model.__name__ = "fn_with_model"

    examples = get_examples("./media/demo/")
    demo = gr.Interface(
        fn=fn_with_model,
        title=WEBSITE,
        inputs=inputs,
        outputs=output,
        allow_flagging="never",
        examples=examples,
        cache_examples=True,
    )
    demo.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", nargs="?", help="Path to the checkpoint")
    args = parser.parse_args()
    config = OmegaConf.load("configs/model/resnet50.yaml")
    print(config)
    fire.Fire(run_demo)
    # device
    device = torch.