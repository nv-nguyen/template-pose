import fire
import gradio as gr
from functools import partial
from PIL import Image
import numpy as np
from src.utils.gradio_utils import CameraVisualizer, calc_cam_cone_pts_3d
import os


WEBSITE = """
<h1 style='text-align: center'>Templates for 3D Object Pose Estimation Revisited: : <br>
                Generalization to New Objects and Robustness to Occlusions</br> </h1>

<h3 style='text-align: center'>
<a href="https://nv-nguyen.github.io/" target="_blank"><nobr>Van Nguyen Nguyen</nobr></a> &emsp;
<a href="https://yinlinhu.github.io/" target="_blank"><nobr>Yinlin Hu</nobr></a> &emsp;
<a href="https://youngxiao13.github.io/" target="_blank"><nobr>Yang Xiao</nobr></a> &emsp;
<a href="https://people.epfl.ch/mathieu.salzmann" target="_blank"><nobr>Mathieu Salzmann</nobr></a> &emsp;
<a href="https://vincentlepetit.github.io/" target="_blank"><nobr>Vincent Lepetit</nobr></a>
</h3>

<h3 style='text-align: center'>
<nobr>arXiv 2023</nobr>
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
    examples = []  # reference, query, num_neighbors
    for name in name_example:
        query_path = os.path.join(name, "query.png")
        reference_path = os.path.join(name, "reference.png")
        examples.append([reference_path, query_path])
    return examples


def main(model, device, cam_vis, reference_image, query_image, num_neighbors):
    if num_neighbors is None:  # dirty fix for examples
        num_neighbors = 3
    # update the number of neighbors
    cam_vis.neighbors_change(num_neighbors)

    # calculate predicted pose distribution and neighbors
    proba = Image.open("media/demo/proba.png")
    elevations = np.random.uniform(-60, 0, size=num_neighbors)
    azimuths = np.random.uniform(0, 360, size=num_neighbors)

    # update the figure
    cam_vis.polar_change(elevations)
    cam_vis.azimuth_change(azimuths)
    tmp = np.array(reference_image.convert("RGB"))
    cam_vis.encode_image(np.uint8(tmp))
    new_fig = cam_vis.update_figure()
    return [new_fig, proba]


def run_demo():
    inputs = [
        gr.Image(label="reference", type="pil", image_mode="RGB"),
        gr.Image(label="query", type="pil", image_mode="RGB"),
        gr.Slider(0, 5, value=3, step=1, label="Number of neighbors to show"),
    ]
    vis_output = gr.Plot(label="Predictions")
    neighbors_output = gr.Image(label="Pose ditribution", type="pil", image_mode="RGBA")

    cam_vis = CameraVisualizer(vis_output)
    fn_with_model = partial(main, None, None, cam_vis)
    fn_with_model.__name__ = "fn_with_model"

    examples = get_examples(dir="media/demo")
    demo = gr.Interface(
        fn=fn_with_model,
        title=WEBSITE,
        inputs=inputs,
        outputs=[vis_output, neighbors_output],
        allow_flagging="never",
        examples=examples,
        cache_examples=True,
    )
    demo.launch(share=True)


if __name__ == "__main__":
    fire.Fire(run_demo)
