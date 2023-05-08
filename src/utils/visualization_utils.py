import os
import torch.nn.functional as F

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms, utils
import torch
from PIL import Image
import numpy as np
import io

# from moviepy.video.io.bindings import mplfig_to_npimage
import cv2


def put_image_to_grid(list_imgs, adding_margin=True):
    num_col = len(list_imgs)
    b, c, h, w = list_imgs[0].shape
    device = list_imgs[0].device
    if adding_margin:
        num_all_col = num_col + 1
    else:
        num_all_col = num_col
    grid = torch.zeros((b * num_all_col, 3, h, w), device=device).to(torch.float16)
    idx_grid = torch.arange(0, grid.shape[0], num_all_col, device=device).to(
        torch.int64
    )
    for i in range(num_col):
        grid[idx_grid + i] = list_imgs[i].to(torch.float16)
    return grid, num_col + 1


def convert_cmap(tensor, vmin=None, vmax=None):
    b, h, w = tensor.shape
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    output = torch.zeros((b, 3, h, w), device=tensor.device)
    for i in range(b):
        cmap = matplotlib.cm.get_cmap("magma")
        tmp = cmap(ndarr[i])[..., :3]
        data = transforms.ToTensor()(np.array(tmp)).to(tensor.device)
        output[i] = data
    return output


def resize_tensor(tensor, size):
    return F.interpolate(tensor, size, mode="bilinear", align_corners=True)


def render_pts_to_image(cvImg, meshPts, K, openCV_obj_pose, color):
    R, T = openCV_obj_pose[:3, :3], openCV_obj_pose[:3, 3]
    pts = np.matmul(K, np.matmul(R, meshPts.T) + T.reshape(-1, 1))
    xs = pts[0] / (pts[2] + 1e-8)
    ys = pts[1] / (pts[2] + 1e-8)
    for pIdx in range(len(xs)):
        cvImg = cv2.circle(cvImg, (int(xs[pIdx]), int(ys[pIdx])), 3, (int(color[0]), int(color[1]), int(color[2])), -1)
    return cvImg


if __name__ == "__main__":
    import numpy as np

    similarity = np.random.rand(4, 7)
