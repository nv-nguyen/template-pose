import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from lib.poses import utils
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml


def load_info(path):
    with open(path, 'r') as f:
        info = yaml.load(f, Loader=yaml.CLoader)
        for eid in info.keys():
            if 'cam_K' in info[eid].keys():
                info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape(
                    (3, 3))
            if 'cam_R_w2c' in info[eid].keys():
                info[eid]['cam_R_w2c'] = np.array(
                    info[eid]['cam_R_w2c']).reshape((3, 3))
            if 'cam_t_w2c' in info[eid].keys():
                info[eid]['cam_t_w2c'] = np.array(
                    info[eid]['cam_t_w2c']).reshape((3, 1))
    return info


def load_bop(path):
    with open(path, 'r') as f:
        info = yaml.load(f, Loader=yaml.CLoader)
    return info


def load_gt(path):
    with open(path, 'r') as f:
        gts = yaml.load(f, Loader=yaml.CLoader)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
    return gts


def read_template_poses(opengl_camera, dense=False):
    if dense:
        if os.path.exists("./lib/poses/predefined_poses/sphere_level3.npy"):
            template_poses = np.load("./lib/poses/predefined_poses/sphere_level3.npy")
        else:
            template_poses = np.load("../lib/poses/predefined_poses/sphere_level3.npy")
    else:
        if os.path.exists("./lib/poses/predefined_poses/sphere_level2.npy"):
            template_poses = np.load("./lib/poses/predefined_poses/sphere_level2.npy")
        else:
            template_poses = np.load("../lib/poses/predefined_poses/sphere_level2.npy")
    if opengl_camera:
        for id_frame in range(len(template_poses)):
            template_poses[id_frame] = utils.opencv2opengl(template_poses[id_frame])
    return template_poses


def read_opencv_pose_tless(root_dir, dataset, split, id_obj, idx_frame, all=False):
    if dataset == "tless":
        pose_path = os.path.join(root_dir, "tless/opencv_pose/", "{:02d}_{}.json".format(id_obj, split))
    with open(pose_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.CLoader)

    if idx_frame in data:
        pose = np.zeros((4, 4))
        pose[3, 3] = 1
        pose[:3, :3] = np.asarray(data[idx_frame]['cam_R_m2c']).reshape(3, 3)
        pose[:3, 3] = np.asarray(data[idx_frame]['cam_t_m2c']).reshape(3)
        if all:
            return [data[idx_frame]['id_scene'], data[idx_frame]['id_frame'], data[idx_frame]['rgb_path'],
                    data[idx_frame]['depth_path'], data[idx_frame]['visib_fract'],
                    data[idx_frame]['idx_obj_in_scene'], data[idx_frame]['cam_R_m2c'],
                    data[idx_frame]['cam_t_m2c'], data[idx_frame]['cam_K'], data[idx_frame]['depth_scale']], pose
        else:
            return pose
    else:
        return None


def read_real_intrinsic_tless(root_dir, dataset, split, id_obj, idx_frame):
    if dataset == "tless":
        pose_path = os.path.join(root_dir, "tless/opencv_pose/", "{:02d}_{}.json".format(id_obj, split))
    with open(pose_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.CLoader)
    return np.asarray(data[idx_frame]['cam_K']).reshape(3, 3)


def open_real_image_tless(root_path, split, id_scene, id_frame, idx_obj_in_scene, image_type):
    assert image_type in ["rgb", "depth", "mask"], print("Image type is not correct!")
    scene_dir = os.path.join(root_path, "tless", split, "{:06d}".format(id_scene))
    if split == "train":
        assert idx_obj_in_scene == 0, print("Training scenes only contain one object")

    if image_type == "rgb":
        return Image.open(os.path.join(scene_dir, image_type, "{:06d}.png".format(id_frame)))
    elif image_type == "depth":
        return cv2.imread((os.path.join(scene_dir, image_type, "{:06d}.png".format(id_frame))), -1) / 10.  # to mm
    elif image_type == "mask":
        return Image.open(os.path.join(scene_dir, "mask", "{:06d}_{:06d}.png".format(id_frame, idx_obj_in_scene)))


def open_template_tless(root_path, id_obj, idx_template, image_type, inplane, dense=False):
    assert image_type in ["rgb", "mask"], print("Image type is not correct!")
    if dense:
        scene_dir = os.path.join(root_path, "templates_dense/tless", "{:02d}".format(id_obj))
    else:
        scene_dir = os.path.join(root_path, "templates/tless", "{:02d}".format(id_obj))

    if image_type == "rgb":
        img = Image.open(os.path.join(scene_dir, "{:06d}.png".format(idx_template)))
    elif image_type == "mask":
        img = Image.open(os.path.join(scene_dir, "mask_{:06d}.png".format(idx_template)))

    img = img.rotate(inplane)
    return img


def save_results(results, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez(save_path, obj_pose=np.array(results["obj_pose"]), id_obj=np.array(results["id_obj"]),
             id_scene=np.array(results["id_scene"]), id_frame=np.array(results["id_frame"]),
             idx_frame=np.array(results["idx_frame"]),
             idx_obj_in_scene=np.array(results["idx_obj_in_scene"]),
             gt_idx_template=np.array(results["gt_idx_template"]),
             gt_inplane=np.array(results["gt_inplane"]),
             pred_template_pose=np.array(results["pred_template_pose"]),
             pred_idx_template=np.array(results["pred_idx_template"]),
             pred_inplane=np.array(results["pred_inplane"]),
             visib_fract=np.array(results["visib_fract"]))


def read_unseen_detections(root_path, id_obj, id_scene, id_frame):
    detections_path = os.path.join(root_path, "tless/unseen_detections/", "{:02d}.json".format(id_obj))
    with open(detections_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.CLoader)
    id_scene_id_frame = "id_scene_{:02d}_id_frame_{:04d}".format(id_scene, id_frame)
    if id_scene_id_frame in data:
        return data[id_scene_id_frame]
