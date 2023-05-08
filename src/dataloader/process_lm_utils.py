import numpy as np
import os.path as osp
import os
from PIL import Image
from tqdm import tqdm
import logging
from src.utils.inout import load_json, save_json


def process_folder(input_dir, save_dir, poses):
    rgb_dir = osp.join(input_dir, "rgb")
    mask_dir = osp.join(input_dir, "mask")
    pose_dir = osp.join(input_dir, "pose")

    save_rgb_dir = osp.join(save_dir, "rgb")
    save_mask_dir = osp.join(save_dir, "mask")
    os.makedirs(save_rgb_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    # rename and convert image to png format
    rgb_paths = sorted(os.listdir(rgb_dir))
    mask_paths = sorted(os.listdir(mask_dir))
    pose_paths = sorted(os.listdir(pose_dir))
    assert len(rgb_paths) == len(poses)
    logging.info(
        f"Number of rgb {len(rgb_paths)}, mask {len(mask_paths)}, pose {len(pose_paths)}"
    )
    # process image

    scene_gt_info = {}
    for i, rgb_path in tqdm(enumerate(rgb_paths)):
        rgb_path = osp.join(rgb_dir, rgb_path)
        new_rgb_path = osp.join(save_rgb_dir, "{:06d}.png".format(i))

        # convert jpeg to png
        img = Image.open(rgb_path)
        img.save(new_rgb_path)

        frame_id = int(osp.splitext(osp.basename(rgb_path))[0].replace(".jpg", ""))
        mask_path = osp.join(mask_dir, f"{frame_id:04d}.png")
        new_mask_path = osp.join(save_mask_dir, "{:06d}.png".format(i))
        if osp.exists(mask_path):
            mask = Image.open(mask_path)
            mask.save(new_mask_path)
            bbox = Image.open(new_mask_path).getbbox()
        else:
            # create empty mask
            mask = Image.new("L", img.size, 0)
            mask.save(new_mask_path)
            bbox = [0, 0, 0, 0]
        scene_gt_info[f"{i}"] = [{"bbox_visib": bbox}]
    save_json(osp.join(save_dir, "scene_gt_info.json"), scene_gt_info)

    # create scene_gt.json, scene_gt_info.json and scene_camera.json
    K = np.array(
        [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
    ).reshape(-1)
    scene_camera = {f"{i}": {"cam_K": K.tolist()} for i in range(len(rgb_paths))}
    save_json(osp.join(save_dir, "scene_camera.json"), scene_camera)
    save_json(osp.join(save_dir, "scene_gt.json"), poses)
    logging.info("Done")


def process_poses(obj_pose_path, id_obj):
    poses = load_json(obj_pose_path)
    bop_poses = {}
    # add id_obj to each pose
    for id_frame in poses:
        pose = poses[id_frame]
        new_pose = {}
        new_pose["cam_R_m2c"] = pose["cam_R_w2c"]
        new_pose["cam_t_m2c"] = (np.array(pose["cam_t_w2c"]) * 1000).tolist()
        new_pose["obj_id"] = int(id_obj + 1)
        bop_poses[f"{id_frame}"] = []
        bop_poses[f"{id_frame}"].append(new_pose)
    return bop_poses


def process_occlusionLM(
    input_dir, pose_dir, save_dir, query_names, occlusion_query_names
):
    rgb_dir = osp.join(input_dir, "RGB-D", "rgb_noseg")
    depth_dir = osp.join(input_dir, "RGB-D", "depth_noseg")
    save_rgb_dir = osp.join(save_dir, "rgb")
    save_depth_dir = osp.join(save_dir, "depth")
    os.makedirs(save_rgb_dir, exist_ok=True)
    os.makedirs(save_depth_dir, exist_ok=True)
    rgb_paths = sorted(os.listdir(rgb_dir))
    depth_paths = sorted(os.listdir(depth_dir))
    logging.info(f"Number of rgb {len(rgb_paths)}, depth {len(depth_paths)}")

    # process image to BOP format
    for i, path in tqdm(enumerate(rgb_paths)):
        rgb_path = osp.join(rgb_dir, path)
        depth_path = osp.join(depth_dir, depth_paths[i])
        new_rgb_path = osp.join(save_rgb_dir, "{:06d}.png".format(i))
        new_depth_path = osp.join(save_depth_dir, "{:06d}.png".format(i))
        os.system("cp {} {}".format(rgb_path, new_rgb_path))
        os.system("cp {} {}".format(depth_path, new_depth_path))
    img_size = Image.open(new_rgb_path).size
    bbox = [0, 0, img_size[0], img_size[1]]
    idx_occlusionLM = np.asarray([0, 3, 4, 5, 6, 7, 8, 9])
    occlusion_real_ids = np.asarray([0, 4, 5, 7, 8, 9, 10, 11])
    avail_poses = {}
    for idx, id_obj in tqdm(enumerate(idx_occlusionLM)):
        obj_pose_path = osp.join(pose_dir, f"{query_names[id_obj]}.json")
        assert query_names[id_obj] in occlusion_query_names, f"{query_names[id_obj]}"
        poses = process_poses(obj_pose_path, occlusion_real_ids[idx])
        avail_poses[id_obj] = poses
    # convert to bop format
    scene_gt = {}
    scene_gt_info = {}
    for id_frame in tqdm(range(len(rgb_paths))):
        scene_gt_info[f"{id_frame}"] = []
        scene_gt[f"{id_frame}"] = []
        for id_obj in idx_occlusionLM:
            if id_frame in avail_poses[id_obj] or f"{id_frame}" in avail_poses[id_obj]:
                scene_gt[f"{id_frame}"].append(
                    avail_poses[id_obj][id_frame][0]
                    if id_frame in avail_poses[id_obj]
                    else avail_poses[id_obj][f"{id_frame}"][0]
                )
                scene_gt_info[f"{id_frame}"].append({"bbox_visib": bbox})
            else:
                scene_gt_info[f"{id_frame}"].append({"bbox_visib": [0, 0, 0, 0]})
        save_json(osp.join(save_dir, "scene_gt.json"), scene_gt)
        save_json(osp.join(save_dir, "scene_gt_info.json"), scene_gt_info)
        K = np.array(
            [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
        ).reshape(-1)
        scene_camera = {f"{i}": {"cam_K": K.tolist()} for i in range(len(rgb_paths))}
        save_json(osp.join(save_dir, "scene_camera.json"), scene_camera)
