# this file defines the offset between the dataset LINEMOD and Occlusion (provided links) w.r.t BOP challenge
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm
import cv2
from PIL import Image
from lib.datasets.linemod import inout
from lib.poses import utils

# https://github.com/paroj/ObjRecPoseEst/blob/b9ce7221f73f105dbcbae1296e19f8310a6670c4/src/data/camera.py#L8
intrinsic = np.array([[572.4114, 0., 325.2611],
                      [0., 573.57043, 242.04899],
                      [0., 0., 1.]])


def get_transformation_LINEMOD(bop_path, linemod_path, id_obj):
    rotation_z = [True, False, False, False, True,
                  False, True, True, False, True,
                  True, False, False, False, False]
    object_names = ['ape', 'benchvise', 'bowl', 'cam', 'can', 'cat',
                    'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                    'iron', 'lamp', 'phone']
    matrix_4x4 = np.zeros((4, 4))
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= 0.001
    bop_mesh = trimesh.load(os.path.join(bop_path, "obj_{:06d}.ply".format(id_obj + 1)))
    bop_mesh = bop_mesh.apply_transform(scale_matrix)
    bop_mesh_bound = bop_mesh.bounds
    location_bop = np.mean(bop_mesh_bound, axis=0)

    if rotation_z[id_obj]:
        rotz180 = R.from_euler('z', 180, degrees=True).as_matrix()
    else:
        rotz180 = R.from_euler('z', 0, degrees=True).as_matrix()
    normalized_mesh = trimesh.load(os.path.join(linemod_path, "{}.ply".format(object_names[id_obj])))
    rot_mat = np.zeros((4, 4))
    rot_mat[3, 3] = 1
    rot_mat[:3, :3] = rotz180
    normalized_mesh = normalized_mesh.apply_transform(rot_mat)
    normalized_mesh_bound = normalized_mesh.bounds
    location_normalized = np.mean(normalized_mesh_bound, axis=0)
    delta = location_bop - location_normalized

    matrix_4x4[:3, :3] = rotz180
    matrix_4x4[:3, 3] = delta
    matrix_4x4[3, 3] = 1
    return matrix_4x4


def get_transformation_occlusionLINEMOD(idx_obj):
    rotz_occlusion_linemod = [-90, -90, 90, -90, 90, -90, -90, 90]
    # there is not offset of translation in occlusion LINEMOD
    rotx90 = R.from_euler('x', 90, degrees=True).as_matrix()
    rotz90 = R.from_euler('z', rotz_occlusion_linemod[idx_obj], degrees=True).as_matrix()
    rot_matx = np.zeros((4, 4))
    rot_matx[3, 3] = 1
    rot_matx[:3, :3] = rotx90

    rot_matz = np.zeros((4, 4))
    rot_matz[3, 3] = 1
    rot_matz[:3, :3] = rotz90
    return rot_matz.dot(rot_matx)


def read_template_poses(split, opengl_camera):
    if split == "test":
        template_poses = np.load("./lib/poses/predefined_poses/half_sphere_level2.npy")
    elif split == "train":
        template_poses = np.load("./lib/poses/predefined_poses/half_sphere_level2_and_level3.npy")
    if opengl_camera:
        for id_frame in range(len(template_poses)):
            template_poses[id_frame] = utils.opencv2opengl(template_poses[id_frame])
    return template_poses


def find_best_template(query_pose, template_poses, id_symmetry, base_on_angle=True):
    if base_on_angle:
        delta = np.zeros((len(template_poses)))
        for i in range(len(template_poses)):
            delta[i] = angle_between_two_viewpoints(query_pose, template_poses[i], id_symmetry)
        return int(np.argmin(delta)), np.min(delta)
    else:
        delta = template_poses[:, 2, :3] - query_pose[np.newaxis, :, :][:, 2, :3]
        delta = np.linalg.norm(delta, axis=1)
        best_index = np.argmin(delta)
        return int(best_index), angle_between_two_viewpoints(query_pose, template_poses[best_index],
                                                             id_symmetry).tolist()


def viewpoint_difference(location1, location2):
    division_term = np.linalg.norm(location1) * np.linalg.norm(location2)
    cosine_similarity = np.clip(location1.dot(location2) / division_term, a_min=-1, a_max=1)
    angle = np.arccos(cosine_similarity)
    return angle


def angle_between_two_viewpoints(pose1, pose2, id_symmetry):
    """
    The input pose are in opengl coordinate system
    """
    location1 = -pose1[2, :3]
    location2 = -pose2[2, :3]
    angle = viewpoint_difference(location1, location2)
    if id_symmetry == 1:
        # symmetry objects around z axis: eggbox, glue
        location1_sym = location1
        location1_sym[:2] *= -1  # rotation 180 in Z axis
        angle_sym = viewpoint_difference(location1_sym, location2)
        angle = np.min((np.float32(angle), np.float32(angle_sym)))
    elif id_symmetry == 2:
        # considering only the rotation in Z axis
        location1_sym = location1
        location1_sym[:2] = location2[:2]
        angle_sym = viewpoint_difference(location1_sym, location2)
        angle = np.min((np.float32(angle), np.float32(angle_sym)))
    return np.rad2deg(angle)


def crop_frame(opencv_pose, img, crop_size, virtual_bbox_size, ignore_inplane=True):
    origin_obj = np.array([0, 0, 0, 1.])
    origin_in_cam = np.dot(opencv_pose, origin_obj)[0:3]  # center pt in camera space
    if ignore_inplane:
        upper = np.array([0., -origin_in_cam[2], origin_in_cam[1]])
        right = np.array([origin_in_cam[1] * origin_in_cam[1] + origin_in_cam[2] * origin_in_cam[2],
                          -origin_in_cam[0] * origin_in_cam[1], -origin_in_cam[0] * origin_in_cam[2]])
    else:
        upV = np.array([0, 0, 6]) - origin_in_cam
        upV = (np.dot(opencv_pose, [upV[0], upV[1], upV[2], 1]))[0:3]
        right = np.cross(origin_in_cam, upV)
        upper = np.cross(right, origin_in_cam)
        # normalize, resize

    upper = upper * (virtual_bbox_size / 2) / np.linalg.norm(upper)
    right = right * (virtual_bbox_size / 2) / np.linalg.norm(right)

    # world coord of corner points
    w1 = origin_in_cam + upper - right
    w2 = origin_in_cam - upper - right
    w3 = origin_in_cam + upper + right
    w4 = origin_in_cam - upper + right

    # coord of corner points on image plane
    virtual_bbox = np.concatenate((w1.reshape((1, 3)), w2.reshape((1, 3)),
                                   w3.reshape((1, 3)), w4.reshape((1, 3))), axis=0)
    virtual_bbox2d = utils.perspective(intrinsic, np.eye(4), virtual_bbox)
    virtual_bbox2d = virtual_bbox2d.astype(np.int32)
    target_virtual_bbox2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32) * crop_size
    M = cv2.getPerspectiveTransform(virtual_bbox2d.astype(np.float32), target_virtual_bbox2d)
    crop_img = cv2.warpPerspective(np.asarray(img), M, (crop_size, crop_size))
    return Image.fromarray(np.uint8(crop_img))


def crop_dataset(idx_obj, dataset, root_path, save_dir, crop_size, split=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if dataset == "LINEMOD":
        dataset_dir = os.path.join(root_path, "linemod", dataset, "objects")
        object_names = inout.LINEMOD_names
    elif dataset == "occlusionLINEMOD":
        dataset_dir = os.path.join(root_path, "linemod", dataset, "RGB-D/rgb_noseg")
        object_names = inout.occlusion_LINEMOD_names
    elif dataset == "templatesLINEMOD":
        dataset_dir = os.path.join(root_path, "templates/linemod", split)
        object_names = inout.LINEMOD_names

    save_obj_crop_dir = os.path.join(save_dir, object_names[idx_obj])
    if not os.path.exists(save_obj_crop_dir):
        os.makedirs(save_obj_crop_dir)

    if dataset == "LINEMOD":
        img_obj_dir = os.path.join(dataset_dir, object_names[idx_obj], "rgb")
        img_name = "{:04d}.jpg"
        num_images = len(os.listdir(img_obj_dir))
    elif dataset == "occlusionLINEMOD":
        img_obj_dir = dataset_dir
        img_name = "color_{:05d}.png"
        num_images = len(os.listdir(img_obj_dir))
    elif dataset == "templatesLINEMOD":
        img_obj_dir = os.path.join(dataset_dir, object_names[idx_obj])
        img_name = "{:06d}.png"
        num_images = 301 if split == "test" else 1542

    for idx_frame in tqdm(range(num_images)):
        img = Image.open(os.path.join(img_obj_dir, img_name.format(idx_frame)))
        opencv_pose = inout.read_opencv_pose_linemod(root_dir=root_path, dataset=dataset,
                                                     idx_obj=idx_obj, id_frame=idx_frame, split=split)
        if opencv_pose is not None:
            crop_img = crop_frame(opencv_pose, img=img, crop_size=crop_size,
                                  virtual_bbox_size=0.2, ignore_inplane=False)
            crop_img.save(os.path.join(save_obj_crop_dir, "{:06d}.png".format(idx_frame)))
            if dataset == "templatesLINEMOD":
                mask = Image.open(os.path.join(img_obj_dir, "mask_{:06d}.png".format(idx_frame)))
                crop_mask = crop_frame(opencv_pose, img=mask, crop_size=crop_size,
                                       virtual_bbox_size=0.2, ignore_inplane=False)
                crop_mask.save(os.path.join(save_obj_crop_dir, "{:06d}_mask.png".format(idx_frame)))
