import numpy as np
import os
import json

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

LINEMOD_names = np.asarray(['ape', 'benchvise', 'cam', 'can', 'cat',
                            'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                            'iron', 'lamp', 'phone'])
LINEMOD_real_ids = np.asarray([0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14])
list_id_symmetry = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
list_all_id_symmetry = [0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
LINEMOD_name_to_real_id = {'ape': 0,
                           'benchvise': 1,
                           'cam': 3,
                           'can': 4,
                           'cat': 5,
                           'driller': 7,
                           'duck': 8,
                           'eggbox': 9,
                           'glue': 10,
                           'holepuncher': 11,
                           'iron': 12,
                           'lamp': 13,
                           'phone': 14}
LINEMOD_real_id_to_name = {v: k for k, v in LINEMOD_name_to_real_id.items()}
LINEMOD_plane_symmetry_names = ['eggbox', 'glue']

occlusion_LINEMOD_names = np.asarray(['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher'])
occlusion_real_ids = np.asarray([0, 4, 5, 7, 8, 9, 10, 11])
occlusion_name_to_id = {
    'ape': 0,
    'can': 4,
    'cat': 5,
    'driller': 7,
    'duck': 8,
    'eggbox': 9,
    'glue': 10,
    'holepuncher': 11}
occlusion_real_id_to_name = {v: k for k, v in occlusion_name_to_id.items()}

intrinsic_real_img = np.array([[572.4114, 0., 325.2611],
                               [0., 573.57043, 242.04899],
                               [0., 0., 1.]])


def get_list_id_obj_from_split_name(split_name):
    assert split_name in ["split1", "split2", "split3"], print("Split_name is not correct!!!")
    list_id_obj = LINEMOD_real_ids
    if split_name == "split1":
        seen_id_obj, seen_names = list_id_obj[4:], LINEMOD_names[4:]
        seen_occ_id_obj, seen_occ_names = occlusion_real_ids[2:], occlusion_LINEMOD_names[2:]
        unseen_id_obj, unseen_names = list_id_obj[:4], LINEMOD_names[:4]
        unseen_occ_id_obj, unseen_occ_names = occlusion_real_ids[:2], occlusion_LINEMOD_names[:2]
    elif split_name == "split2":
        seen_id_obj, seen_names = np.concatenate((list_id_obj[:4], list_id_obj[8:])), \
                                  np.concatenate((LINEMOD_names[:4], LINEMOD_names[8:]))
        seen_occ_id_obj, seen_occ_names = np.concatenate((occlusion_real_ids[:2], occlusion_real_ids[6:])), \
                                          np.concatenate((occlusion_LINEMOD_names[:2], occlusion_LINEMOD_names[6:]))
        unseen_id_obj, unseen_names = list_id_obj[4:8], LINEMOD_names[4:8]
        unseen_occ_id_obj, unseen_occ_names = occlusion_real_ids[2:6], occlusion_LINEMOD_names[2:6]
    elif split_name == "split3":
        seen_id_obj, seen_names = list_id_obj[:8], LINEMOD_names[:8]
        seen_occ_id_obj, seen_occ_names = occlusion_real_ids[:6], occlusion_LINEMOD_names[:6]
        unseen_id_obj, unseen_names = list_id_obj[8:], LINEMOD_names[8:]
        unseen_occ_id_obj, unseen_occ_names = occlusion_real_ids[6:], occlusion_LINEMOD_names[6:]
    print("Seen: {}".format(seen_names))
    print("Occluded Seen: {}".format(seen_occ_names))
    print("Unseen: {}".format(unseen_names))
    print("Occluded Unseen: {}".format(unseen_occ_names))
    return seen_id_obj, seen_names, seen_occ_id_obj, seen_occ_names, \
           unseen_id_obj, unseen_names, unseen_occ_id_obj, unseen_occ_names


def read_original_pose_linemod(linemod_dir, id_obj, id_frame):
    pose_path = os.path.join(linemod_dir, "objects", LINEMOD_names[id_obj], "pose/{:04d}.txt".format(id_frame))
    return np.loadtxt(pose_path)


def read_original_pose_occlusion_linemod(occlusion_dir, id_obj, id_frame):
    # https://github.com/chensong1995/HybridPose/blob/master/lib/datasets/occlusion_linemod.py
    occlusion_name_to_id = {
        'Ape': 0,
        'Can': 3,
        'Cat': 4,
        'Driller': 5,
        'Duck': 6,
        'Eggbox': 7,
        'Glue': 8,
        'Holepuncher': 9}
    occlusion_real_id_to_Name = {v: k for k, v in occlusion_name_to_id.items()}
    pose_path = os.path.join(occlusion_dir, "poses", occlusion_real_id_to_Name[id_obj], "info_{:05d}.txt".format(id_frame))
    read_rotation = False
    read_translation = False
    R = []
    T = []
    with open(pose_path) as f:
        for line in f:
            if read_rotation:
                R.append(line.split())
                if len(R) == 3:
                    read_rotation = False
            elif read_translation:
                T = line.split()
                read_translation = False
            if line.startswith('rotation'):
                read_rotation = True
            elif line.startswith('center'):
                read_translation = True
    if len(R) > 0:
        matrix4x4 = np.zeros((4, 4))
        matrix4x4[3, 3] = 1
        matrix4x4[:3, :3] = np.array(R, dtype=np.float32)  # 3*3
        matrix4x4[:3, 3] = np.array(T, dtype=np.float32).reshape(3)  # 3*1
        return matrix4x4
    else:
        return None


def read_opencv_pose_linemod(root_dir, dataset, idx_obj, id_frame, split=None):
    if dataset in ['LINEMOD', 'occlusionLINEMOD']:
        if dataset == 'LINEMOD':
            pose_path = os.path.join(root_dir, "linemod/opencv_pose/", dataset,
                                     "{}.json".format(LINEMOD_names[idx_obj]))
        elif dataset == "occlusionLINEMOD":
            pose_path = os.path.join(root_dir, "linemod/opencv_pose/", dataset,
                                     "{}.json".format(occlusion_LINEMOD_names[idx_obj]))
        with open(pose_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.CLoader)
        if id_frame in data:
            pose = np.zeros((4, 4))
            pose[3, 3] = 1
            pose[:3, :3] = np.asarray(data[id_frame]['cam_R_w2c']).reshape(3, 3)
            pose[:3, 3] = np.asarray(data[id_frame]['cam_t_w2c']).reshape(3)
            return pose
        else:
            return None
    elif dataset == "templatesLINEMOD":
        if split == "test":
            template_poses = np.load("./lib/poses/predefined_poses/half_sphere_level2.npy")
        elif split == "train":
            template_poses = np.load("./lib/poses/predefined_poses/half_sphere_level2_and_level3.npy")
        template_poses[:, :3, 3] /= 1000  # scale from m to mm
        return template_poses[id_frame]


if __name__ == '__main__':
    for split in ["split1", "split2", "split3"]:
        get_list_id_obj_from_split_name(split)
        print("--------------------")
