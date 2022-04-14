import os
import numpy as np
from mathutils import Euler, Matrix, Vector
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
from lib.utils.inout_BOPformat import save_info
from lib.datasets.tless import inout
from lib.poses import utils


def create_gt_obj(index, list_id_obj, split, root_path):
    id_obj = list_id_obj[index]
    save_path = os.path.join(root_path, "opencv_pose", "{:02d}_{}.json".format(id_obj, split))
    num_img_total = 0
    all_poses = {}
    if split == "train":
        list_id_scene = range(1, 31)
    else:
        list_id_scene = range(1, 21)
    for id_scene in list_id_scene:
        scene_gt = inout.load_bop(os.path.join(root_path, split, "{:06d}/scene_gt.json".format(id_scene)))
        scene_gt_info = inout.load_bop(os.path.join(root_path, split, "{:06d}/scene_gt_info.json".format(id_scene)))
        scene_camera = inout.load_bop(os.path.join(root_path, split, "{:06d}/scene_camera.json".format(id_scene)))
        if check_if_scene_contain_obj(scene_gt=scene_gt, id_obj=id_obj):
            num_img_total += len(scene_gt)
            update_gt_obj(all_poses=all_poses, id_scene=id_scene,
                          scene_gt=scene_gt, scene_gt_info=scene_gt_info, scene_camera=scene_camera, id_obj=id_obj)
    print("Id obj {}: {}".format(id_obj, num_img_total))
    save_info(save_path, all_poses, save_all=True)


def check_if_scene_contain_obj(scene_gt, id_obj):
    scene_obj_ids = set()
    for gt in scene_gt["0"]:
        scene_obj_ids.add(gt['obj_id'])
    if id_obj in scene_obj_ids:
        return True
    else:
        return False


def update_gt_obj(all_poses, id_scene, scene_gt_info, scene_camera, scene_gt, id_obj):
    start_index = len(all_poses)
    for id_frame in range(len(scene_gt)):
        for idx_obj_in_scene, gt_info in enumerate(scene_gt_info["{}".format(id_frame)]):
            gt = scene_gt["{}".format(id_frame)][idx_obj_in_scene]
            if gt["obj_id"] == id_obj:
                rgb_path = os.path.join("{:06d}/rgb/{:06d}.png".format(id_scene, id_frame))
                depth_path = os.path.join("{:06d}/depth/{:06d}.png".format(id_scene, id_frame))
                gt_frame = {'id_scene': id_scene,
                            'id_frame': id_frame,
                            'rgb_path': rgb_path,
                            'depth_path': depth_path,
                            'idx_obj_in_scene': idx_obj_in_scene,
                            'bbox_obj': gt_info['bbox_obj'],
                            'bbox_visib': gt_info['bbox_visib'],
                            'visib_fract': gt_info['visib_fract'],
                            'cam_R_m2c': gt['cam_R_m2c'],
                            'cam_t_m2c': gt['cam_t_m2c'],
                            'cam_K': scene_camera["{}".format(id_frame)]['cam_K'],
                            'depth_scale': scene_camera["{}".format(id_frame)]['depth_scale'],
                            'elev': scene_camera["{}".format(id_frame)]['elev'],
                            'mode': scene_camera["{}".format(id_frame)]['mode']}
                all_poses[start_index + id_frame] = gt_frame
    return all_poses


def create_inp_rotation_matrix(bin_size):
    angles = np.arange(0, 360, bin_size) * np.pi / 180
    list_inp = np.zeros((len(angles), 3, 3))
    for i in range(len(angles)):
        list_inp[i] = np.asarray(Euler((0.0, 0.0, -angles[i])).to_matrix())
    return list_inp


def find_nearest_inplane(inplane, bin_size=10):
    assert -180 <= inplane <= 180, print("Range inplane is not correct!")
    angles = np.arange(0, 370, bin_size) - 180
    idx = (np.abs(angles - inplane)).argmin()
    # angle = -180 is equivalent to angle = 180
    if angles[idx] == -180:
        return np.asarray([180])
    else:
        return angles[idx]


def find_best_template(query_opencv, templates_opengl):
    """
    Find best template based on euclidean distance
    :param query_opencv:
    :param templates_opengl:
    :return:
    """
    # remove rotation 2D
    query_cam_loc = utils.opencv2opengl(query_opencv)[2, :3]
    delta = templates_opengl[:, 2, :3] - query_cam_loc[np.newaxis, :]
    delta = np.linalg.norm(delta, axis=1)
    best_index = np.argmin(delta)
    best_template_opencv = utils.opencv2opengl(templates_opengl[best_index])

    # compute in-plane rotation
    rot_query = query_opencv[:3, :3]
    rot_template = best_template_opencv[:3, :3]
    delta = rot_template.dot(rot_query.T)
    from scipy.spatial.transform import Rotation as R
    inp = R.from_matrix(delta).as_euler('zyx', degrees=True)[0]

    # double check to make sure that reconved rotation is correct
    R_inp = R.from_euler('z', -inp, degrees=True).as_matrix()
    recovered_R1 = R_inp.dot(rot_template)
    err = utils.geodesic_numpy(recovered_R1, rot_query)
    if err >= 15:
        print("WARINING, error of recovered pose is >=15, err=", err)
    return inp, int(best_index), err
