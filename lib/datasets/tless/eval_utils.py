import os
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from lib.datasets.tless import inout
from lib.datasets import image_utils
from bop_toolkit_lib.vsd_metric import vsd
import matplotlib.pyplot as plt


def get_diagonal_from_bbox(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return np.sqrt(width ** 2 + height ** 2)


def get_center_from_bbox(bbox):
    c_x = (bbox[2] + bbox[0]) / 2
    c_y = (bbox[3] + bbox[1]) / 2
    return [c_x, c_y]


intrinsic = np.asarray([1075.65091572, 0.0, 360,
                        0.0, 1073.90347929, 270,
                        0.0, 0.0, 1.0]).reshape(3, 3)


def eval_vsd(idx, list_prediction_path, list_save_path, root_dir, detected_bbox=False):
    """
    From predicted idx_template and predicted inplane rotation, we predict the 6D pose of the objects
    https://github.com/DLR-RM/AugmentedAutoencoder/blob/28708f1663f559a1876590cc9895bdff18680383/auto_pose/ae/codebook.py#L79

    Output is under BOP format
    https://bop.felk.cvut.cz/challenges/bop-challenge-2020/#howtoparticipate
    """
    data = np.load(list_prediction_path[idx])
    id_obj = data["id_obj"][0]
    eval_cad_path = os.path.join(root_dir, "tless/models/models_eval/obj_{:06d}.ply".format(id_obj))
    template_opencv_poses = inout.read_template_poses(opengl_camera=False, dense=False)
    num_frames = len(data["id_scene"])
    vsd_error, percentage_visiblility = np.zeros(num_frames), np.zeros(num_frames)
    for i in tqdm(range(num_frames)):
        id_scene, id_frame, idx_obj_in_scene = data["id_scene"][i], data["id_frame"][i], data["idx_obj_in_scene"][i]
        if not detected_bbox:
            query_mask = inout.open_real_image_tless(root_path=root_dir, split="test",
                                                     id_scene=id_scene, id_frame=id_frame,
                                                     idx_obj_in_scene=idx_obj_in_scene, image_type="mask")
            query_bbox = query_mask.getbbox()
        else:
            query_bbox = inout.read_unseen_detections(root_path=root_dir, id_obj=id_obj,
                                                      id_scene=id_scene, id_frame=id_frame)
        id_obj, pred_idx_template = data["id_obj"][i], data["pred_idx_template"][i]
        pred_inplane, idx_frame = data["pred_inplane"][i], data["idx_frame"][i]
        pred_mask = inout.open_template_tless(root_path=root_dir, id_obj=id_obj, idx_template=pred_idx_template,
                                              image_type="mask", inplane=pred_inplane, dense=False)
        template_bbox = pred_mask.getbbox()

        # initialize the predicted pose with the pose of template
        predicted_pose = np.copy(template_opencv_poses[pred_idx_template])

        # taking in account the predicted inplane
        inplane_matrix = R.from_euler('z', -pred_inplane, degrees=True).as_matrix()
        predicted_pose[:3, :3] = inplane_matrix.dot(predicted_pose[:3, :3])

        # predict now the translation
        intrinsic_query = inout.read_real_intrinsic_tless(root_dir=root_dir, dataset="tless", idx_frame=idx_frame,
                                                          split="test", id_obj=id_obj)
        K_diag_ratio = np.sqrt(intrinsic_query[0, 0] ** 2 + intrinsic_query[1, 1] ** 2) / \
                       np.sqrt(intrinsic[0, 0] ** 2 + intrinsic[1, 1] ** 2)
        assert K_diag_ratio == 1, print("Wrong K ratio!!!")

        gt_bb_diag = get_diagonal_from_bbox(query_bbox)
        template_bb_diag = get_diagonal_from_bbox(template_bbox)
        predicted_pose[2, 3] = predicted_pose[2, 3] * template_bb_diag / gt_bb_diag

        center_obj_x_template = (template_bbox[0] + template_bbox[2]) / 2. - intrinsic[0, 2]
        center_obj_y_template = (template_bbox[1] + template_bbox[3]) / 2. - intrinsic[1, 2]

        center_obj_x_query = (query_bbox[0] + query_bbox[2]) / 2 - intrinsic_query[0, 2]
        center_obj_y_query = (query_bbox[1] + query_bbox[3]) / 2 - intrinsic_query[1, 2]

        center_mm_tx = center_obj_x_query * predicted_pose[2, 3] / intrinsic_query[0, 0] - \
                       center_obj_x_template * template_opencv_poses[pred_idx_template, 2, 3] / intrinsic[0, 0]
        center_mm_ty = center_obj_y_query * predicted_pose[2, 3] / intrinsic_query[1, 1] - \
                       center_obj_y_template * template_opencv_poses[pred_idx_template, 2, 3] / intrinsic[1, 1]

        predicted_pose[:2, 3] = [center_mm_tx, center_mm_ty]

        gt_pose = inout.read_opencv_pose_tless(root_dir=root_dir, dataset="tless", split="test",
                                               id_obj=id_obj, idx_frame=idx_frame)
        depth_test = inout.open_real_image_tless(root_path=root_dir,
                                                 split="test", id_scene=id_scene,
                                                 id_frame=id_frame, idx_obj_in_scene=idx_obj_in_scene,
                                                 image_type="depth")
        results = vsd(R_est=predicted_pose[:3, :3], t_est=predicted_pose[:3, 3],
                      R_gt=gt_pose[:3, :3], t_gt=gt_pose[:3, 3],
                      depth_test=depth_test, K=intrinsic_query, delta=15, taus=[20],
                      normalized_by_diameter=False, diameter=None,
                      eval_cad_path=eval_cad_path, cost_type='step')
        vsd_error[i] = results
        percentage_visiblility[i] = data["visib_fract"][i]
    np.savez(list_save_path[idx], vsd_error=vsd_error, percentage_visiblility=percentage_visiblility)
    valid_vsd_error = vsd_error[percentage_visiblility >= 0.1]
    print("Mean of vsd {}, final score {}".format(np.mean(valid_vsd_error), np.mean((valid_vsd_error <= 0.3) * 1)))
    return (valid_vsd_error <= 0.3) * 1