import pyrender
import trimesh
import numpy as np
import argparse
from PIL import Image
import os
# headless rendering in server
os.environ['DISPLAY'] = ':1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'


class Precomputer(object):
    # credit :  https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py#L110
    """
    Caches pre_Xs, pre_Ys for a 30% speedup of depth_im_to_dist_im()
    """
    xs, ys = None, None
    pre_Xs, pre_Ys = None, None
    depth_im_shape = None
    K = None

    @staticmethod
    def precompute_lazy(depth_im, K):
        """ Lazy precomputation for depth_im_to_dist_im() if depth_im.shape or K changes
        :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
          is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y],
          or 0 if there is no such 3D point (this is a typical output of the
          Kinect-like sensors).
        :param K: 3x3 ndarray with an intrinsic camera matrix.
        :return: hxw ndarray (Xs/depth_im, Ys/depth_im)
        """
        if depth_im.shape != Precomputer.depth_im_shape:
            Precomputer.depth_im_shape = depth_im.shape
            Precomputer.xs, Precomputer.ys = np.meshgrid(
                np.arange(depth_im.shape[1]), np.arange(depth_im.shape[0]))

        if depth_im.shape != Precomputer.depth_im_shape \
                or not np.all(K == Precomputer.K):
            Precomputer.K = K
            Precomputer.pre_Xs = (Precomputer.xs - K[0, 2]) / np.float64(K[0, 0])
            Precomputer.pre_Ys = (Precomputer.ys - K[1, 2]) / np.float64(K[1, 1])

        return Precomputer.pre_Xs, Precomputer.pre_Ys


def depth_im_to_dist_im_fast(depth_im, K):
    # credit: https://github.com/thodan/bop_toolkit/blob/529f11135315cffd536d7b5ea44bcf326daa9a6a/
    # bop_toolkit_lib/misc.py#L143
    """Converts a depth image to a distance image.
    :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
      is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y],
      or 0 if there is no such 3D point (this is a typical output of the
      Kinect-like sensors).
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :return: hxw ndarray with the distance image, where dist_im[y, x] is the
      distance from the camera center to the 3D point [X, Y, Z] that projects to
      pixel [x, y], or 0 if there is no such 3D point.
    """
    # Only recomputed if depth_im.shape or K changes.
    pre_Xs, pre_Ys = Precomputer.precompute_lazy(depth_im, K)

    dist_im = np.sqrt(
        np.multiply(pre_Xs, depth_im) ** 2 +
        np.multiply(pre_Ys, depth_im) ** 2 +
        depth_im.astype(np.float64) ** 2)

    return dist_im


def _estimate_visib_mask(d_test, d_model, delta, visib_mode='bop19'):
    # credit: https://github.com/thodan/bop_toolkit/blob/529f11135315cffd536d7b5ea44bcf326daa9a6a/
    # bop_toolkit_lib/visibility.py#L9
    """Estimates a mask of the visible object surface.
    :param d_test: Distance image of a scene in which the visibility is estimated.
    :param d_model: Rendered distance image of the object model.
    :param delta: Tolerance used in the visibility test.
    :param visib_mode: Visibility mode:
    1) 'bop18' - Object is considered NOT VISIBLE at pixels with missing depth.
    2) 'bop19' - Object is considered VISIBLE at pixels with missing depth. This
         allows to use the VSD pose error function also on shiny objects, which
         are typically not captured well by the depth sensors. A possible problem
         with this mode is that some invisible parts can be considered visible.
         However, the shadows of missing depth measurements, where this problem is
         expected to appear and which are often present at depth discontinuities,
         are typically relatively narrow and therefore this problem is less
         significant.
    :return: Visibility mask.
    """
    assert (d_test.shape == d_model.shape)

    if visib_mode == 'bop18':
        mask_valid = np.logical_and(d_test > 0, d_model > 0)
        d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
        visib_mask = np.logical_and(d_diff <= delta, mask_valid)

    elif visib_mode == 'bop19':
        d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
        visib_mask = np.logical_and(
            np.logical_or(d_diff <= delta, d_test == 0), d_model > 0)

    else:
        raise ValueError('Unknown visibility mode.')

    return visib_mask


def estimate_visib_mask_gt(d_test, d_gt, delta, visib_mode='bop19'):
    # credit: https://github.com/thodan/bop_toolkit/blob/529f11135315cffd536d7b5ea44bcf326daa9a6a/
    # bop_toolkit_lib/visibility.py#L45
    """Estimates a mask of the visible object surface in the ground-truth pose.
    :param d_test: Distance image of a scene in which the visibility is estimated.
    :param d_gt: Rendered distance image of the object model in the GT pose.
    :param delta: Tolerance used in the visibility test.
    :param visib_mode: See _estimate_visib_mask.
    :return: Visibility mask.
    """
    visib_gt = _estimate_visib_mask(d_test, d_gt, delta, visib_mode)
    return visib_gt


def estimate_visib_mask_est(d_test, d_est, visib_gt, delta, visib_mode='bop19'):
    # credit: https://github.com/thodan/bop_toolkit/blob/529f11135315cffd536d7b5ea44bcf326daa9a6a/
    # bop_toolkit_lib/visibility.py#L58
    """Estimates a mask of the visible object surface in the estimated pose.
    For an explanation of why the visibility mask is calculated differently for
    the estimated and the ground-truth pose, see equation (14) and related text in
    Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16.
    :param d_test: Distance image of a scene in which the visibility is estimated.
    :param d_est: Rendered distance image of the object model in the est. pose.
    :param visib_gt: Visibility mask of the object model in the GT pose (from
      function estimate_visib_mask_gt).
    :param delta: Tolerance used in the visibility test.
    :param visib_mode: See _estimate_visib_mask.
    :return: Visibility mask.
    """
    visib_est = _estimate_visib_mask(d_test, d_est, delta, visib_mode)
    visib_est = np.logical_or(visib_est, np.logical_and(visib_gt, d_est > 0))
    return visib_est


def render(cad_path, R, t, intrinsic, img_size=[540, 720], open_cv_camera=True):
    # https://github.com/thodan/bop_toolkit/blob/1454e87118109c0715fbd1f0623451429d25b53f/bop_toolkit_lib/dataset_params.py#L221
    ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
    ambient_light = np.asarray([ambient_light[0], ambient_light[1], ambient_light[2], 1])
    scene = pyrender.Scene(bg_color=np.array([1.0, 1.0, 1.0, 0.0]),
                           ambient_light=ambient_light)
    light = pyrender.SpotLight(color=np.ones(3), intensity=0.3, innerConeAngle=np.pi / 16.0,
                               outerConeAngle=np.pi / 6.0)
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000)
    scene.add(light, pose=np.eye(4))

    cam_pose = np.eye(4)
    if open_cv_camera:
        cam_pose[1][1] = -1
        cam_pose[2][2] = -1

    scene.add(camera, pose=cam_pose)
    mesh = trimesh.load_mesh(cad_path)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    obj_pose = np.zeros((4, 4))
    obj_pose[3, 3] = 1
    obj_pose[:3, :3] = R
    obj_pose[:3, 3] = t # in mm
    scene.add(mesh, pose=obj_pose)
    # pyrender.Viewer(scene, use_raymond_lighting=True)
    render_engine = pyrender.OffscreenRenderer(img_size[1], img_size[0])
    flags = pyrender.RenderFlags.DEPTH_ONLY
    depth = render_engine.render(scene, flags=flags)
    return depth # in mm


def vsd(R_est, t_est, R_gt, t_gt, depth_test, K, delta, taus,
        normalized_by_diameter, diameter, eval_cad_path, cost_type='step'):
    # credit: https://github.com/thodan/bop_toolkit/blob/529f11135315cffd536d7b5ea44bcf326daa9a6a/bop_toolkit_lib/
    # pose_error.py#L17
    """
    vsd_delta to calculate visibility
    vsd_tau tolerance for the error (<vsd_tau, pixel is correct)
    """
    """Visible Surface Discrepancy -- by Hodan, Michel et al. (ECCV 2018).
  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param depth_test: hxw ndarray with the test depth image.
  :param K: 3x3 ndarray with an intrinsic camera matrix.
  :param delta: Tolerance used for estimation of the visibility masks.
  :param taus: A list of misalignment tolerance values.
  :param normalized_by_diameter: Whether to normalize the pixel-wise distances
      by the object diameter.
  :param diameter: Object diameter.
  :param renderer: Instance of the Renderer class (see renderer.py).
  :param obj_id: Object identifier.
  :param cost_type: Type of the pixel-wise matching cost:
      'tlinear' - Used in the original definition of VSD in:
          Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16
      'step' - Used for SIXD Challenge 2017 onwards.
  :return: List of calculated errors (one for each misalignment tolerance).
  """
    depth_est = render(cad_path=eval_cad_path, R=R_est, t=t_est, intrinsic=K)
    depth_gt = render(cad_path=eval_cad_path, R=R_gt, t=t_gt, intrinsic=K)
    # Convert depth images to distance images.
    dist_test = depth_im_to_dist_im_fast(depth_test, K)
    dist_gt = depth_im_to_dist_im_fast(depth_gt, K)
    dist_est = depth_im_to_dist_im_fast(depth_est, K)

    # Visibility mask of the model in the ground-truth pose.
    visib_gt = estimate_visib_mask_gt(
        dist_test, dist_gt, delta, visib_mode='bop19')

    # Visibility mask of the model in the estimated pose.
    visib_est = estimate_visib_mask_est(
        dist_test, dist_est, visib_gt, delta, visib_mode='bop19')

    # Intersection and union of the visibility masks.
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()

    # Pixel-wise distances.
    dists = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])

    # Normalization of pixel-wise distances by object diameter.
    if normalized_by_diameter:
        dists /= diameter

    # Calculate VSD for each provided value of the misalignment tolerance.
    if visib_union_count == 0:
        errors = [1.0] * len(taus)
    else:
        errors = []
        for tau in taus:

            # Pixel-wise matching cost.
            if cost_type == 'step':
                costs = dists >= tau
            elif cost_type == 'tlinear':  # Truncated linear function.
                costs = dists / tau
                costs[costs > 1.0] = 1.0
            else:
                raise ValueError('Unknown pixel matching cost.')

            e = (np.sum(costs) + visib_comp_count) / float(visib_union_count)
            errors.append(e)
    return errors[0]


if __name__ == '__main__':
    from lib.utils.config import Config
    parser = argparse.ArgumentParser()
    parser.add_argument('cad_path', nargs='?', help="Path to the model file")
    config_global = Config(config_file="../config.json").get_config()

