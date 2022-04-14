import numpy as np
from mathutils import Euler, Matrix, Vector
from scipy.linalg import logm
import math


def inverse_matrix_world(matrix_4x4):
    """
    Inverse matrix transform
    """
    rotation = matrix_4x4[:3, :3]
    translation = matrix_4x4[:3, 3]
    r_transpose_x = rotation[0, 0] * translation[0] + rotation[1, 0] * translation[1] + rotation[2, 0] * translation[2]
    r_transpose_y = rotation[0, 1] * translation[0] + rotation[1, 1] * translation[1] + rotation[2, 1] * translation[2]
    r_transpose_z = rotation[0, 2] * translation[0] + rotation[1, 2] * translation[1] + rotation[2, 2] * translation[2]
    matrix_world_inverse = np.array([
        [rotation[0, 0], rotation[1, 0], rotation[2, 0], -r_transpose_x],
        [rotation[0, 1], rotation[1, 1], rotation[2, 1], -r_transpose_y],
        [rotation[0, 2], rotation[1, 2], rotation[2, 2], -r_transpose_z],
        [0, 0, 0, 1.0]])
    return matrix_world_inverse


def opencv2opengl(cam_matrix_world):
    """
    Change coordinate system from OpenCV to OpenGL or from OpenGL to OpenCV
    """
    from scipy.spatial.transform import Rotation as R
    rot180x = R.from_euler('x', 180, degrees=True).as_matrix()
    rotation = cam_matrix_world[:3, :3]
    translation = cam_matrix_world[:3, 3]
    output = np.copy(cam_matrix_world)
    output[:3, :3] = np.asarray(Matrix(rot180x) @ Matrix(rotation).to_3x3())
    output[:3, 3] = np.asarray(Matrix(rot180x) @ Vector(translation))
    return output


def get_camera_location_from_obj_pose(obj_pose):
    """
    R_tranpose x (-T)
    """
    trans = obj_pose[:3, 3]
    T_cam = obj_pose[:3, :3].T.dot(-trans)
    T_cam = T_cam / np.linalg.norm(T_cam)
    return T_cam


def look_at(location):
    """
    Get object pose from a viewpoint location
    # Taken from https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/view_sampler.py#L216
    IMPORTANT: output of this function is the object pose defined in OPENGL coordinate convention
    """
    f = -np.array(location)  # Forward direction.
    f /= np.linalg.norm(f)

    u = np.array([0.0, 0.0, 1.0])  # Up direction.
    s = np.cross(f, u)  # Side direction.
    if np.count_nonzero(s) == 0:
        # f and u are parallel, i.e. we are looking along or against Z axis.
        s = np.array([1.0, 0.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)  # Recompute up.
    R = np.array([[s[0], s[1], s[2]],
                  [u[0], u[1], u[2]],
                  [-f[0], -f[1], -f[2]]])
    t = - R.dot(np.array(location).reshape((3, 1)))
    obj_pose = np.zeros((4, 4))
    obj_pose[:3, :3] = R
    obj_pose[:3, 3] = -t.reshape(-1)
    obj_pose[3, 3] = 1
    return obj_pose


def remove_inplane_rotation(opencv_pose, return_symmetry_rot=False):
    """
    TODO: this function can be improved and simplified
    """
    cam_location = get_camera_location_from_obj_pose(opencv_pose)
    obj_opengl_pose = look_at(cam_location)
    opencv_pose_wo_inplane = opencv2opengl(obj_opengl_pose)
    opencv_pose_wo_inplane[:3, 3] = opencv_pose[:3, 3]  # do not change the translation
    if return_symmetry_rot:
        opposite_cam_location = cam_location
        opposite_cam_location[:2] *= -1
        obj_opengl_pose_opposite = look_at(opposite_cam_location)
        opencv_pose_wo_inplane_opposite = opencv2opengl(obj_opengl_pose_opposite)
        opencv_pose_wo_inplane_opposite[:3, 3] = opencv_pose[:3, 3]  # do not change the translation
        return opencv_pose_wo_inplane, opencv_pose_wo_inplane_opposite
    else:
        return opencv_pose_wo_inplane


def perspective(K, obj_pose, pts):
    results = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        R, T = obj_pose[:3, :3], obj_pose[:3, 3]
        rep = np.matmul(K, np.matmul(R, pts[i].reshape(3, 1)) + T.reshape(3, 1))
        x = np.int32(rep[0] / rep[2])  # as matplot flip  x axis
        y = np.int32(rep[1] / rep[2])
        results[i] = [x, y]
    return results


def geodesic_numpy(R1, R2):
    theta = (np.trace(R2.dot(R1.T)) - 1) / 2
    theta = np.clip(theta, -1, 1)
    return np.degrees(np.arccos(theta))