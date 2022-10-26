import bpy
import bmesh
import math
import numpy as np
import mathutils
import os
from math import radians


def get_camera_positions(nSubDiv):
    """
    * Construct an icosphere
    * subdived
    """

    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0), enter_editmode=True)
    # bpy.ops.export_mesh.ply(filepath='./sphere.ply')
    icos = bpy.context.object
    me = icos.data

    # -- cut away lower part
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] < 0]

    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # -- subdivide and move new vertices out to the surface of the sphere
    #    nSubDiv = 3
    for i in range(nSubDiv):
        bpy.ops.mesh.subdivide()

        bm = bmesh.from_edit_mesh(me)
        for v in bm.verts:
            l = math.sqrt(v.co[0] ** 2 + v.co[1] ** 2 + v.co[2] ** 2)
            v.co[0] /= l
            v.co[1] /= l
            v.co[2] /= l
        bmesh.update_edit_mesh(me)

    # -- cut away zero elevation
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] <= 0]
    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # convert vertex positions to az,el
    positions = []
    angles = []
    bm = bmesh.from_edit_mesh(me)
    for v in bm.verts:
        x = v.co[0]
        y = v.co[1]
        z = v.co[2]
        az = math.atan2(x, y)  # *180./math.pi
        el = math.atan2(z, math.sqrt(x ** 2 + y ** 2))  # *180./math.pi
        # positions.append((az,el))
        angles.append((el, az))
        positions.append((x, y, z))

    bpy.ops.object.editmode_toggle()

    # sort positions, first by az and el
    data = zip(angles, positions)
    positions = sorted(data)
    positions = [y for x,y in positions]
    angles = sorted(angles)
    return angles, positions


def convert_location_to_rotation(locations):
    obj_poses = np.zeros((len(locations), 4, 4))
    for idx, pt in enumerate(locations):
        f = -np.array(locations[idx])  # Forward direction.
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
        t = pt  # R.dot(np.array(pt).reshape((3, 1)))
        obj_poses[idx, :3, :3] = R
        obj_poses[idx, :3, 3] = t.reshape(-1)
        obj_poses[idx, 3, 3] = 1
    return obj_poses

# to get predefined poses for level 3
level = 3
sphere_level3 = np.load("./lib/poses/predefined_poses/sphere_level3.npy")
position_icosphere = np.asarray(get_camera_positions(level)[1])
matrix = convert_location_to_rotation(position_icosphere)
print(matrix.shape)
print("Reproducing errors", np.sum(matrix-sphere_level3))