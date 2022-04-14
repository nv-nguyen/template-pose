import blenderproc as bproc
import numpy as np
import argparse
import os, sys
import imageio
from PIL import Image


def render_blender_proc(cad_path, output_dir, obj_poses, img_size, intrinsic,
                        scale_meter=True, recenter_origin=False, is_tless=False):
    bproc.init()
    if scale_meter:
        # convert to scale meter to get better light condition (instead of using light.set_energy(+infinity))
        obj_poses[:, :3, 3] *= 0.001
        obj_poses[:, :3, :3] *= 0.001

    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(np.eye(4), ["X", "-Y", "-Z"])
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_intrinsics_from_K_matrix(
        intrinsic, img_size[1], img_size[0]
    )
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([1, -1, 1])
    light.set_energy(200)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-1, -1, -1])
    light.set_energy(200)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-1, 0, -1])
    light.set_energy(20)
    light.set_type("POINT")
    light.set_location([1, 0, 1])
    light.set_energy(20)

    # load the objects into the scene
    obj = bproc.loader.load_obj(cad_path)[0]
    if recenter_origin:
        # recenter origin of object at center of its bounding box
        import bpy
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    if is_tless:
        mat = obj.get_materials()[0]
        grey_col = 0.4  # np.random.uniform(0.1, 0.9)
        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
    else:
        # Use vertex color for texturing
        for mat in obj.get_materials():
            mat.map_vertex_color()

    obj.set_cp("category_id", 1)
    # activate normal and distance rendering
    bproc.renderer.enable_distance_output(True)
    # set the amount of samples, which should be used for the color rendering
    # bproc.rendererset_max_amount_of_samples(100)
    black_img = Image.new('RGB', (img_size[1], img_size[0]))
    for idx_frame, obj_pose in enumerate(obj_poses):
        obj.set_local2world_mat(obj_pose)
        data = bproc.renderer.render()
        data.update(bproc.renderer.render_segmap(map_by="class", use_alpha_channel=True))
        # # Map distance to depth
        depth = bproc.postprocessing.dist2depth(data["distance"])[0]
        mask = np.uint8((depth < 1000) * 255)
        mask = Image.fromarray(mask)
        mask.save(os.path.join(output_dir, "mask_{:06d}.png".format(idx_frame)))

        rgb = Image.fromarray(np.uint8(data["colors"][0]))
        img = Image.composite(rgb, black_img, mask)
        img.save(os.path.join(output_dir, "{:06d}.png".format(idx_frame)))
    # imageio.imwrite(os.path.join(output_dir, "{:06d}.png".format(idx_frame)), )
    # bproc.writer.write_bop(output_dir, depth, data["colors"], m2mm=True, append_to_existing_output=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cad_path', nargs='?', help="Path to the model file")
    parser.add_argument('obj_pose', nargs='?', help="Path to the model file")
    parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved")
    parser.add_argument('disable_output', nargs='?', help="Disable output of blender")
    args = parser.parse_args()

    poses = np.load(args.obj_pose)
    if "tless" in args.output_dir:
        intrinsic = np.asarray([1075.65091572, 0.0, 360,
                                0.0, 1073.90347929, 270,
                                0.0, 0.0, 1.0]).reshape(3, 3)
        img_size = [540, 720]
        is_tless = True
    else:
        intrinsic = np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]])
        img_size = [480, 640]
        is_tless = False

    if args.disable_output == "true":
        # redirect output to log file
        logfile = os.path.join(args.output_dir, 'render.log')
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
    # scale_meter do not change the binary mask but recenter_origin change it
    render_blender_proc(args.cad_path, args.output_dir, poses, intrinsic=intrinsic, img_size=img_size,
                        scale_meter=True, recenter_origin=True, is_tless=is_tless)
    if args.disable_output == "true":
        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
        os.system("rm {}".format(logfile))
