import numpy as np
import matplotlib.pyplot as plt
import os
from lib.datasets import image_utils
from lib.datasets.tless import inout


def visualization_gt_templates(root_path, split, id_scene, id_frame, id_obj, idx_obj_in_scene, index_nearest_template,
                               gt_inplane, nearest_inplane, best_error):
    # get query image
    query = inout.open_real_image_tless(root_path=root_path, split=split, id_scene=id_scene, id_frame=id_frame,
                                        idx_obj_in_scene=idx_obj_in_scene, image_type="rgb")
    query_mask = inout.open_real_image_tless(root_path=root_path, split=split, id_scene=id_scene, id_frame=id_frame,
                                             idx_obj_in_scene=idx_obj_in_scene, image_type="mask")
    query = image_utils.crop_image(query, bbox=query_mask.getbbox(), keep_aspect_ratio=True)
    query = query.resize((128, 128))

    # get template with nearest inplane rotation
    nearest_template = inout.open_template_tless(root_path=root_path, id_obj=id_obj,
                                                 idx_template=index_nearest_template,
                                                 image_type="rgb", inplane=nearest_inplane)
    nearest_mask_cad = inout.open_template_tless(root_path=root_path, id_obj=id_obj,
                                                 idx_template=index_nearest_template,
                                                 image_type="mask", inplane=nearest_inplane)
    nearest_template = image_utils.crop_image(nearest_template, bbox=nearest_mask_cad.getbbox(), keep_aspect_ratio=True)
    nearest_template = nearest_template.resize((128, 128))

    # get template with gt inplane rotation
    gt_template = inout.open_template_tless(root_path=root_path, id_obj=id_obj, idx_template=index_nearest_template,
                                            image_type="rgb", inplane=gt_inplane)
    gt_mask_cad = inout.open_template_tless(root_path=root_path, id_obj=id_obj, idx_template=index_nearest_template,
                                            image_type="mask", inplane=gt_inplane)
    gt_template = image_utils.crop_image(gt_template, bbox=gt_mask_cad.getbbox(), keep_aspect_ratio=True)
    gt_template = gt_template.resize((128, 128))

    plt.figure(figsize=(5, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(query)
    plt.axis('off')
    plt.title("Query, Err={:.2f}".format(best_error), fontsize=10)

    plt.subplot(1, 3, 2)
    plt.imshow(nearest_template)
    plt.axis('off')
    plt.title("Nearest inp={}".format(nearest_inplane), fontsize=10)

    plt.subplot(1, 3, 3)
    plt.imshow(gt_template)
    plt.axis('off')
    plt.title("GT inp={:.2f}".format(gt_inplane), fontsize=10)

    save_dir = os.path.join(root_path, "visualization_tless", "{:02d}".format(id_obj))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # print(os.path.join(save_dir, "{:06d}_{:06d}.png".format(id_scene, id_frame)))
    plt.savefig(os.path.join(save_dir, "{:06d}_{:06d}.png".format(id_scene, id_frame)), bbox_inches='tight', dpi=100)
    plt.close("all")