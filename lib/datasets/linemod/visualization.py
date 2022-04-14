import os
import matplotlib.pyplot as plt
from PIL import Image


def visualize_gt_templates(crop_dir, dataset, obj_name, idx_frame, idx_test_template, idx_train_template, test_error,
                           train_error):
    save_dir = os.path.join(crop_dir, "visualization_linemod", dataset, obj_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    query_img = Image.open(os.path.join(crop_dir, dataset, obj_name, '{:06d}.png'.format(idx_frame)))
    test_template = Image.open(os.path.join(crop_dir, "templatesLINEMOD", "test", obj_name,
                                            '{:06d}.png'.format(idx_test_template)))
    train_template = Image.open(os.path.join(crop_dir, "templatesLINEMOD", "train", obj_name,
                                             '{:06d}.png'.format(idx_train_template)))
    plt.figure(figsize=(5, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(query_img)
    plt.axis('off')
    plt.title("Query")

    plt.subplot(1, 3, 2)
    plt.imshow(test_template)
    plt.axis('off')
    plt.title("Test, Err={:.2f}".format(test_error))

    plt.subplot(1, 3, 3)
    plt.imshow(train_template)
    plt.axis('off')
    plt.title("Train, Err={:.2f}".format(train_error))

    plt.savefig(os.path.join(save_dir, "{:06d}.png".format(idx_frame)), bbox_inches='tight', dpi=100)
    plt.close("all")