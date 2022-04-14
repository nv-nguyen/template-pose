# Adapted from BOP toolkit:  https://github.com/thodan/bop_toolkit
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import numpy as np


def load_info(path):
    with open(path, 'r') as f:
        info = yaml.load(f, Loader=yaml.CLoader)
        for eid in info.keys():
            if 'cam_K' in info[eid].keys():
                info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape(
                    (3, 3))
            if 'cam_R_w2c' in info[eid].keys():
                info[eid]['cam_R_w2c'] = np.array(
                    info[eid]['cam_R_w2c']).reshape((3, 3))
            if 'cam_t_w2c' in info[eid].keys():
                info[eid]['cam_t_w2c'] = np.array(
                    info[eid]['cam_t_w2c']).reshape((3, 1))
    return info


def save_info(path, info, save_all=False):
    if not save_all:
        for im_id in sorted(info.keys()):
            im_info = info[im_id]
            if 'cam_K' in im_info.keys():
                im_info['cam_K'] = im_info['cam_K'].flatten().tolist()
            if 'cam_R_w2c' in im_info.keys():
                im_info['cam_R_w2c'] = im_info['cam_R_w2c'].flatten().tolist()
            if 'cam_t_w2c' in im_info.keys():
                im_info['cam_t_w2c'] = im_info['cam_t_w2c'].flatten().tolist()
        with open(path, 'w') as f:
            yaml.dump(info, f, Dumper=yaml.CDumper, width=10000)
    else:
        for im_id in sorted(info.keys()):
            im_info = info[im_id]
            for key in im_info.keys():
                if key in ['obj_bb', 'cam_R_m2c', 'cam_t_m2c', 'cam_K']:
                    im_info[key] = np.asarray(im_info[key]).flatten().tolist()
        with open(path, 'w') as f:
            yaml.dump(info, f, Dumper=yaml.CDumper, width=10000)


def load_gt(path):
    with open(path, 'r') as f:
        gts = yaml.load(f, Loader=yaml.CLoader)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
    return gts