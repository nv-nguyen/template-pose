from tqdm import tqdm
import os
import argparse
import time
from functools import partial
import multiprocessing
from lib.utils.config import Config
from lib.datasets.tless.processing_utils import create_gt_obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process T-LESS dataset to have gt for each object')
    parser.add_argument('--config', type=str, default="./config.json")
    parser.add_argument('--num_workers', type=int, default=10)
    args = parser.parse_args()

    config = Config(config_file=args.config).get_config()
    root_tless_path = os.path.join(config.root_path, config.TLESS.local_path)

    # create GT per objects for test scene
    pool = multiprocessing.Pool(processes=args.num_workers)
    save_path = os.path.join(root_tless_path, "opencv_pose")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    start_time = time.time()
    create_gt_obj_with_index = partial(create_gt_obj,
                                       list_id_obj=range(1, 31),
                                       split="train",
                                       root_path=root_tless_path)
    mapped_values = list(tqdm(pool.imap_unordered(create_gt_obj_with_index, range(30)), total=30))
    finish_time = time.time()
    print("Total time to create GT pose for each object of training set:", finish_time - start_time)

    start_time = time.time()
    create_gt_obj_with_index = partial(create_gt_obj,
                                       list_id_obj=range(1, 31),
                                       split="test",
                                       root_path=root_tless_path)
    mapped_values = list(tqdm(pool.imap_unordered(create_gt_obj_with_index, range(30)), total=30))
    finish_time = time.time()
    print("Total time to create GT pose for each object of test set:", finish_time - start_time)