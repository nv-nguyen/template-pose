import time
import os
from functools import partial
import multiprocessing
import argparse
from lib.datasets.linemod import inout, processing_utils
from lib.utils.config import Config
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Template Matching cropping image scripts for LINEMOD')
    parser.add_argument('--config', type=str, default="./config.json")
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    config = Config(config_file=args.config).get_config()
    pool = multiprocessing.Pool(processes=args.num_workers)
    # crop image for LINEMOD objects
    for dataset in ["LINEMOD", "occlusionLINEMOD"]:
        if dataset == "LINEMOD":
            list_obj = range(13)
        elif dataset == "occlusionLINEMOD":
            list_obj = range(8)
        for crop_size in [64, 224]:
            start_time = time.time()
            save_dir = os.path.join(config.root_path, "crop_image{}".format(crop_size), dataset)
            crop_dataset_with_index = partial(processing_utils.crop_dataset, dataset=dataset,
                                              root_path=config.root_path, save_dir=save_dir,
                                              crop_size=crop_size, split=None)
            mapped_values = list(
                tqdm(pool.imap_unordered(crop_dataset_with_index, range(len(list_obj))), total=len(list_obj)))
            finish_time = time.time()
            print("Total time to crop images:", finish_time - start_time)

    # crop image for template images
    list_obj = range(13)
    for split in ["train", "test"]:
        for crop_size in [64, 224]:
            start_time = time.time()
            save_dir = os.path.join(config.root_path, "crop_image{}".format(crop_size), "templatesLINEMOD", split)
            crop_dataset_with_index = partial(processing_utils.crop_dataset, dataset="templatesLINEMOD",
                                              root_path=config.root_path, save_dir=save_dir,
                                              crop_size=crop_size, split=split)
            mapped_values = list(
                tqdm(pool.imap_unordered(crop_dataset_with_index, range(len(list_obj))), total=len(list_obj)))
            finish_time = time.time()
            print("Total time to crop image:", finish_time - start_time)