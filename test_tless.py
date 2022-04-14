import argparse
import os, sys
import numpy as np
import time
from tqdm import tqdm
# multiprocessing to accelerate the rendering
from functools import partial
import multiprocessing

from lib.utils import gpu_utils, weights, metrics
from lib.utils.config import Config
from lib.datasets.dataloader_utils import init_dataloader
from lib.models.network import FeatureExtractor

from lib.datasets.tless.dataloader_query import Tless
from lib.datasets.tless.dataloader_template import TemplatesTless
from lib.datasets.tless import training_utils, testing_utils
from lib.datasets.tless.eval_utils import eval_vsd

parser = argparse.ArgumentParser()
parser.add_argument('--use_slurm', action='store_true')
parser.add_argument('--use_distributed', action='store_true')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--config_path', type=str)
parser.add_argument('--checkpoint', type=str)
args = parser.parse_args()

config_global = Config(config_file="./config.json").get_config()
config_run = Config(args.config_path).get_config()

# initialize global config for the training
dir_name = (args.config_path.split('/')[-1]).split('.')[0]
print("config", dir_name)
save_path = os.path.join(config_global.root_path, config_run.log.weights, dir_name)
trainer_dir = os.path.join(os.getcwd(), "logs")
tb_logdir = os.path.join(config_global.root_path, config_run.log.tensorboard, dir_name)
trainer_logger, tb_logger, is_master, world_size, local_rank = gpu_utils.init_gpu(use_slurm=args.use_slurm,
                                                                                  use_distributed=args.use_distributed,
                                                                                  local_rank=args.local_rank,
                                                                                  ngpu=args.ngpu,
                                                                                  gpus=args.gpus,
                                                                                  save_path=save_path,
                                                                                  trainer_dir=trainer_dir,
                                                                                  tb_logdir=tb_logdir,
                                                                                  trainer_logger_name="TLESS_test")

# initialize network
model = FeatureExtractor(config_model=config_run.model, threshold=0.2)
model.apply(weights.KaiMingInit)
model.cuda()
# load pretrained weight if backbone are ResNet50
if config_run.model.backbone == "resnet50":
    print("Loading pretrained weights from MOCO...")
    weights.load_pretrained_backbone(prefix="backbone.",
                                     model=model, pth_path=os.path.join(config_global.root_path,
                                                                        config_run.model.pretrained_weights_resnet50))
weights.load_checkpoint(model=model, pth_path=args.checkpoint)

ids = range(1, 31)
config_loader = []
for id_obj in ids:
    config_loader.append(["test", "test_{:02d}".format(id_obj), "query", [id_obj], False])
    config_loader.append(["test", "templates_{:02d}".format(id_obj), "template", id_obj])

datasetLoader = {}
for config in config_loader:
    print("Dataset", config[0], config[1], config[2])
    save_sample_path = os.path.join(config_global.root_path,
                                    config_run.dataset.sample_path, dir_name, config[1])
    if config[2] == "query":
        loader = Tless(root_dir=config_global.root_path, split=config[0], use_augmentation=config[4],
                       list_id_obj=config[3], image_size=config_run.dataset.image_size,
                       save_path=save_sample_path, is_master=is_master)
    else:
        loader = TemplatesTless(root_dir=config_global.root_path, id_obj=config[3],
                                image_size=config_run.dataset.image_size, save_path=save_sample_path,
                                is_master=is_master)
    datasetLoader[config[1]] = loader
    print("---" * 20)

train_sampler, datasetLoader = init_dataloader(dict_dataloader=datasetLoader, use_distributed=args.use_distributed,
                                               batch_size=config_run.train.batch_size,
                                               num_workers=config_run.train.num_workers)

# Run and save prediction into a dataframe
for id_obj in tqdm(ids):
    save_prediction_obj_path = os.path.join(config_global.root_path,
                                            config_run.save_prediction_path, dir_name, "{:02d}".format(id_obj))
    prediction_npz_path = os.path.join(config_global.root_path, save_prediction_obj_path, "epoch_{:02d}".format(0))
    testing_score = testing_utils.test(query_data=datasetLoader["test_{:02d}".format(id_obj)],
                                       template_data=datasetLoader["templates_{:02d}".format(id_obj)],
                                       model=model, id_obj=id_obj,
                                       save_prediction_path=prediction_npz_path,
                                       epoch=0, logger=trainer_logger, tb_logger=tb_logger, is_master=is_master)

# compute VSD metric with multiprocessing to accelerate

# seen objects 1-18
pool = multiprocessing.Pool(processes=config_run.train.num_workers)
seen_objects = range(1, 19)
list_pred_path = []
list_save_path = []
for i in seen_objects:
    pred_obj_path = os.path.join(config_global.root_path, config_run.save_prediction_path, dir_name,
                                 "{:02d}".format(i), "epoch_{:02d}.npz".format(0))
    save_dir = os.path.join(config_global.root_path, config_run.save_prediction_path, dir_name,
                            "{:02d}".format(i), "pred_pose_epoch_{:02d}.npz".format(0))
    list_pred_path.append(pred_obj_path)
    list_save_path.append(save_dir)
eval_vsd_with_index = partial(eval_vsd, list_prediction_path=list_pred_path,
                              list_save_path=list_save_path, root_dir=config_global.root_path)
start_time = time.time()
list_index = range(len(seen_objects))
mapped_values = list(tqdm(pool.imap_unordered(eval_vsd_with_index, list_index), total=len(list_index)))
finish_time = time.time()
seen_scores = []
for score in mapped_values:
    seen_scores.extend(score)
print("Final score of seen object: {}".format(np.mean(seen_scores)))
print("Total time to evaluate T-LESS on seen objects ", finish_time - start_time)

# unseen objects 19-31
unseen_objects = range(19, 31)
list_pred_path = []
list_save_path = []

for i in unseen_objects:
    pred_obj_path = os.path.join(config_global.root_path, config_run.save_prediction_path, dir_name,
                                 "{:02d}".format(i), "epoch_{:02d}.npz".format(0))
    save_dir = os.path.join(config_global.root_path, config_run.save_prediction_path, dir_name,
                            "{:02d}".format(i), "pred_pose_epoch_{:02d}.npz".format(0))
    list_pred_path.append(pred_obj_path)
    list_save_path.append(save_dir)
eval_vsd_with_index = partial(eval_vsd, list_prediction_path=list_pred_path,
                              list_save_path=list_save_path, root_dir=config_global.root_path)
start_time = time.time()
list_index = range(len(unseen_objects))
mapped_values = list(tqdm(pool.imap_unordered(eval_vsd_with_index, list_index), total=len(list_index)))
finish_time = time.time()
unseen_scores = []
for score in mapped_values:
    unseen_scores.extend(score)
print("Final score of unseen object: {}".format(np.mean(unseen_scores)))
print("Total time to evaluate T-LESS on unseen objects ", finish_time - start_time)
