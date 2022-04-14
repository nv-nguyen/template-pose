import argparse
import os, sys
import torch

from lib.utils import gpu_utils, weights, metrics
from lib.utils.config import Config

from lib.models.base_network import BaseFeatureExtractor
from lib.models.network import FeatureExtractor
from lib.datasets.dataloader_utils import init_dataloader

from lib.datasets.linemod.dataloader_query import LINEMOD
from lib.datasets.linemod.dataloader_template import TemplatesLINEMOD

from lib.datasets.linemod import inout
from lib.datasets.linemod import training_utils, testing_utils

parser = argparse.ArgumentParser()
parser.add_argument('--use_slurm', action='store_true')
parser.add_argument('--use_distributed', action='store_true')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--split', type=str, choices=['split1', 'split2', 'split3'])
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
                                                                                  trainer_logger_name="test")
# initialize network
# load pretrained weight if backbone are ResNet50
if config_run.model.backbone == "resnet50":
    model = FeatureExtractor(config_model=config_run.model, threshold=0.2)
    model.apply(weights.KaiMingInit)
    model.cuda()
    print("Loading pretrained weights from MOCO...")
    weights.load_pretrained_backbone(prefix="backbone.",
                                     model=model, pth_path=os.path.join(config_global.root_path,
                                                                        config_run.model.pretrained_weights_resnet50))
else:
    model = BaseFeatureExtractor(config_model=config_run.model, threshold=0.2)
    model.apply(weights.KaiMingInit)
    model.cuda()

# load checkpoint if it's available
if args.checkpoint is not None:
    print("Loading checkpoint...")
    weights.load_checkpoint(model=model, pth_path=args.checkpoint)

# create dataloader for query wo occlusion: train_loader, (test_seen_loader, test_unseen_loader)
# query with occlusion: (test_seen_occ_loader, test_unseen_occ_loader),
# template: (template_loader,  template_unseen_loader)
seen_id_obj, seen_names, seen_occ_id_obj, seen_occ_names, unseen_id_obj, unseen_names, \
unseen_occ_id_obj, unseen_occ_names = inout.get_list_id_obj_from_split_name(config_run.dataset.split)
config_loader = [["seen_test", "seen_test", "LINEMOD", seen_id_obj],
                 ["unseen_test", "test", "LINEMOD", unseen_id_obj],
                 ["seen_template", "test", "templatesLINEMOD", seen_id_obj],
                 ["unseen_template", "test", "templatesLINEMOD", unseen_id_obj],

                 ["seen_occ_test", "test", "occlusionLINEMOD", seen_occ_id_obj],
                 ["unseen_occ_test", "test", "occlusionLINEMOD", unseen_occ_id_obj],
                 ["seen_occ_template", "test", "templatesLINEMOD", seen_occ_id_obj],
                 ["unseen_occ_template", "test", "templatesLINEMOD", unseen_occ_id_obj]]

datasetLoader = {}
for config in config_loader:
    print("Dataset", config[0], config[2], config[3])
    save_sample_path = os.path.join(config_global.root_path, config_run.dataset.sample_path, dir_name,
                                    config[0])
    if config[2] == "templatesLINEMOD":
        loader = TemplatesLINEMOD(root_dir=config_global.root_path, dataset=config[2], list_id_obj=config[3],
                                  split=config[1], image_size=config_run.dataset.image_size,
                                  save_path=save_sample_path, is_master=is_master)
    else:
        loader = LINEMOD(root_dir=config_global.root_path,
                         dataset=config[2], list_id_obj=config[3], split=config[1],
                         image_size=config_run.dataset.image_size, save_path=save_sample_path,
                         is_master=is_master)
    datasetLoader[config[0]] = loader
    print("---" * 20)

train_sampler, datasetLoader = init_dataloader(dict_dataloader=datasetLoader,
                                               use_distributed=args.use_distributed,
                                               batch_size=config_run.train.batch_size,
                                               num_workers=config_run.train.num_workers)

new_score = {}
for config_split in [["seen", seen_id_obj], ["seen_occ", seen_occ_id_obj],
                     ["unseen", unseen_id_obj], ["unseen_occ", unseen_occ_id_obj]]:
    query_name = config_split[0] + "_test"
    template_name = config_split[0] + "_template"
    testing_score = testing_utils.test(query_data=datasetLoader[query_name],
                                       template_data=datasetLoader[template_name],
                                       model=model, split_name=config_split[0],
                                       list_id_obj=config_split[1].tolist(), epoch=0,
                                       logger=trainer_logger,
                                       tb_logger=tb_logger, is_master=is_master)
    new_score[config_split[0] + "_err"] = testing_score[0]
    new_score[config_split[0] + "_acc"] = testing_score[-1]
print(new_score)