import argparse
import os, sys
import torch
from tqdm import tqdm

from lib.utils import gpu_utils, weights, metrics
from lib.utils.config import Config
from lib.datasets.dataloader_utils import init_dataloader
from lib.utils.optimizer import adjust_learning_rate

from lib.models.network import FeatureExtractor

from lib.datasets.tless.dataloader_query import Tless
from lib.datasets.tless.dataloader_template import TemplatesTless
from lib.datasets.tless import training_utils, testing_utils


parser = argparse.ArgumentParser()
parser.add_argument('--use_slurm', action='store_true')
parser.add_argument('--use_distributed', action='store_true')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--config_path', type=str)
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
                                                                                  trainer_logger_name=dir_name)

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

seen_ids, unseen_ids = range(1, 18), range(19, 31)
config_loader = [["train", "train", "query", seen_ids, config_run.dataset.use_augmentation]]
for id_obj in unseen_ids:
    config_loader.append(["test", "test_{:02d}".format(id_obj), "query", [id_obj], False])
    config_loader.append(["test", "templates_{:02d}".format(id_obj), "template", id_obj])

datasetLoader = {}
for config in config_loader:
    print("Dataset", config[0], config[1], config[2])
    save_sample_path = os.path.join(config_global.root_path,
                                    config_run.dataset.sample_path, dir_name, config[1])
    if config[2] == "query":
        loader = Tless(root_dir=config_global.root_path, split=config[0], use_augmentation=config[4],
                       list_id_obj=config[3],
                       image_size=config_run.dataset.image_size, save_path=save_sample_path, is_master=is_master)
    else:
        loader = TemplatesTless(root_dir=config_global.root_path, id_obj=config[3],
                                image_size=config_run.dataset.image_size, save_path=save_sample_path,
                                is_master=is_master)
    datasetLoader[config[1]] = loader
    print("---" * 20)

train_sampler, datasetLoader = init_dataloader(dict_dataloader=datasetLoader, use_distributed=args.use_distributed,
                                               batch_size=config_run.train.batch_size,
                                               num_workers=config_run.train.num_workers)

# initialize optimizer
optimizer = torch.optim.Adam(list(model.parameters()), lr=config_run.train.optimizer.lr, weight_decay=0.0005)
scores = metrics.init_score()

for epoch in tqdm(range(0, 25)):
    if args.use_slurm and args.use_distributed:
        train_sampler.set_epoch(epoch)

    # update learning rate
    if epoch in config_run.train.scheduler.milestones:
        adjust_learning_rate(optimizer, config_run.train.optimizer.lr, config_run.train.scheduler.gamma)

    if epoch % 3 == 0 and epoch > 0:
        for id_obj in unseen_ids:
            save_prediction_obj_path = os.path.join(config_global.root_path,
                                                    config_run.save_prediction_path, dir_name, "{:02d}".format(id_obj))
            testing_score = testing_utils.test(query_data=datasetLoader["test_{:02d}".format(id_obj)],
                                               template_data=datasetLoader["templates_{:02d}".format(id_obj)],
                                               model=model, id_obj=id_obj,
                                               save_prediction_path=os.path.join(config_global.root_path,
                                                                                 save_prediction_obj_path,
                                                                                 "epoch_{:02d}".format(epoch)),
                                               epoch=epoch,
                                               logger=trainer_logger, tb_logger=tb_logger, is_master=is_master)

    train_loss = training_utils.train(train_data=datasetLoader["train"],
                                      model=model, optimizer=optimizer,
                                      warm_up_config=[1000, config_run.train.optimizer.lr],
                                      epoch=epoch, logger=trainer_logger, tb_logger=tb_logger,
                                      log_interval=config_run.log.log_interval,
                                      is_master=is_master)

    text = '\nEpoch-{}: train_loss={} \n\n'
    if is_master:
        weights.save_checkpoint({'model': model.state_dict()},
                                os.path.join(save_path, 'model_epoch{}.pth'.format(epoch)))
