import torch
import os
import subprocess
import socket
import numpy as np
import random
from lib.utils import logger
import os
from torch.nn.parallel import DistributedDataParallel


class MyDataParallel(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def init_distributed(local_rank, world_size):
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl',
                                         init_method='env://',
                                         world_size=world_size,
                                         rank=local_rank)
    print("Initialize local_rank={}, world_size={} done!".format(local_rank, world_size))


def init_distributed_slurm():
    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])

    # local rank on the current node / global rank
    local_rank = int(os.environ['SLURM_LOCALID'])
    global_rank = int(os.environ['SLURM_PROCID'])

    # number of processes / GPUs per node
    world_size = int(os.environ['SLURM_NTASKS'])
    n_gpu_per_node = world_size // n_nodes

    # define master address and master port
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    master_addr = hostnames.split()[0].decode('utf-8')

    # set environment variables for 'env://'
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(29500)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)

    # define whether this is the master process / if we are in distributed mode
    is_master = node_id == 0 and local_rank == 0
    multi_node = n_nodes > 1
    multi_gpu = world_size > 1

    # summary
    PREFIX = "%i - " % global_rank
    print(PREFIX + "Number of nodes: %i" % n_nodes)
    print(PREFIX + "Node ID        : %i" % node_id)
    print(PREFIX + "Local rank     : %i" % local_rank)
    print(PREFIX + "Global rank    : %i" % global_rank)
    print(PREFIX + "World size     : %i" % world_size)
    print(PREFIX + "GPUs per node  : %i" % n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(is_master))
    print(PREFIX + "Multi-node     : %s" % str(multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(multi_gpu))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(local_rank)

    print("Initializing PyTorch distributed ...")
    torch.distributed.init_process_group(
        init_method='env://',
        backend='nccl',
    )
    print("Initialize local_rank={}, world_size={} done!".format(local_rank, world_size))
    return is_master, local_rank, world_size


def init_gpu(use_slurm, use_distributed, local_rank, ngpu, gpus, save_path, trainer_dir, tb_logdir, trainer_logger_name=None):
    if not os.path.exists(trainer_dir):
        os.makedirs(trainer_dir)
    if use_slurm:
        if use_distributed:
            print("Running at cluster with Slurm and Distributed!")
            is_master, local_rank, world_size = init_distributed_slurm()
        else:
            print("Running at cluster with Slurm and NOT Distributed!")
            is_master, local_rank, world_size = True, 0, 1
        if is_master:
            print("Logging...")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if trainer_logger_name is not None:
                trainer_logger = logger.get_logger(trainer_dir, trainer_logger_name)
            else:
                trainer_logger = logger.get_logger(trainer_dir, "trainer")
            logger.print_and_log_info(trainer_logger, "Model's weights at {}".format(save_path))
            logger.print_and_log_info(trainer_logger, "Logdir is saved at {}".format(tb_logdir))
            tb_logger = logger.TensorboardLogger(logdir=tb_logdir)
        else:
            trainer_logger, tb_logger = None, None
        return trainer_logger, tb_logger, is_master, world_size, local_rank
    else:
        if local_rank == 0:
            print("Not running at cluster (no Slurm)")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if trainer_logger_name is not None:
                trainer_logger = logger.get_logger(trainer_dir, trainer_logger_name)
            else:
                trainer_logger = logger.get_logger(trainer_dir, "trainer")
            logger.print_and_log_info(trainer_logger, "Model's weights at {}".format(save_path))
            logger.print_and_log_info(trainer_logger, "Logdir is saved at {}".format(tb_logdir))
            tb_logger = logger.TensorboardLogger(logdir=tb_logdir)
            is_master = True
        else:
            is_master = False
            trainer_logger, tb_logger = None, None
        if is_master:
            logger.print_and_log_info(trainer_logger, "Using gpus {}".format(gpus))
        if use_distributed:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            world_size = ngpu
            init_distributed(trainer_logger, local_rank, world_size)
        else:
            print("No distributed!!!")
            is_master = True
            world_size = 1
        return trainer_logger, tb_logger, is_master, world_size, local_rank
