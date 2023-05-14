import logging
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch.nn as nn
from src.utils.weight import load_checkpoint
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset
import trimesh
import pyrender
import glob
import numpy as np

pl.seed_everything(2022)
# set level logging
logging.basicConfig(level=logging.INFO)


def load_mesh(cad_path):
    mesh = trimesh.load_mesh(cad_path)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    return mesh


@hydra.main(version_base=None, config_path="configs", config_name="train")
def test(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg["runtime"]["output_dir"]
    os.makedirs(cfg.callback.checkpoint.dirpath, exist_ok=True)
    logging.info(
        f"Training script. The outputs of hydra will be stored in: {output_path}"
    )
    logging.info(f"Checkpoints will be stored in: {cfg.callback.checkpoint.dirpath}")

    # Delayed imports to get faster parsing
    from hydra.utils import instantiate

    logging.info("Initializing logger, callbacks and trainer")
    os.environ["WANDB_API_KEY"] = cfg.user.wandb_api_key
    if cfg.machine.dryrun:
        os.environ["WANDB_MODE"] = "offline"
    logging.info(f"Wandb logger initialized at {cfg.save_dir}")

    if cfg.machine.name == "slurm":
        num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        num_nodes = int(os.environ["SLURM_NNODES"])
        cfg.machine.trainer.devices = num_gpus
        cfg.machine.trainer.num_nodes = num_nodes
        logging.info(f"Slurm config: {num_gpus} gpus,  {num_nodes} nodes")
    num_devices = cfg.machine.trainer.devices
    cfg.machine.trainer.devices = [0]  # testing run only on single gpu
    trainer = instantiate(cfg.machine.trainer)
    logging.info(f"Trainer initialized")

    model = instantiate(cfg.model)
    logging.info(f"Model '{cfg.model.modelname}' loaded")
    if cfg.model.pretrained_weight is not None:
        load_checkpoint(
            model.backbone,
            cfg.model.pretrained_weight,
            prefix="",
            checkpoint_key="model",
        )
    for obj_id in range(1, 31):
        test_datasets = []
        for mode in ["query", "template"]:
            config_dataloader = cfg.data.tless_test.dataloader
            config_dataloader.reset_metaData = False
            config_dataloader.split = "test_primesense"
            config_dataloader.obj_id = int(obj_id)
            config_dataloader.mode = mode
            config_dataloader.batch_size = (
                cfg.machine.batch_size * num_devices if mode == "query" else None
            )
            test_dataset = instantiate(config_dataloader)
            if mode == "query":
                assert (
                    len(test_dataset) % cfg.machine.batch_size * num_devices == 0
                ), logging.warning(
                    f"test_dataset size {len(test_dataset)} should be divided by {cfg.machine.batch_size} to load templates correctly"
                )
            test_datasets.append(test_dataset)

        test_datasets = ConcatDataset(test_datasets)
        test_dataloaders = DataLoader(
            test_datasets,
            batch_size=cfg.machine.batch_size,
            num_workers=cfg.machine.num_workers,
            shuffle=False,
        )
        logging.info(f"Testing object {obj_id+1}: test_size={len(test_dataloaders)}...")
        model.metric_eval = "vsd"
        model.dataset_name = "tless"
        model.obj_id = obj_id
        model.testing_cad = load_mesh(
            f"{config_dataloader.root_dir}/models/models_eval/obj_{obj_id:06d}.ply"
        )
        trainer.num_workers = cfg.machine.num_workers
        trainer.test(
            model,
            dataloaders=test_dataloaders,
            ckpt_path=cfg.model.checkpoint_path,
        )
        logging.info(f"---" * 20)

    # calculatin mean vsd
    result_dir = model.log_dir
    vsd_files = []
    for obj_group in [range(1, 19), range(19, 31)]:
        for obj_id in obj_group:
            tmp = glob.glob(f"{result_dir}/vsd_obj_{obj_id}*.npy")
            vsd_files.extend(tmp)
        list_vsd = []
        for vsd_file in vsd_files:
            vsd = np.load(vsd_file)
            list_vsd.append(vsd)
        vsd = np.stack(list_vsd, axis=0)
        vsd_acc = (vsd <= 0.3) * 100.0
        logging.info(f"VSD for obj ({obj_group[0]} to {obj_group[-1]}): {vsd_acc.mean():.2f} %")

if __name__ == "__main__":
    test()
