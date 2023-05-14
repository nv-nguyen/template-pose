import logging
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch.nn as nn
from src.utils.weight import load_checkpoint
from src.dataloader.lm_utils import get_list_id_obj_from_split_name
import pytorch_lightning as pl
from src.utils.dataloader import concat_dataloader
from torch.utils.data import ConcatDataset
from src.dataloader.lm_utils import query_real_ids, query_symmetry

pl.seed_everything(2022)
# set level logging
logging.basicConfig(level=logging.INFO)


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
    for obj_id in query_real_ids:
        test_datasets = []
        for mode in ["query", "template"]:
            config_dataloader = cfg.data.lm.dataloader
            config_dataloader.reset_metaData = False
            config_dataloader.split = "test"
            config_dataloader.obj_id = int(obj_id + 1)
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
        model.obj_symmetry = query_symmetry[obj_id]
        model.metric_eval = "geodesic"
        trainer.test(
            model,
            dataloaders=test_dataloaders,
            ckpt_path=cfg.model.checkpoint_path,
        )
        logging.info(f"---" * 20)


if __name__ == "__main__":
    test()
