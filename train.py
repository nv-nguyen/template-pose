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

pl.seed_everything(2022)
# set level logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):
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

    val_dataloaders = {}
    for data_name in cfg.train_datasets:
        if data_name == "hope":
            continue
        config_dataloader = cfg.data[data_name].dataloader
        splits = [
            split
            for split in os.listdir(config_dataloader.root_dir)
            if os.path.isdir(os.path.join(config_dataloader.root_dir, split))
        ]
        splits = [
            split
            for split in splits
            if split.startswith("train") or split.startswith("val")
        ]
        assert len(splits) == 1, f"Found {splits} train splits for {data_name}"
        split = splits[0]
        config_dataloader.reset_metaData = True
        config_dataloader.split = split
        config_dataloader.isTesting = True
        val_dataloader = DataLoader(
            instantiate(config_dataloader),
            batch_size=cfg.machine.batch_size,
            num_workers=cfg.machine.num_workers,
            shuffle=False, # for visualize different samples
        )
        val_dataloaders[data_name] = val_dataloader
        logging.info(
            f"Loading validation dataloader with {data_name}, size {len(val_dataloader)} done!"
        )
    val_dataloaders = concat_dataloader(val_dataloaders)

    train_dataloaders = {}
    for data_name in cfg.train_datasets:
        config_dataloader = cfg.data[data_name].dataloader
        splits = [
            split
            for split in os.listdir(config_dataloader.root_dir)
            if os.path.isdir(os.path.join(config_dataloader.root_dir, split))
        ]
        splits = [
            split
            for split in splits
            if split.startswith("train") or split.startswith("val")
        ]
        assert len(splits) == 1, f"Found {splits} train splits for {data_name}"
        split = splits[0]
        config_dataloader.split = split
        config_dataloader.reset_metaData = False
        config_dataloader.isTesting = False
        config_dataloader.use_augmentation = cfg.use_augmentation
        config_dataloader.use_random_rotation = cfg.use_random_rotation
        config_dataloader.use_random_scale_translation = (
            cfg.use_random_scale_translation
        )
        config_dataloader.use_additional_negative_samples_for_training = (
            cfg.use_additional_negative_samples_for_training
        )
        train_dataloader = DataLoader(
            instantiate(config_dataloader),
            batch_size=cfg.machine.batch_size,
            num_workers=cfg.machine.num_workers,
            shuffle=True,
        )
        logging.info(
            f"Loading train dataloader with {data_name}, size {len(train_dataloader)} done!"
        )
        logging.info("---" * 100)
        train_dataloaders[data_name] = train_dataloader
    train_dataloaders = concat_dataloader(train_dataloaders)

    logging.info(
        f"Fitting the model: train_size={len(train_dataloaders)}, val_size={len(val_dataloaders)}"
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
    )
    logging.info(f"Fitting done")


if __name__ == "__main__":
    train()
