import logging
import os, sys
import os.path as osp

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import hydra
import gdown
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="render",
)
def download(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    url_moco = cfg.url_template_pose_checkpoint
    save_dir = f"{osp.dirname(osp.dirname(cfg.data.lm.root_dir))}/pretrained"
    os.makedirs(save_dir, exist_ok=True)

    moco_path = os.path.join(save_dir, "template_pose_checkpoint.ckpt")
    gdown.download(url_moco, moco_path, quiet=False, fuzzy=True)
    # use "pip install -U --no-cache-dir gdown --pre" in case gdown refuses because it is not public link

if __name__ == "__main__":
    download()
