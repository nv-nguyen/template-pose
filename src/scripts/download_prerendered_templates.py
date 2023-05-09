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
    url_templates = cfg.url_preRendered_templates
    zip_path = f"{osp.dirname(cfg.data.lm.root_dir)}/zip/templates.zip"
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    gdown.download(url_templates, zip_path, quiet=False, fuzzy=True)
    # use "pip install -U --no-cache-dir gdown --pre" in case gdown refuses because it is not public link
    
    template_path = f"{osp.dirname(cfg.data.lm.root_dir)}"
    unzip_command = "unzip {} -d {}".format(zip_path, template_path)
    os.system(unzip_command)
    
if __name__ == "__main__":
    download()
