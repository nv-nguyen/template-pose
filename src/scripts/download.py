import logging
import os, sys
import os.path as osp

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf


def run_download(config: DictConfig) -> None:
    tmp_path = f"{osp.dirname(config.root_dir)}/zip/{osp.basename(config.root_dir)}.zip"
    cad_tmp_path = (
        f"{osp.dirname(config.root_dir)}/zip/{osp.basename(config.root_dir)}_cad.zip"
    )
    os.makedirs(f"{osp.dirname(config.root_dir)}/zip", exist_ok=True)
    os.makedirs(config.root_dir, exist_ok=True)

    # define command to download RGB
    command = f"wget -O {tmp_path} {config.source.url}"
    if config.source.http:
        command += " --no-check-certificate"
    logging.info(f"Running {command}")
    os.system(command)

    # unzip
    if config.source.unzip_mode == "tar":
        unzip_cmd = "tar xf {} -C {}".format(tmp_path, config.root_dir)
    else:
        unzip_cmd = "unzip {} -d {}".format(tmp_path, config.root_dir)
    logging.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)

    # define command to download CAD models
    command = f"wget -O {cad_tmp_path} {config.source.cad_url}  --no-check-certificate"
    logging.info(f"Running {command}")
    os.system(command)

    # unzip
    if osp.exists(f"{config.root_dir}/models"):
        os.system(f"rm -r {config.root_dir}/models")
    unzip_cmd = "unzip {} -d {}/models".format(cad_tmp_path, config.root_dir)
    logging.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)

    if config.source.processing != "": # only happen with occlusion LINEMOD
        if config.source.processing == "rename":
            current_path = os.path.join(config.root_dir, "OcclusionChallengeICCV2015/*")
            new_path = config.root_dir
            rename_command = f"mv {current_path} {new_path}"
            os.system(rename_command)
            remove_command = f"rm -rf {config.root_dir}/OcclusionChallengeICCV2015"
            os.system(remove_command)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="render",
)
def download(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    for data_cfg in cfg.data.values():
        logging.info(f"Downloading {data_cfg.dataset_name}")
        run_download(data_cfg)
        logging.info(f"---" * 100)


if __name__ == "__main__":
    download()
