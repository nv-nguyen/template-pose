import os
from lib.utils.config import Config
import argparse
import torch
import gdown


def download_and_process(source_url, local_path, save_path):
    # credit: https://github.com/YoungXIAO13/PoseContrast/blob/main/pretrain_models/convert_pretrain.py
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    command = "wget -O {} '{}'".format(local_path, source_url)
    os.system(command)
    print("Download pretrained weight of MOCO done!")
    pretrained_weights = torch.load(local_path, map_location="cpu")
    pretrained_weights = pretrained_weights["state_dict"]
    newmodel = {}
    for k, v in pretrained_weights.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")
        print(old_k, "->", k)
        newmodel[k] = v
    state = {"model": newmodel, "__author__": "MOCO"}
    torch.save(state, save_path)
    print("Processing weight of MOCO done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('template-pose download pretrained-weight scripts')
    parser.add_argument('--config', type=str, default="./config.json")
    parser.add_argument('--model_name', type=str, help="Name of model to download")
    args = parser.parse_args()

    config = Config(config_file=args.config).get_config()
    if args.model_name == "MoCov2":
        download_local_path = os.path.join(config.root_path, config.pretrained_moco.download_local_path)
        save_local_path = os.path.join(config.root_path, config.pretrained_moco.local_path)
        download_and_process(source_url=config.pretrained_moco.source_url,
                             local_path=download_local_path,
                             save_path=save_local_path)
    else:
        assert args.model_name in config.pretrained_weights.keys(), print("Name of model is not correct!")
        # for model_name in config.pretrained_weights.keys():
        local_path = os.path.join(config.root_path, config.pretrained_weights[args.model_name].local_path)
        if not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))
        gdown.download(config.pretrained_weights[args.model_name].source_url, local_path, quiet=False, fuzzy=True)
