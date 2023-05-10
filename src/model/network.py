import torch.nn as nn
import torch
from src.model.base_network import (
    BaseFeatureExtractor,
    conv1x1,
    InfoNCE,
    OcclusionAwareSimilarity,
)
from src.model.resnet import resnet50

import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms, utils
import logging

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
import matplotlib.pyplot as plt
from PIL import Image
import wandb
from src.model.loss import GeodesicError


class FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, descriptor_size, threshold, **kwargs):
        super(BaseFeatureExtractor, self).__init__()
        self.descriptor_size = descriptor_size

        self.backbone = resnet50(
            use_avg_pooling_and_fc=False, num_classes=1
        )  # num_classes is useless

        self.projector = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(2048, 256),
            nn.ReLU(inplace=False),
            conv1x1(256, descriptor_size),
        )
        self.metric = GeodesicError()
        self.loss = InfoNCE()
        self.occlusion_sim = OcclusionAwareSimilarity(threshold=threshold)
        self.sim_distance = nn.CosineSimilarity(dim=1)  # eps=1e-2

        # define optimizer
        self.weight_decay = float(kwargs["weight_decay"])
        self.lr = float(kwargs["lr"])
        self.use_all_gather = kwargs["use_all_gather"]  # multi-gpu contrast learning
        self.warm_up_steps = kwargs["warm_up_steps"]

        self.log_interval = kwargs["log_interval"]
        self.log_dir = kwargs["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        self.transform_inverse = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.projector(feat)
        # feat = F.normalize(feat, dim=1)
        return feat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 50, 80, 100], gamma=0.2
        )
        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    from src.utils.weight import load_checkpoint
