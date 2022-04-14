import torch.nn as nn
import torch
from lib.models.model_utils import conv1x1
from lib.losses.contrast_loss import InfoNCE, OcclusionAwareSimilarity
from lib.models.base_network import BaseFeatureExtractor
from lib.models.resnet import resnet50


class FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config_model, threshold):
        super(BaseFeatureExtractor, self).__init__()
        assert config_model.backbone == "resnet50", print("Backbone should be ResNet50!")

        self.loss = InfoNCE()
        self.occlusion_sim = OcclusionAwareSimilarity(threshold=threshold)
        self.use_global = config_model.use_global
        self.sim_distance = nn.CosineSimilarity(dim=1, eps=1e-6)
        if self.use_global:
            self.backbone = resnet50(use_avg_pooling_and_fc=True, num_classes=config_model.descriptor_size)
        else:
            self.backbone = resnet50(use_avg_pooling_and_fc=False, num_classes=1)  # num_classes is useless
            self.projector = nn.Sequential(nn.ReLU(inplace=False),
                                           conv1x1(2048, 256),
                                           nn.ReLU(inplace=False),
                                           conv1x1(256, config_model.descriptor_size))

    def forward(self, x):
        feat = self.backbone(x)
        if self.use_global:
            return feat
        else:
            feat = self.projector(feat)
            return feat