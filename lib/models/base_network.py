import torch.nn as nn
import torch
from lib.models.model_utils import conv1x1
from lib.losses.contrast_loss import InfoNCE
from lib.losses.contrast_loss import cosine_similarity, OcclusionAwareSimilarity


class BaseFeatureExtractor(nn.Module):
    def __init__(self, config_model, threshold):
        super(BaseFeatureExtractor, self).__init__()
        assert config_model.backbone == "base", print("Initializing with BaseNetwork but not using base backbone!!!")
        self.loss = InfoNCE()
        self.occlusion_sim = OcclusionAwareSimilarity(threshold=threshold)
        self.use_global = config_model.use_global
        self.sim_distance = nn.CosineSimilarity(dim=1, eps=1e-6)
        if self.use_global:
            print("Using base network with ", config_model)
            self.layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(8, 8))
            self.layer2 = nn.Conv2d(in_channels=16, out_channels=7, kernel_size=(5, 5))
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(12 * 12 * 7, 256)
            self.fc2 = nn.Linear(256, config_model.descriptor_size)
            self.backbone = nn.Sequential(self.layer1, self.pooling,
                                          self.layer2, self.pooling,
                                          nn.Flatten(), self.fc1,
                                          nn.ReLU(), self.fc2)
        else:
            # replace all the pooling layers, fc layers with conv1x1
            self.layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(8, 8), stride=(2, 2))
            self.layer2 = nn.Conv2d(in_channels=16, out_channels=7, kernel_size=(5, 5))
            self.projector = nn.Sequential(conv1x1(7, 256), nn.ReLU(), conv1x1(256, config_model.descriptor_size))
            self.backbone = nn.Sequential(self.layer1, nn.ReLU(),
                                          self.layer2, nn.ReLU(),
                                          self.projector)

    def forward(self, x):
        feat = self.backbone(x)
        return feat

    def calculate_similarity(self, feat_query, feat_template, mask, training=True):
        """
        Calculate similarity for each batch
        input:
        feat_query: BxCxHxW
        feat_template: BxCxHxW
        output: similarity Bx1
        """
        if self.use_global:
            similarity = self.sim_distance(feat_query, feat_template)
            return similarity
        else:
            B, C, H, W = feat_query.size(0), feat_query.size(1), feat_query.size(2), feat_query.size(3)
            mask_template = mask.repeat(1, C, 1, 1)
            num_non_zero = mask.squeeze(1).sum(axis=2).sum(axis=1)
            if training:  # don't use occlusion similarity during training
                similarity = self.sim_distance(feat_query * mask_template,
                                               feat_template * mask_template).sum(axis=2).sum(axis=1) / num_non_zero
            else:  # apply occlusion aware similarity with predefined threshold
                similarity = self.sim_distance(feat_query * mask_template,
                                               feat_template * mask_template)
                similarity = self.occlusion_sim(similarity).sum(axis=2).sum(axis=1) / num_non_zero
            return similarity

    def calculate_similarity_for_search(self, feat_query, feat_templates, mask, training=True):
        """
        calculate pairwise similarity:
        input:
        feat_query: BxCxHxW
        feat_template: NxCxHxW
        output: similarity BxN
        """
        B, N, C = feat_query.size(0), feat_templates.size(0), feat_query.size(1)
        if self.use_global:
            similarity = cosine_similarity(feat_query, feat_templates)
            return similarity
        else:
            similarity = torch.zeros((B, N)).type_as(feat_query)
            for i in range(B):
                query4d = feat_query[i].unsqueeze(0).repeat(N, 1, 1, 1)
                mask_template = mask.repeat(1, C, 1, 1)
                num_feature = mask.squeeze(1).sum(axis=2).sum(axis=1)
                sim = self.sim_distance(feat_templates * mask_template,
                                        query4d * mask_template)
                if training:
                    similarity[i] = sim.sum(axis=2).sum(axis=1) / num_feature
                else:
                    sim = self.occlusion_sim(sim)
                    similarity[i] = sim.sum(axis=2).sum(axis=1) / num_feature
            return similarity

    def calculate_global_loss(self, positive_pair, negative_pair, neg_pair_regression=None, delta=None):
        loss = self.loss(pos_sim=positive_pair, neg_sim=negative_pair)
        if delta is not None:
            mse = nn.MSELoss()
            delta_loss = mse(neg_pair_regression, delta)
            loss[2] += delta_loss
            return loss[0], loss[1], loss[2], delta_loss
        else:
            return loss[0], loss[1], loss[2]
