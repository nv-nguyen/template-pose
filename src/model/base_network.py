import torch.nn as nn
import torch
import pytorch_lightning as pl
import logging
import torch.nn.functional as F
from src.utils.visualization_utils import put_image_to_grid
from torchvision.utils import make_grid, save_image
import os
import wandb
import torchvision.transforms as transforms


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


class InfoNCE(nn.Module):
    def __init__(self, tau=0.1, extra_contrast_type=None):
        super(InfoNCE, self).__init__()
        self.tau = tau
        self.extra_contrast_type = extra_contrast_type

    def forward(self, pos_sim, neg_sim, sim_extra_obj=None):
        """
        neg_sim: BxB
        pos_sim: Bx1
        sim_extra: BxB use extra object as negative
        """
        b = neg_sim.shape[0]
        logits = (1 - torch.eye(b)).type_as(neg_sim) * neg_sim + torch.eye(b).type_as(
            pos_sim
        ) * pos_sim

        labels = torch.arange(b, dtype=torch.long, device=logits.device)
        if sim_extra_obj is not None:
            sim_extra_obj = sim_extra_obj[:b]
            if self.extra_contrast_type == "BOP_ShapeNet":
                # Add more negative samples by taking pairs (BOP, ShapeNet)
                logits = torch.cat((logits, sim_extra_obj), dim=1)
            elif self.extra_contrast_type == "ShapeNet_ShapeNet":
                # Add more negative samples by taking pairs (ShapeNet, ShapeNet), duplicate the positive samples from BOP to get Identity matrix
                extra_logits = (1 - torch.eye(b)).type_as(
                    sim_extra_obj
                ) * sim_extra_obj + torch.eye(b).type_as(pos_sim) * pos_sim
                logits = torch.cat((logits, extra_logits), dim=0)  # 2BxB
                extra_labels = torch.arange(
                    b, dtype=torch.long, device=logits.device
                ).cuda()
                labels = torch.cat(
                    (labels, extra_labels), dim=0
                )  # 2B as [Identity, Identity]
        logits = logits / self.tau
        loss = F.cross_entropy(logits, labels)
        return [torch.mean(pos_sim), torch.mean(neg_sim), loss]


class OcclusionAwareSimilarity(nn.Module):
    def __init__(self, threshold):
        super(OcclusionAwareSimilarity, self).__init__()
        self.threshold = threshold

    def forward(self, similarity_matrix):
        indicator_zero = similarity_matrix <= self.threshold
        similarity_matrix[indicator_zero] = 0
        return similarity_matrix


class BaseFeatureExtractor(pl.LightningModule):
    def __init__(self, descriptor_size, threshold, **kwargs):

        # define the network
        super(BaseFeatureExtractor, self).__init__()
        self.loss = InfoNCE()
        self.occlusion_sim = OcclusionAwareSimilarity(threshold=threshold)
        self.sim_distance = nn.CosineSimilarity(dim=1)  # eps=1e-2

        # remove all the pooling layers, fc layers with conv1x1
        layer1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(8, 8), stride=(2, 2)
        )
        layer2 = nn.Conv2d(in_channels=16, out_channels=7, kernel_size=(5, 5))
        projector = nn.Sequential(
            conv1x1(7, 256), nn.ReLU(), conv1x1(256, descriptor_size)
        )
        self.backbone = nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), projector)

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

        from src.utils.weight import KaiMingInitPL

        KaiMingInitPL(self.backbone, exclude_name=None)

    def warm_up_lr(self):
        for optim in self.trainer.optimizers:
            for pg in optim.param_groups:
                pg["lr"] = self.global_step / float(self.warm_up_steps) * self.lr
            if self.global_step % 50 == 0:
                logging.info(f"Step={self.global_step}, lr warm up: lr={pg['lr']}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            self.lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 50], gamma=0.5
        )
        return [optimizer], [lr_scheduler]

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
        B, C, H, W = (
            feat_query.size(0),
            feat_query.size(1),
            feat_query.size(2),
            feat_query.size(3),
        )
        mask_template = mask.repeat(1, C, 1, 1)
        num_non_zero = mask.squeeze(1).sum(axis=2).sum(axis=1)
        if training:  # don't use occlusion similarity during training
            similarity = (
                self.sim_distance(
                    feat_query * mask_template, feat_template * mask_template
                )
                .sum(axis=2)
                .sum(axis=1)
                / num_non_zero
            )
        else:  # apply occlusion aware similarity with predefined threshold
            similarity = self.sim_distance(
                feat_query * mask_template, feat_template * mask_template
            )
            similarity = (
                self.occlusion_sim(similarity).sum(axis=2).sum(axis=1) / num_non_zero
            )
        return similarity

    def calculate_similarity_for_search(
        self, feat_query, feat_templates, mask, training=True
    ):
        """
        calculate pairwise similarity:
        input:
        feat_query: BxCxHxW
        feat_template: NxCxHxW
        output: similarity BxN
        """
        B, N, C = feat_query.size(0), feat_templates.size(0), feat_query.size(1)
        similarity = torch.zeros((B, N)).type_as(feat_query)
        assert torch.isnan(similarity).any() == False
        for i in range(B):
            query4d = feat_query[i].unsqueeze(0).repeat(N, 1, 1, 1)
            mask_template = mask.repeat(1, C, 1, 1)
            num_feature = mask.squeeze(1).sum(axis=2).sum(axis=1)
            sim = self.sim_distance(
                feat_templates * mask_template, query4d * mask_template
            )
            if training:
                similarity[i] = sim.sum(axis=2).sum(axis=1) / num_feature
            else:
                sim = self.occlusion_sim(sim)
                similarity[i] = sim.sum(axis=2).sum(axis=1) / num_feature
        return similarity

    def calculate_global_loss(
        self, positive_pair, negative_pair, extra_negative_pair=None
    ):
        loss = self.loss(
            pos_sim=positive_pair,
            neg_sim=negative_pair,
            sim_extra_obj=extra_negative_pair,
        )
        return loss[0], loss[1], loss[2]

    def visualize_batch(self, batch, split):
        query = batch["query"]
        template = batch["template"]
        mask_vis = batch["template_mask"].float()
        mask_vis = mask_vis.repeat(1, 3, 1, 1)
        mask_vis = F.interpolate(mask_vis, size=query.shape[2:], mode="nearest")
        vis_img, _ = put_image_to_grid(
            [
                self.transform_inverse(query),
                self.transform_inverse(template),
                mask_vis,
            ],
            adding_margin=False,
        )
        save_path = os.path.join(
            self.log_dir, f"step{self.global_step}_rank{self.global_rank}.png"
        )
        save_image(
            vis_img.float(),
            save_path,
            nrow=int(query.shape[0] ** 0.5) * 3,
        )
        logging.info(f"save image to {save_path}")
        self.logger.experiment.log({f"sample/{split}": wandb.Image(save_path)})

    def training_step(self, batch, idx):
        if self.trainer.global_step < self.warm_up_steps:
            self.warm_up_lr()
        elif self.trainer.global_step == self.warm_up_steps:
            logging.info(f"Finished warm up, setting lr to {self.lr}")

        feature_query, feature_template, mask = [], [], []
        for dataset_name in batch:
            query_i = batch[dataset_name]["query"]
            template_i = batch[dataset_name]["template"]
            mask_i = batch[dataset_name]["template_mask"]

            feature_query_i = self.forward(query_i)
            feature_template_i = self.forward(template_i)

            feature_query.append(feature_query_i)
            feature_template.append(feature_template_i)
            mask.append(mask_i)

            if self.global_step % self.log_interval == 0 and self.global_rank == 0:
                self.visualize_batch(batch[dataset_name], f"train_{dataset_name}")
        # collect from all datasets
        feature_query = torch.cat(feature_query, dim=0)
        feature_template = torch.cat(feature_template, dim=0)
        mask = torch.cat(mask, dim=0)

        # collect data from all devices
        if self.use_all_gather:
            feature_query = self.all_gather(feature_query, sync_grads=True)
            feature_template = self.all_gather(feature_template, sync_grads=True)
            mask = self.all_gather(mask, sync_grads=True)

            # reshape data from (num_devices, B, C, H, W) to (num_devices*B, C, H, W)
            feature_query = feature_query.reshape(-1, *feature_query.shape[2:])
            feature_template = feature_template.reshape(-1, *feature_template.shape[2:])
            mask = mask.reshape(-1, *mask.shape[2:])

        positive_similarity = self.calculate_similarity(
            feature_query, feature_template, mask
        )
        negative_similarity = self.calculate_similarity_for_search(
            feature_query, feature_template, mask
        )  # B x B
        avg_pos_sim, avg_neg_sim, loss = self.calculate_global_loss(
            positive_pair=positive_similarity, negative_pair=negative_similarity
        )
        self.log(
            "loss_InfoNCE",
            loss,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "similarity/positive_pairs",
            avg_pos_sim,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            "similarity/negative_pairs",
            avg_neg_sim,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return loss

    def validation_step(
        self,
    ):
        return

    def monitoring_score(self, dict_scores, split_name):
        for key, value in dict_scores.items():
            self.log(
                f"{key}/{split_name}",
                value,
                sync_dist=True,
                add_dataloader_idx=False,
            )
