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
from src.model.loss import GeodesicError
import multiprocessing
from src.poses.vsd import vsd_obj
from functools import partial
import time
from tqdm import tqdm
import numpy as np


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
        self.metric = GeodesicError()
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
            batch_size = batch[dataset_name]["query"].size(0)
            if "neg_query" in batch[dataset_name]:
                # shuffle when using additional negative samples
                index = torch.randperm(
                    batch_size * 2, device=batch[dataset_name]["query"].device
                )
                # concat
                batch[dataset_name]["query"] = torch.cat(
                    (batch[dataset_name]["query"], batch[dataset_name]["neg_query"])
                )[index]
                batch[dataset_name]["template"] = torch.cat(
                    (
                        batch[dataset_name]["template"],
                        batch[dataset_name]["neg_template"],
                    )
                )[index]
                batch[dataset_name]["template_mask"] = torch.cat(
                    (
                        batch[dataset_name]["template_mask"],
                        batch[dataset_name]["neg_template_mask"],
                    )
                )[index]

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

    def validation_step(self, batch, idx):
        for dataset_name in batch:
            self.eval_batch(batch[dataset_name], dataset_name)

    def eval_batch(self, batch, dataset_name, k=5):
        query = batch["query"]  # B x C x W x H
        templates = batch["templates"]  # B x N x C x W x H
        template_masks = batch["template_masks"]
        template_poses = batch["template_poses"]
        feature_query = self.forward(query)

        # get predictions
        batch_size = query.shape[0]
        pred_indexes = torch.zeros(batch_size, k, device=self.device).long()
        for idx in range(batch_size):
            feature_template = self.forward(templates[idx, :])
            mask = template_masks[idx, :]
            matrix_sim = self.calculate_similarity_for_search(
                feature_query[idx].unsqueeze(0), feature_template, mask, training=False
            )
            weight_sim, pred_index = matrix_sim.topk(k=k)
            pred_indexes[idx] = pred_index.reshape(-1)

        retrieved_template = templates[
            torch.arange(0, batch_size, device=query.device), pred_indexes[:, 0]
        ]
        retrieved_poses = template_poses[
            torch.arange(0, batch_size, device=query.device).unsqueeze(1).repeat(1, k),
            pred_indexes,
        ]
        # visualize prediction
        save_image_path = os.path.join(
            self.log_dir,
            f"retrieved_val_step{self.global_step}_rank{self.global_rank}.png",
        )
        vis_imgs = [
            self.transform_inverse(query),
            self.transform_inverse(retrieved_template),
        ]
        vis_imgs, ncol = put_image_to_grid(vis_imgs)
        vis_imgs_resized = vis_imgs.clone()
        vis_imgs_resized = F.interpolate(
            vis_imgs_resized, (64, 64), mode="bilinear", align_corners=False
        )
        save_image(
            vis_imgs_resized,
            save_image_path,
            nrow=ncol * 4,
        )
        self.logger.experiment.log(
            {f"retrieval/{dataset_name}": wandb.Image(save_image_path)},
        )

        # calculate the scores
        error, acc = self.metric(
            predR=retrieved_poses,
            gtR=batch["query_pose"],
            symmetry=torch.zeros(batch_size, device=self.device).long(),
        )
        self.monitoring_score(dict_scores=acc, split_name=f"{dataset_name}")

    def monitoring_score(self, dict_scores, split_name):
        for key, value in dict_scores.items():
            self.log(
                f"{key}/{split_name}",
                value,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def test_step(self, batch, idx):
        if "template" in batch:  # loading all templates
            template = batch["template"]  # B x C x W x H
            template_mask = batch["template_mask"]
            template_pose = batch["template_pose"]
            feature_template = self.forward(template)
            return {
                "template": F.interpolate(
                    template, (64, 64), mode="bilinear", align_corners=False
                ),
                "feature_template": feature_template,
                "template_pose": template_pose,
                "template_mask": template_mask,
            }
        else:  # loading all templates
            query = batch["query"]
            query_pose = batch["query_pose"]
            feature_query = self.forward(query)
            samples = {
                "query": F.interpolate(
                    query, (64, 64), mode="bilinear", align_corners=False
                ),
                "feature_query": feature_query,
                "query_pose": query_pose,
            }
            # loading additional metric for VSD metric
            if "intrinsic" in batch:
                samples["intrinsic"] = batch["intrinsic"]
                samples["depth_path"] = batch["depth_path"]
                samples["query_translation"] = batch["query_translation"]
            return samples

    def get_vsd(
        self,
        predR,
        gtR,
        query_translation,
        intrinsic,
        depth_path,
        save_path,
        pred_bbox=None,
        gt_bbox=None,
    ):
        pool = multiprocessing.Pool(processes=self.trainer.num_workers)

        # to calculate vsd, it requires: GT, predicted pose, intrinsic, depth map and CAD model
        data = {}
        data["intrinsic"] = intrinsic.cpu().numpy()
        data["depth_path"] = depth_path

        pred_poses = torch.zeros((predR.shape[0], 1, 4, 4), device=predR.device)
        pred_poses[:, :, 3, 3] = 1
        pred_poses[:, 0, :3, :3] = predR
        # it requries pred_bbox to calculate predicted translation, otherwise using GT translation
        if pred_bbox is None:
            pred_poses[:, 0, :3, 3] = query_translation
        else:
            raise NotImplementedError
        data["pred_poses"] = pred_poses.cpu().numpy()

        gt_poses = torch.cat((gtR, query_translation.unsqueeze(2)), dim=2)
        gt_poses = torch.cat(
            (gt_poses, torch.zeros((predR.shape[0], 1, 4), device=predR.device)), dim=1
        )
        gt_poses[:, 3, 3] = 1.0
        data["query_pose"] = gt_poses.cpu().numpy()

        vsd_obj_from_index = partial(
            vsd_obj, list_frame_data=data, mesh=self.testing_cad
        )
        start_time = time.time()
        vsd_error = list(
            tqdm(
                pool.imap_unordered(vsd_obj_from_index, range(len(data["query_pose"]))),
                total=len(data["query_pose"]),
            )
        )
        vsd_error = np.stack(vsd_error, axis=0)  # Bxk where k is top k retrieved
        np.save(save_path, vsd_error)
        finish_time = time.time()
        logging.info(
            f"Total time to render at rank {self.global_rank}: {finish_time - start_time}",
        )
        final_scores = {}
        for k in [1]:
            best_vsd = np.min(vsd_error[:, :k], 1)
            final_scores[f"top {k}, vsd_median"] = np.median(best_vsd)
            for threshold in [0.15, 0.3, 0.45]:
                vsd_acc = (best_vsd <= threshold) * 100.0
                # same for median
                final_scores[f"top {k}, vsd_scores {threshold}"] = np.mean(vsd_acc)
        return final_scores

    def test_epoch_end(self, test_step_outputs):
        data = {}
        # collect template from all devices
        for idx_batch in range(len(test_step_outputs)):
            if "feature_template" in test_step_outputs[idx_batch]:
                for name in [
                    "template",
                    "feature_template",
                    "template_pose",
                    "template_mask",
                ]:
                    if name not in data:
                        data[name] = []
                    data[name].append(test_step_outputs[idx_batch][name])
        # concat template
        for name in ["template", "feature_template", "template_pose", "template_mask"]:
            data[name] = torch.cat(data[name], dim=0)

        # find nearest neighbors
        for idx_batch in range(len(test_step_outputs)):
            if "feature_query" in test_step_outputs[idx_batch]:
                query = test_step_outputs[idx_batch]["query"]
                feature_query = test_step_outputs[idx_batch]["feature_query"]
                query_pose = test_step_outputs[idx_batch]["query_pose"]
                # get best template
                matrix_sim = self.calculate_similarity_for_search(
                    feature_query,
                    data["feature_template"],
                    data["template_mask"],
                    training=False,
                )
                weight_sim, pred_index = matrix_sim.topk(k=1)
                pred_pose = data["template_pose"][pred_index.reshape(-1)]
                pred_template = data["template"][pred_index.reshape(-1)]

                # calculate the scores depending on metric used
                if self.metric_eval == "geodesic":
                    error, acc = self.metric(
                        predR=pred_pose,
                        gtR=query_pose,
                        symmetry=torch.ones(
                            query_pose.shape[0], device=self.device
                        ).long()
                        * self.obj_symmetry,
                    )
                    np.save(os.path.join(
                            self.log_dir, f"geodesic_obj_{self.obj_id}_batch{idx_batch}.npy"
                        ), error.cpu().numpy())
                elif self.metric_eval == "vsd":
                    acc = self.get_vsd(
                        predR=pred_pose,
                        gtR=query_pose,
                        query_translation=test_step_outputs[idx_batch][
                            "query_translation"
                        ],
                        intrinsic=test_step_outputs[idx_batch]["intrinsic"],
                        depth_path=test_step_outputs[idx_batch]["depth_path"],
                        save_path=os.path.join(
                            self.log_dir, f"vsd_obj_{self.obj_id}_batch{idx_batch}.npy"
                        ),
                    )
                self.monitoring_score(
                    dict_scores=acc,
                    split_name=f"{self.dataset_name}_obj_{self.obj_id}",
                )
                # visualize prediction
                vis_imgs = [
                    self.transform_inverse(query),
                    self.transform_inverse(pred_template),
                ]
                vis_imgs, ncol = put_image_to_grid(vis_imgs)
                save_image_path = os.path.join(
                    self.log_dir,
                    f"retrieved_test_step{self.global_step}_rank{self.global_rank}.png",
                )
                save_image(
                    vis_imgs,
                    save_image_path,
                    nrow=ncol * 4,
                )
                self.logger.experiment.log(
                    {
                        f"retrieval/{self.dataset_name}_obj_{self.obj_id}": wandb.Image(
                            save_image_path
                        )
                    },
                )
