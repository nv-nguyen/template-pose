import torch.nn.functional as F
import torch.nn as nn
import torch


def cosine_similarity(a, b, normalize=True):
    if normalize:
        w1 = a.norm(p=2, dim=1, keepdim=True)
        w2 = b.norm(p=2, dim=1, keepdim=True)
        sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
    else:
        sim_matrix = torch.mm(a, b.t())
    return sim_matrix


class InfoNCE(nn.Module):
    def __init__(self, tau=0.1):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def forward(self, pos_sim, neg_sim, sim_extra_obj=None):
        """
        neg_sim: BxB
        pos_sim: Bx1
        sim_extra: BxB use extra object as negative
        """
        b = neg_sim.shape[0]
        logits = (1 - torch.eye(b)).type_as(neg_sim) * neg_sim + torch.eye(b).type_as(pos_sim) * pos_sim
        if sim_extra_obj is not None:
            logits = torch.cat((logits, sim_extra_obj), dim=1)
        logits = logits / self.tau
        labels = torch.arange(b, dtype=torch.long).cuda()
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


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = nn.MSELoss(reduction="none")

    def forward(self, neg_sim, pos_sim):
        loss = (1 - (pos_sim / (neg_sim + self.margin))).clamp(min=0)
        return [torch.mean(pos_sim), torch.mean(neg_sim), torch.mean(loss)]


class TripletLossDistance(nn.Module):
    def __init__(self, margin=0.01):
        super(TripletLossDistance, self).__init__()
        self.margin = margin

    def forward(self, positive_distance, negative_distance):
        loss = (1 - (negative_distance / (positive_distance + self.margin))).clamp(min=0)
        return [torch.mean(positive_distance), torch.mean(negative_distance), torch.mean(loss)]