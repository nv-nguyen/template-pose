# https://github.com/monniert/dti-clustering/blob/b57a77d4c248b16b4b15d6509b6ec493c53257ef/src/optimizer/__init__.py
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
# https://github.com/monniert/dti-clustering/blob/b57a77d4c248b16b4b15d6509b6ec493c53257ef/src/optimizer/__init__.py
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, _LRScheduler
import math
import torch


def get_optimizer(name):
    if name is None:
        name = 'sgd'
    return {
        "sgd": SGD,
        "adam": Adam,
        "asgd": ASGD,
        "adamax": Adamax,
        "adadelta": Adadelta,
        "adagrad": Adagrad,
        "rmsprop": RMSprop,
    }[name]


class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def get_scheduler(name):
    if name is None:
        name = 'constant_lr'
    return {
        "constant_lr": ConstantLR,
        "poly_lr": PolynomialLR,
        "multi_step": MultiStepLR,
        "cosine_annealing": CosineAnnealingLR,
        "exp_lr": ExponentialLR,
    }[name]


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.optimizer.__class__.__name__)


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]

    def __str__(self):
        params = [
            'optimizer: {}'.format(self.optimizer.__class__.__name__),
            'decay_iter: {}'.format(self.decay_iter),
            'max_iter: {}'.format(self.max_iter),
            'gamma: {}'.format(self.gamma),
        ]
        return '{}({})'.format(self.__class__.__name__, ','.join(params))


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


def adjust_learning_rate(optimizer, base_lr, gamma):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = base_lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr