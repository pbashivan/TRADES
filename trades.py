import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import get_attack


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                target,
                optimizer,
                beta=1.0,
                dataset='',
                attack_name='kl_linf_pgd'):

    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    adversary = get_attack(model, attack_name, dataset)
    x_adv = adversary.perturb(x_natural, target)
    batch_size = len(x_natural)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss
