import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack
from attacks.deepfool import DeepfoolLinfAttack
from autoattack import AutoAttack

import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_data_loaders(data_path):
  train_loader, test_loader = get_tiny_imagenet_dataset(data_path=data_path, img_size=32)
  return train_loader, test_loader


def get_tiny_imagenet_dataset(data_path, img_size=32, train_batch_size=128, test_batch_size=128):
    # preprocess data with https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4
    print(data_path)
    train_root = os.path.join(data_path, 'train')
    test_root = os.path.join(data_path, 'val')
    # mean = [x / 255 for x in [127.5, 127.5, 127.5]]
    # std = [x / 255 for x in [127.5, 127.5, 127.5]]
    train_transform = transforms.Compose(
      [
       transforms.Resize((img_size, img_size)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(img_size, padding=4),
       transforms.ToTensor(),
      #  transforms.Normalize(mean, std)
      ])
    test_transform = transforms.Compose(
      [transforms.Resize((img_size, img_size)), transforms.ToTensor(), 
      # transforms.Normalize(mean, std)
      ])
    train_data = datasets.ImageFolder(train_root, transform=train_transform)
    test_data = datasets.ImageFolder(test_root, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader


def get_attack(model, attack_name, dataset):
  if dataset == 'mnist':

    if attack_name == 'linf_pgd':
      return LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)

    elif attack_name == 'kl_linf_pgd':
      return KLPGDAttack(
        model, eps=0.031,
        nb_iter=10, eps_iter=0.003, clip_min=0., clip_max=1.0, distance='l_inf')

    elif attack_name == 'kl_l2_pgd':
      return KLPGDAttack(
        model, eps=0.031,
        nb_iter=10, eps_iter=0.003, clip_min=0., clip_max=1.0, distance='l_2')

    elif attack_name == 'linf_deepfool':
      return DeepfoolLinfAttack(
        model, num_classes=10, nb_iter=40, eps=0.3, clip_min=0.0, clip_max=1.0)

    elif attack_name == 'cw':
      return CarliniWagnerL2Attack(
        model, num_classes=10, max_iterations=20, learning_rate=0.1,
        clip_min=0.0, clip_max=1.0)

    elif attack_name == 'aa_apgdt':
      aa = AA(model, norm='Linf', eps=0.3, n_iter=20, version='standard', verbose=False)
      aa.attacks_to_run =['apgd-t']
      return aa

    elif attack_name == 'aa_apgdce':
      aa = AA(model, norm='Linf', eps=0.3, n_iter=40, version='standard', verbose=False)
      aa.attacks_to_run =['apgd-ce']
      return aa

    else:
      raise NotImplementedError(f'Attack name not recognized ({attack_name})')

  elif 'cifar' in dataset:

    if attack_name == 'linf_pgd':
      return LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8. / 255,
        nb_iter=20, eps_iter=2. / 255, rand_init=True, clip_min=0., clip_max=1.0,
        targeted=False)

    elif attack_name == 'kl_linf_pgd':
      return KLPGDAttack(
          model, eps=0.031,
          nb_iter=10, eps_iter=0.003, clip_min=0., clip_max=1.0, distance='l_inf')

    elif attack_name == 'kl_l2_pgd':
      return KLPGDAttack(
        model, eps=0.031,
        nb_iter=10, eps_iter=0.003, clip_min=0., clip_max=1.0, distance='l_2')

    elif attack_name == 'aa_apgdce':
      aa = AA(model, norm='Linf', eps=8./255, n_iter=20, version='standard', verbose=False)
      aa.attacks_to_run =['apgd-ce']
      return aa

    else:
      raise NotImplementedError(f'Attack name not recognized ({attack_name})')

  elif 'tiny-imagenet-200' in dataset:
    if attack_name == 'linf_pgd':
      return LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=4. / 255,
        nb_iter=5, eps_iter=2. / 255, rand_init=True, clip_min=0., clip_max=1.0,
        targeted=False)
    elif attack_name == 'aa_apgdce':
      aa = AA(model, norm='Linf', eps=4./255, n_iter=5, version='standard', verbose=False)
      aa.attacks_to_run =['apgd-ce']
      return aa

  else:
    raise NotImplementedError(f'Dataset not recognized ({dataset})')


class KLPGDAttack:
  def __init__(self, model, eps_iter=0.007, eps=0.031, nb_iter=5, clip_min=0., clip_max=1.0, distance='l_inf'):
    self.model = model
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.distance = distance

  def perturb(self, x_natural, target=None):
      # define KL-loss
      criterion_kl = nn.KLDivLoss(size_average=False)
      batch_size = len(x_natural)
      # generate adversarial example
      x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
      if self.distance == 'l_inf':
          for _ in range(self.nb_iter):
              x_adv.requires_grad_()
              with torch.enable_grad():
                  loss_kl = criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                        F.softmax(self.model(x_natural), dim=1))
              grad = torch.autograd.grad(loss_kl, [x_adv])[0]
              x_adv = x_adv.detach() + self.eps_iter * torch.sign(grad.detach())
              x_adv = torch.min(torch.max(x_adv, x_natural - self.eps), x_natural + self.eps)
              x_adv = torch.clamp(x_adv, 0.0, 1.0)
      elif self.distance == 'l_2':
          delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
          delta = Variable(delta.data, requires_grad=True)

          # Setup optimizers
          optimizer_delta = optim.SGD([delta], lr=self.eps / self.nb_iter * 2)

          for _ in range(self.nb_iter):
              adv = x_natural + delta

              # optimize
              optimizer_delta.zero_grad()
              with torch.enable_grad():
                  loss = (-1) * criterion_kl(F.log_softmax(self.model(adv), dim=1),
                                            F.softmax(self.model(x_natural), dim=1))
              loss.backward()
              # renorming gradient
              grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
              delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
              # avoid nan or inf if gradient is 0
              if (grad_norms == 0).any():
                  delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
              optimizer_delta.step()

              # projection
              delta.data.add_(x_natural)
              delta.data.clamp_(0, 1).sub_(x_natural)
              delta.data.renorm_(p=2, dim=0, maxnorm=self.eps)
          x_adv = Variable(x_natural + delta, requires_grad=False)
      else:
          x_adv = torch.clamp(x_adv, 0.0, 1.0)
      return x_adv


class AA(AutoAttack):
  def __init__(self, model, attacks_to_run=[], norm='Linf', eps=0.3, n_iter=20, version='standard', verbose=False):
    super(AA, self).__init__(model, attacks_to_run=attacks_to_run,
    norm=norm, eps=eps, n_iter=n_iter, version=version, verbose=verbose)

  def perturb(self, x_orig, y_orig):
    return self.run_standard_evaluation_individual(x_orig, y_orig, bs=x_orig.shape[0])[self.attacks_to_run[0]]

