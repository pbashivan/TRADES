from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models.net_mnist import *
from models.small_cnn import *
from trades import trades_loss

# import sys
# # Change these paths to point to where resnet50 model is
# sys.path.insert(0, '/braintree/home/bashivan/dropbox/Codes/afd/models')
# sys.path.insert(0, '/braintree/home/bashivan/dropbox/Codes/afd/')
from afd_models.mnist_models import LeNetFeats, LeNetDecoder
from afd_models.resnet_mnist import ResNet18Feats, ResNet18FeatsNorm, ResNet18FeatsNormLeaky

parser = argparse.ArgumentParser(description='PyTorch MNIST TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.3,
                    help='perturbation')
parser.add_argument('--num-steps', default=40,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.01,
                    help='perturb step size')
parser.add_argument('--beta', default=1.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', default='./chkpts', type=str, help='path to where to save checkpoints')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--attack_name', default='kl_linf_pgd', type=str)
parser.add_argument('--enc_model', default='resnet18norm', type=str)
parser.add_argument('--exp_name', default='TRADES', type=str, help='experiment name')
args = parser.parse_args()


# settings
if args.enc_model == 'resnet18':
  enc_model = ResNet18Feats()
elif args.enc_model == 'resnet18norm':
  enc_model = ResNet18FeatsNorm()
elif args.enc_model == 'resnet18normleaky':
  enc_model = ResNet18FeatsNormLeaky()
else:
  raise ValueError()

ROOT_PATH = args.save_path
TRAINED_MODEL_PATH = os.path.join(ROOT_PATH, f'trained_models/mnist', args.exp_name)
DATA_PATH = os.path.join(ROOT_PATH, 'data', 'mnist')

postfix = 1
safe_path = TRAINED_MODEL_PATH
while os.path.exists(safe_path):
  safe_path = TRAINED_MODEL_PATH + f'_{postfix}'
  postfix += 1
TRAINED_MODEL_PATH = safe_path
os.makedirs(TRAINED_MODEL_PATH)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_PATH, train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_PATH, train=False,
                   transform=transforms.ToTensor()),
                   batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           target=target,
                           optimizer=optimizer,
                           beta=args.beta,
                           dataset='mnist',
                           attack_name=args.attack_name)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 55:
        lr = args.lr * 0.1
    if epoch >= 75:
        lr = args.lr * 0.01
    if epoch >= 90:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, Net() can be also used here for training
    E, Dc = enc_model, LeNetDecoder(10, num_features=64)
    model = nn.Sequential(E, Dc)
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(TRAINED_MODEL_PATH, 'model-nn-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(TRAINED_MODEL_PATH, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
