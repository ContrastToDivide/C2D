from __future__ import print_function

import argparse
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import models

from dataloaders import dataloader_cifar as dataloader
from models import bit_models
from models.PreResNet import *
from models.resnet import SupCEResNet
from train_cifar import run_train_loop


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--noise_mode', default='sym')
    parser.add_argument('--alpha', default=4., type=float, help='parameter for Beta')
    parser.add_argument('--alpha-loss', default=0.5, type=float, help='parameter for Beta in loss')
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=360, type=int)
    parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
    parser.add_argument('--id', default='')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
    parser.add_argument('--net', default='resnet18', type=str, help='net')
    parser.add_argument('--method', default='reg', type=str, help='method')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--experiment-name', required=True, type=str)
    parser.add_argument('--aug', dest='aug', action='store_true', help='use stronger aug')
    parser.add_argument('--use-std', dest='use_std', action='store_true', help='use stronger aug')
    parser.add_argument('--drop', dest='drop', action='store_true', help='use drop')
    parser.add_argument('--not-rampup', dest='not_rampup', action='store_true', help='not rumpup')
    parser.add_argument('--supcon', dest='supcon', action='store_true', help='use supcon')
    parser.add_argument('--use-aa', dest='use_aa', action='store_true', help='use supcon')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
        torch.cuda.manual_seed_all(args.seed)
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


def linear_rampup(current, warm_up, lambda_u, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, lambda_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up, lambda_u)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model_reg(net='resnet18', dataset='cifar100', num_classes=100, device='cuda:0', drop=0):
    if net == 'resnet18':
        model = ResNet18(num_classes=num_classes, drop=drop)
        model = model.to(device)
        return model
    else:
        model = SupCEResNet(net, num_classes=num_classes)
        model = model.to(device)
        return model


def create_model_selfsup(net='resnet18', dataset='cifar100', num_classes=100, device='cuda:0', drop=0):
    chekpoint = torch.load('pretrained/ckpt_{}_{}.pth'.format(dataset, net))
    sd = {}
    for ke in chekpoint['model']:
        nk = ke.replace('module.', '')
        sd[nk] = chekpoint['model'][ke]
    model = SupCEResNet(net, num_classes=num_classes)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    return model


def create_model_bit(net='resnet18', dataset='cifar100', num_classes=100, device='cuda:0', drop=0):
    if net == 'resnet50':
        model = bit_models.KNOWN_MODELS['BiT-S-R50x1'](head_size=num_classes, zero_head=True)
        model.load_from(np.load("pretrained/BiT-S-R50x1.npz"))
        model = model.to(device)
    elif net == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512 * 1, num_classes)
        model = model.to(device)
    else:
        raise ValueError()
    return model


def main():
    args = parse_args()
    os.makedirs('./checkpoint', exist_ok=True)
    log_name = './checkpoint/%s_%s_%.2f_%.1f_%s' % (
        args.experiment_name, args.dataset, args.r, args.lambda_u, args.noise_mode)
    stats_log = open(log_name + '_stats.txt', 'w')
    test_log = open(log_name + '_acc.txt', 'w')
    loss_log = open(log_name + '_loss.txt', 'w')

    # define warmup
    if args.dataset == 'cifar10':
        if args.method == 'reg':
            warm_up = 20 if args.aug else 10
        else:
            warm_up = 5
        num_classes = 10
    elif args.dataset == 'cifar100':
        if args.method == 'reg':
            warm_up = 60 if args.aug else 30
        else:
            warm_up = 5
        num_classes = 100
    else:
        raise ValueError('Wrong dataset')

    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=5, root_dir=args.data_path, log=stats_log,
                                         noise_file='%s/%.2f_%s.json' % (args.data_path, args.r, args.noise_mode),
                                         stronger_aug=args.aug)

    print('| Building net')
    if args.method == 'bit':
        create_model = create_model_bit
    elif args.method == 'reg':
        create_model = create_model_reg
    elif args.method == 'selfsup':
        create_model = create_model_selfsup
    else:
        raise ValueError()
    net1 = create_model(net=args.net, dataset=args.dataset, num_classes=num_classes, device=args.device, drop=args.drop)
    net2 = create_model(net=args.net, dataset=args.dataset, num_classes=num_classes, device=args.device, drop=args.drop)
    cudnn.benchmark = False  # True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    sched1 = torch.optim.lr_scheduler.StepLR(optimizer1, 150, gamma=0.1)
    sched2 = torch.optim.lr_scheduler.StepLR(optimizer2, 150, gamma=0.1)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode == 'asym':
        conf_penalty = NegEntropy()
    else:
        conf_penalty = None
    all_loss = [[], []]  # save the history of losses from two networks
    run_train_loop(net1, optimizer1, sched1, net2, optimizer2, sched2, criterion, CEloss, CE, loader, args.p_threshold,
                   warm_up, args.num_epochs, all_loss, args.batch_size, num_classes, args.device, args.lambda_u, args.T,
                   args.alpha, args.noise_mode, args.dataset, args.r, conf_penalty, stats_log, loss_log, test_log)


if __name__ == '__main__':
    main()
