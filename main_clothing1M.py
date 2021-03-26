from __future__ import print_function

import argparse
import os
import random
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.mixture import GaussianMixture

from dataloaders import dataloader_clothing1M as dataloader
from models.resnet import SupCEResNet
from train import train, warmup


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
    parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=80, type=int)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--id', default='clothing1m')
    parser.add_argument('--data_path', default='../../Clothing1M/data', type=str, help='path to dataset')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_class', default=14, type=int)
    parser.add_argument('--num_batches', default=1000, type=int)
    parser.add_argument('--experiment-name', required=True, type=str)
    parser.add_argument('--method', default='reg', type=str, help='method')
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


def val(net, val_loader, best_acc, k, exp_id, experiment_name):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Validation\t Net%d  Acc: %.2f%%" % (k, acc))
    if acc > best_acc[k - 1]:
        best_acc[k - 1] = acc
        print('| Saving Best Net%d ...' % k)
        save_point = './checkpoint/%s_%s_net%d.pth.tar' % (exp_id, experiment_name, k)
        torch.save(net.state_dict(), save_point)
    return acc


def run_test(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Acc: %.2f%%\n" % (acc))
    return acc


def eval_train(epoch, model, eval_loader, criterion, num_batches, batch_size, stats_log):
    model.eval()
    num_samples = num_batches * batch_size + 37497  # add for intersection
    losses = torch.zeros(num_samples)
    paths = []
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, targets, clean_target, path, _) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            for b in range(inputs.size(0)):
                losses[n] = loss[b]
                paths.append(path[b])
                n += 1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' % (batch_idx))
            sys.stdout.flush()

    losses_noisy = losses[:num_batches * batch_size]
    losses = (losses - losses_noisy.min()) / (losses_noisy.max() - losses_noisy.min())
    losses_noisy = losses[:num_batches * batch_size]
    losses, losses_noisy = losses.reshape(-1, 1), losses_noisy.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=100, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses_noisy)

    clean_idx, noisy_idx = gmm.means_.argmin(), gmm.means_.argmax()
    stats_log.write('GMM results: {} with variance {} and weight {}\t'
                    '{} with variance {} and weight {}\n'.format(gmm.means_[clean_idx], gmm.covariances_[clean_idx],
                                                                 gmm.weights_[clean_idx],
                                                                 gmm.means_[noisy_idx], gmm.covariances_[noisy_idx],
                                                                 gmm.weights_[noisy_idx], ))
    stats_log.flush()

    prob = gmm.predict_proba(losses)
    prob = prob[:, clean_idx]
    return prob, paths


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model_selfsup(net='resnet50', num_class=14):
    chekpoint = torch.load('pretrained/ckpt_clothing_{}.pth'.format(net))
    sd = {}
    for ke in chekpoint['model']:
        nk = ke.replace('module.', '')
        sd[nk] = chekpoint['model'][ke]
    model = SupCEResNet(net, num_classes=num_class, pool=True)
    model.load_state_dict(sd, strict=False)
    model = model.cuda()
    return model


def create_model_reg(net='resnet50', num_class=14):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, num_class)
    model = model.cuda()
    return model


def main():
    args = parse_args()
    os.makedirs('./checkpoint', exist_ok=True)
    log_name = './checkpoint/%s_%s' % (args.experiment_name, args.id)
    stats_log = open(log_name + '_stats.txt', 'w')
    test_log = open(log_name + '_acc.txt', 'w')
    test_log.flush()

    loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=5,
                                            num_batches=args.num_batches, log=stats_log)

    print('| Building net')
    if args.method == 'reg':
        create_model = create_model_reg
    elif args.method == 'selfsup':
        create_model = create_model_selfsup
    else:
        raise ValueError()

    net1 = create_model(net='resnet50', num_class=args.num_class)
    net2 = create_model(net='resnet50', num_class=args.num_class)
    cudnn.benchmark = True

    optimizer1 = optim.AdamW(net1.parameters(), lr=args.lr, weight_decay=1e-3)
    optimizer2 = optim.AdamW(net2.parameters(), lr=args.lr, weight_decay=1e-3)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()

    best_acc = [0, 0]
    for epoch in range(args.num_epochs + 1):
        lr = args.lr
        if epoch >= 40:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        if epoch < args.warmup:  # warm up
            train_loader = loader.run('warmup')
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, train_loader, CEloss, conf_penalty, args.device, 'clothing', None,
                   args.num_epochs, None)
            train_loader = loader.run('warmup')
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, train_loader, CEloss, conf_penalty, args.device, 'clothing', None,
                   args.num_epochs, None)

            if epoch > 1:
                print('\n\nEval Net2')
                pred2 = (prob2 > args.p_threshold)
                labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2,
                                                                        paths=paths2)  # co-divide
        else:
            pred1 = (prob1 > args.p_threshold)  # divide dataset
            pred2 = (prob2 > args.p_threshold)

            print('\n\nTrain Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2, paths=paths2)  # co-divide
            train(epoch, net1, net2, None, optimizer1, labeled_trainloader, unlabeled_trainloader, 0, args.batch_size,
                  args.num_class, args.device, args.T, args.alpha, args.warmup, 'clothing', None, None,
                  args.num_epochs, smooth_clean=True)  # train net1
            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1, paths=paths1)  # co-divide
            train(epoch, net2, net1, None, optimizer2, labeled_trainloader, unlabeled_trainloader, 0, args.batch_size,
                  args.num_class, args.device, args.T, args.alpha, args.warmup, 'clothing', None, None,
                  args.num_epochs, smooth_clean=True)  # train net2

        val_loader = loader.run('val')  # validation
        acc1 = val(net1, val_loader, best_acc, 1, args.id, args.experiment_name)
        acc2 = val(net2, val_loader, best_acc, 2, args.id, args.experiment_name)
        test_log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n' % (epoch, acc1, acc2))
        test_log.flush()
        print('\n==== net 1 evaluate next epoch training data loss ====')
        eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch
        prob1, paths1 = eval_train(epoch, net1, eval_loader, CE, args.num_batches, args.batch_size, stats_log)
        print('\n==== net 2 evaluate next epoch training data loss ====')
        eval_loader = loader.run('eval_train')
        prob2, paths2 = eval_train(epoch, net2, eval_loader, CE, args.num_batches, args.batch_size, stats_log)

    test_loader = loader.run('test')
    net1.load_state_dict(torch.load('./checkpoint/%s_%s_net1.pth.tar' % (args.id, args.experiment_name)))
    net2.load_state_dict(torch.load('./checkpoint/%s_%s_net2.pth.tar' % (args.id, args.experiment_name)))
    acc = run_test(net1, net2, test_loader)

    test_log.write('Test Accuracy:%.2f\n' % (acc))
    test_log.flush()


if __name__ == '__main__':
    main()
