import json
import os
import pickle
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def save_preds(exp, probability, clean):
    name = './stats/cifar100/stats{}.pcl'
    nm = name.format(exp)
    if os.path.exists(nm):
        probs_history, clean_history = pickle.load(open(nm, "rb"))
    else:
        probs_history, clean_history = [], []
    probs_history.append(probability)
    clean_history.append(clean)
    pickle.dump((probs_history, clean_history), open(nm, "wb"))


def get_asym_cifar100(root_dir):
    super_class = {}
    super_class['aquatic mammals'] = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
    super_class['fish'] = ['aquarium fish', 'flatfish', 'ray', 'shark', 'trout']
    super_class['flowers'] = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
    super_class['food containers'] = ['bottle', 'bowl', 'can', 'cup', 'plate']
    super_class['fruit and vegetables'] = ['apple', 'mushroom', 'orange', 'pear', 'sweet pepper']
    super_class['household electrical devices'] = ['clock', 'keyboard', 'lamp', 'telephone', 'television']
    super_class['household furniture'] = ['bed', 'chair', 'couch', 'table', 'wardrobe']
    super_class['insects'] = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']
    super_class['large carnivores'] = ['bear', 'leopard', 'lion', 'tiger', 'wolf']
    super_class['large man-made outdoor things'] = ['bridge', 'castle', 'house', 'road', 'skyscraper']
    super_class['large natural outdoor scenes'] = ['cloud', 'forest', 'mountain', 'plain', 'sea']
    super_class['large omnivores and herbivores'] = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
    super_class['medium mammals'] = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']
    super_class['non-insect invertebrates'] = ['crab', 'lobster', 'snail', 'spider', 'worm']
    super_class['people'] = ['baby', 'boy', 'girl', 'man', 'woman']
    super_class['reptiles'] = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
    super_class['small mammals'] = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
    super_class['trees'] = ['maple tree', 'oak tree', 'palm tree', 'pine tree', 'willow tree']
    super_class['vehicles 1'] = ['bicycle', 'bus', 'motorcycle', 'pickup truck', 'train']
    super_class['vehicles 2'] = ['lawn mower', 'rocket', 'streetcar', 'tank', 'tractor']

    classes_to_mix = [[] for _ in range(20)]
    with open('{}/meta'.format(root_dir), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        for j, fine in enumerate(entry['fine_label_names']):
            fine = fine.replace('_', ' ')
            for i, coarse in enumerate(entry['coarse_label_names']):
                coarse = coarse.replace('_', ' ')
                if fine in super_class[coarse]:
                    classes_to_mix[i].append(j)

    return classes_to_mix


class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[],
                 log='', oracle='none', mix_labelled=True):
        assert oracle in ('none', 'positive', 'negative', 'all', 'negative_shuffle')
        assert dataset in ('cifar10', 'cifar100')
        without_class = False

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        # class transition for asymmetric noise

        if dataset == 'cifar10':
            self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
        elif dataset == 'cifar100':
            self.transition = get_asym_cifar100(root_dir)

        self.mix_labelled = mix_labelled
        self.num_classes = 10 if dataset == 'cifar10' else 100

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
            else:  # inject noise
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r * 50000)
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            if without_class:
                                noiselabel = random.randint(0, self.num_classes - 2)
                                if noiselabel >= train_label[i]:
                                    noiselabel += 1
                            else:
                                noiselabel = random.randint(0, self.num_classes - 1)
                            noise_label.append(noiselabel)
                        elif noise_mode == 'asym':
                            if dataset == 'cifar10':
                                noiselabel = self.transition[train_label[i]]
                                noise_label.append(noiselabel)
                            elif dataset == 'cifar100':
                                z = [x.copy() for x in self.transition if train_label[i] in x][0]
                                z.remove(train_label[i])
                                noiselabel = random.choice(z)
                                noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])
                print("save noisy labels to %s ..." % noise_file)
                json.dump(noise_label, open(noise_file, "w"))

            self.clean = (np.array(noise_label) == np.array(train_label))
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
                self.train_label = train_label
            else:
                clean = (np.array(noise_label) == np.array(train_label))

                if oracle == 'negative':
                    pred = pred * (clean == 1)  # don't take noisy
                elif oracle == 'negative_shuffle':
                    pred_clean = (pred == 1) * (clean == 0)  # shuffle labels of FP
                    noise_label = np.array(noise_label)
                    noise_label[pred_clean] = np.random.randint(0, self.num_classes, len(noise_label[pred_clean]))
                elif oracle == 'positive':
                    pred = (pred + clean) > 0  # take all clean
                elif oracle == 'all':
                    pred = clean  # take only clean

                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    auc = roc_auc_score(clean, probability) if self.r > 0 else 1
                    tp, fp, fn = (np.equal(pred, clean) * (clean == 1)).sum(), \
                                 (np.not_equal(pred, clean) * (clean == 0)).sum(), \
                                 (np.not_equal(pred, clean) * (clean == 1)).sum()
                    # pc,nc = (clean==1).sum(), (clean==0).sum()
                    log.write('Number of labeled samples:%d\t'
                              'AUC:%.3f\tTP:%.3f\tFP:%.3f\tFN:%.3f\t'
                              'Noise in labeled dataset:%.3f\n' % (
                                  pred.sum(), auc, tp, fp, fn, fp / (tp + fp)))

                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, index, prob if self.mix_labelled else target
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode == 'all':
            img, target, clean = self.train_data[index], self.noise_label[index], self.train_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, index, clean
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file='',
                 stronger_aug=False):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_warmup = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15.,
                                        translate=(0.1, 0.1),
                                        scale=(2. / 3, 3. / 2),
                                        shear=(-0.1, 0.1, -0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            aug = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomCrop(32, padding=4),
                # transforms.Pad(4),
                # transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(3./4, 4./3), interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15.,
                                        translate=(0.1, 0.1),
                                        scale=(2. / 3, 3. / 2),
                                        shear=(-0.1, 0.1, -0.1, 0.1)),
                # transforms.RandomGrayscale(p=0.25),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                # transforms.RandomErasing(value='random', inplace=True),
            ])

            self.transform_warmup = aug
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

        self.transform_warmup = self.transform_warmup if stronger_aug else self.transform_train
        self.clean = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                   root_dir=self.root_dir, transform=self.transform_warmup, mode="all",
                                   noise_file=self.noise_file).clean

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_warmup, mode="all",
                                        noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                            root_dir=self.root_dir, transform=self.transform_train, mode="labeled",
                                            noise_file=self.noise_file, pred=pred, probability=prob, log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                              root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",
                                              noise_file=self.noise_file, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                         noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
