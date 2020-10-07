import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader


class clothing_dataset(Dataset):
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14,
                 add_clean=False, log=None, clean_all=False):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.noisy_labels = {}
        self.clean_labels = {}
        self.val_labels = {}
        self.clean_all = clean_all

        with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.noisy_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.clean_labels[img_path] = int(entry[1])
        # Clean size: 72409. Noisy size: 1037497. Clean/noisy intersection: 37497 (24637/5395/7465)

        if mode == 'all':
            train_imgs = []
            with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    train_imgs.append(img_path)
            random.shuffle(train_imgs)  # 1M images, not including double-labeled
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.noisy_labels[impath]
                if class_num[label] < (num_samples / 14) and len(self.train_imgs) < num_samples:
                    self.train_imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.train_imgs)
            if add_clean:
                inter_imgs = []
                for impath in self.clean_labels:
                    if impath in self.noisy_labels:
                        inter_imgs.append(impath)
                self.train_imgs += inter_imgs  # add images which have a clean label too to be able to calculate metrics
        elif self.mode == "labeled":
            train_imgs = paths[:num_samples]
            pred_idx = pred[:num_samples].nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

            predicted_path = []
            predicted_prob = []
            predicted_clean = []
            predicted_pred = []
            for i, img_path in enumerate(paths[num_samples:]):
                if img_path in self.clean_labels:
                    predicted_path.append(img_path)
                    predicted_prob.append(probability[num_samples + i])
                    predicted_pred.append(pred[num_samples + i])
                    predicted_clean.append(self.clean_labels[img_path] == self.noisy_labels[img_path])
            # print('Curr epoch clean/noisy intersection: {}'.format(len(predicted_path)))
            if len(predicted_path) > 0:
                predicted_prob, predicted_clean = np.array(predicted_prob), np.array(predicted_clean)
                predicted_pred = np.array(predicted_pred)
                auc = roc_auc_score(predicted_clean, predicted_prob)
                tp, fp, fn = (np.equal(predicted_pred, predicted_clean) * (predicted_clean == 1)).sum(), \
                             (np.not_equal(predicted_pred, predicted_clean) * (predicted_clean == 0)).sum(), \
                             (np.not_equal(predicted_pred, predicted_clean) * (predicted_clean == 1)).sum()
                # pc,nc = (clean==1).sum(), (clean==0).sum()
                log.write('Number of labeled samples:%d\t'
                          'AUC:%.3f\tTP:%.3f\tFP:%.3f\tFN:%.3f\t'
                          'Noise in labeled dataset:%.3f\n' % (
                              pred[:num_samples].sum(), auc, tp, fp, fn, fp / (tp + fp)))
                log.flush()
        elif self.mode == "unlabeled":
            train_imgs = paths[:num_samples]
            pred_idx = (1 - pred[:num_samples]).nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))
        elif mode == 'test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    self.test_imgs.append(img_path)
        elif mode == 'val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.noisy_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, 0, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.noisy_labels[img_path] if not self.clean_all else self.clean_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, img_path, 0
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.clean_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.clean_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        if self.mode == 'val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class clothing_dataloader():
    def __init__(self, root, batch_size, num_batches, num_workers, log, stronger_aug=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
        self.log = log

        self.transform_warmup = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomAffine(degrees=15.,
                                    translate=(0.1, 0.1),
                                    scale=(2. / 3, 3. / 2),
                                    shear=(-0.1, 0.1, -0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_warmup = self.transform_warmup if stronger_aug else self.transform_train
        self.warmup_samples = self.num_batches * self.batch_size * 4 if stronger_aug else self.num_batches * self.batch_size * 2

    def run(self, mode, pred=[], prob=[], paths=[]):
        if mode == 'warmup':
            warmup_dataset = clothing_dataset(self.root, transform=self.transform_warmup, mode='all',
                                              num_samples=self.warmup_samples, log=self.log)
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return warmup_loader
        elif mode == 'train':
            labeled_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='labeled', pred=pred,
                                               probability=prob, paths=paths,
                                               num_samples=self.num_batches * self.batch_size, log=self.log)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            unlabeled_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='unlabeled', pred=pred,
                                                 probability=prob, paths=paths,
                                                 num_samples=self.num_batches * self.batch_size, log=self.log)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_loader, unlabeled_loader
        elif mode == 'eval_train':
            eval_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='all',
                                            num_samples=self.num_batches * self.batch_size, add_clean=True,
                                            log=self.log)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
        elif mode == 'test':
            test_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='test', log=self.log)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=1000,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
        elif mode == 'val':
            val_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='val', log=self.log)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=1000,
                shuffle=False,
                num_workers=self.num_workers)
            return val_loader
