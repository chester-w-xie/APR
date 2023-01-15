"""
-------------------------------File info-------------------------
% - File name: Utils.py
% - Description:
% -
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-04-05
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import math
import random
import torch
import numpy as np
import abc
from torch import nn
from torch.utils.data import DataLoader, Dataset

global_first_time = True


#
class CategoriesSampler:

    def __init__(self, index, lenth, way, shot):
        self.index = index  # -
        # -
        #  -
        self.lenth = lenth  # -
        self.way = way
        self.shot = shot
        # print(f'index-1:{index[1]}')

    def __len__(self):
        return self.lenth

    def __iter__(self):

        for lenth in range(self.lenth):  # -
            batch = []
            labels_in_dataset = list(self.index.keys())  # -
            random.shuffle(labels_in_dataset)  # -
            classes_for_fs = labels_in_dataset[:self.way]  # -
            #
            for c in classes_for_fs:
                data_indexes_in_one_class = torch.from_numpy(self.index[c])  # -
                num_data_in_one_class = len(data_indexes_in_one_class)  # -

                shot = torch.randperm(num_data_in_one_class)[:self.shot]
                shot_real = data_indexes_in_one_class[:self.shot]

                batch.append((c*num_data_in_one_class+shot).int())  # -

            temp = torch.stack(batch)
            batch = temp.reshape(-1)

            yield batch


def filter_para(model, args):
    if args.opt == 'opt1':
        return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    elif args.opt == 'opt2':
        return [
            {'params': model.backbone.parameters(), 'lr': args.lr},
            {'params': model.embedding.parameters(), 'lr': args.lr},
            {'params': model.new_proto, 'lr': args.lr},
            {"params": model.IL_attn.parameters(), "lr": args.lr},
                ]


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Acc(self, session_class):
        confusion_matrix = self.confusion_matrix[:session_class, :session_class]
        return confusion_matrix, np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1), np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        # return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=30, warmup_epochs=0, args=None):
        self.mode = mode
        # print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        self.args = args
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.5 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        # if epoch > self.epoch:
        #     print('\n=>Epoches %i, learning rate = %.4f, \
        #         previous best = %.4f' % (epoch, lr, best_pred))
        #     self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        para_len = len(optimizer.param_groups)
        if para_len == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            for i in range(para_len):
                optimizer.param_groups[i]['lr'] = lr * self.args.lr_coefficient[i]
            # for i in range(1, len(optimizer.param_groups)):
            #     optimizer.param_groups[i]['lr'] = lr * 10


class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.compute_means = True

        # settings
        self.memory_budget = 2000
        self.norm_exemplars = True
        self.herding = True

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass

    # ###----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, n):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''

        # set model to eval()-mode
        # mode = self.training
        self.eval()

        n_max = len(dataset)
        exemplar_set = []
        class_mean = None
        if self.herding:
            # compute features for each example in [dataset]
            first_entry = True
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4,
                                    drop_last=False, pin_memory=True)
            for (image_batch, _) in dataloader:
                if self.args.cuda:
                    image_batch = image_batch.cuda()
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch)
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            # calculate mean of all features
            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            exemplar_features = torch.zeros_like(features[:min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum) / (k + 1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1).cpu())
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)

                exemplar_set.append(dataset[index_selected])
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 10000
        else:
            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            for k in indeces_selected:
                exemplar_set.append(dataset[k])
        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(exemplar_set)

        # set mode of model back
        # self.train(mode=mode)
        return class_mean


class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.exemplar_datasets = []
        for class_id in range(len(self.exemplar_sets)):
            self.exemplar_datasets += self.exemplar_sets[class_id]

    def __len__(self):
        return len(self.exemplar_datasets)

    def __getitem__(self, index):
        return self.exemplar_datasets[index]


def calculate_accuracy(target, predict, classes_num1, average=None):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """

    samples_num = len(target)

    correctness = np.zeros(classes_num1)
    total = np.zeros(classes_num1)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')


def calculate_confusion_matrix(target, predict, classes_num2):
    """Calculate confusion matrix.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes

    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num2, classes_num2))
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D '
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h '
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm '
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's '
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms '
        i += 1
    if f == '':
        f = '0ms '
    return f


class MixUpLossCrossEntropyLoss(nn.Module):
    """Adapts the loss function to go with mixup."""

    def __init__(self, crit=None):
        super().__init__()
        if crit is None:  # - 默认执行此句
            self.crit = nn.CrossEntropyLoss(reduction="none")
        else:
            self.crit = crit

    def forward(self, output, target1, target2=None, lmpas=None):
        global global_first_time
        if target2 is None:
            return self.crit(output, target1).mean()  # -
        if global_first_time:
            print("using mix up loss!! ", self.crit)  # -
        global_first_time = False
        # - crit 实际上就是初始化里定义的交叉熵
        loss1, loss2 = self.crit(output, target1), self.crit(output, target2)
        return (loss1 * lmpas + loss2 * (1 - lmpas)).mean()


class MixUpLossCrossEntropyLossV2(nn.Module):
    """Adapts the loss function to go with mixup."""
    #

    def __init__(self, crit=None):
        super().__init__()
        if crit is None:  # -
            # self.crit = nn.CrossEntropyLoss(reduction="none")
            self.crit = nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean') #
        else:
            self.crit = crit

    def forward(self, output, target1, target2=None, lmpas=None):
        global global_first_time
        if target2 is None:
            return self.crit(output, target1).mean()  # -
        if global_first_time:
            print("using mix up loss!! ", self.crit)  # -
        global_first_time = False
        # -
        loss1, loss2 = self.crit(output, target1), self.crit(output, target2)
        return (loss1 * lmpas + loss2 * (1 - lmpas)).mean()


def my_mixup_function(data, alpha):
    rn_indices = torch.randperm(data.size(0))
    lambd = np.random.beta(alpha, alpha, data.size(0))
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd.astype(np.float32))

    return rn_indices, lam


class SGDmScheduleV2:
    # -
    def __init__(self, optimizer, lr_init):
        self.lr_init = lr_init
        self.epochs = 250  # -
        self.start = 50  # -
        self.warmup = 0  # -
        self.lr_warmup = lr_init / 1000  # -
        self.after_lr = lr_init / 1000  # -
        self.optimizer = optimizer
        self.lr = self.lr_init

    # -
    def step(self, epoch):
        if epoch < self.warmup:  # -
            self.lr = self.lr_warmup + (self.lr_init - self.lr_warmup) * epoch / self.warmup
        elif epoch < self.start:
            self.lr = self.lr_init  # -
        elif epoch >= self.epochs:
            self.lr = self.after_lr  # -
        else:
            self.lr = self.lr_init - self.lr_init * (epoch - self.start) / (self.epochs - self.start)
            # -

        adjust_learning_rate(self.optimizer, self.lr)  # -

    # -
    def get_lr(self):
        return [self.lr]


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
