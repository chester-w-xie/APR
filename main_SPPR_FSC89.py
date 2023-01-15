"""
-------------------------------File info-------------------------
% - File name: main_SPPR_ESC_FSCIL.py
% - Description:
% -
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-07-15
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import torch
import numpy as np
import argparse

import random
import os
import logging
import sys
import time
import torch.nn as nn
from DatasetsManager_FSC89 import fsc89_dataset_for_fscil
from FSCIL_model_define_V1 import FSCILmodel
from torchvision import models
from Utils import filter_para, CategoriesSampler, LR_Scheduler, \
    calculate_confusion_matrix, calculate_accuracy, format_time
from torch.utils.data import DataLoader
from tqdm import tqdm
from results_assemble import get_results_assemble


class Trainer(object):
    def __init__(self, args):

        self.scheduler = None
        self.args = args

        self.datasets = fsc89_dataset_for_fscil(args)
        self.label_per_task = [list(np.array(range(args.base_class)))] + [list(np.array(range(args.way)) +
                                                                               args.way * task_id + args.base_class)
                                                                          for task_id in range(args.tasks)]
        self.base_class_num = args.base_class
        self.test_results_one_trial = {}
        self.test_results_all_trial = {}
        self.num_sessions = args.session
        # Define model and optimizer
        model = FSCILmodel(backbone=models.resnet18, pretrained=args.pretrained, args=args)

        self.pretrain_model_dir = os.path.join(args.pretrained_model_path,
                                               'pretrained_model_' + args.dataset_name + '.pth')

        optim_para = filter_para(model, args)  # -

        optimizer = torch.optim.SGD(optim_para, weight_decay=args.weight_decay, nesterov=args.nesterov)

        self.criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')
        self.criterion = self.criterion.cuda()

        # Using cuda
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.cuda()
        self.model, self.optimizer = model, optimizer

        self.best_pred = 0.0
        self.val_loss_min = None
        self.best_result_dic = {}
        self.early_stopping_count = 0
        # history of prediction
        self.acc_history = []
        self.best_model_dir = os.path.join(args.dir_name, 'session_0_bset_model_' + args.dataset_name + '.pth')

    def fit(self):
        # pretraining

        if self.args.pretrained is True:
            logging.info("Using a backbone that has been pre-trained on ImageNet")
        else:
            logging.info('Using a randomly initialized backbone')

        logging.info("Start meta training, a.k.a base session training...")
        self.meta_train()

        logging.info("Start meta-testing, including updating the model with "
                     "the support set, and then evaluate the model \n")
        logging.info('The meta-test consisted of %d trials, each containing %d incremental sessions within the trial.'
                     % (self.args.trials, self.num_sessions - 1))
        logging.info('The final meta-test result is the average of the results of all %d trials.\n' % self.args.trials)
        meta_test_start_time = time.time()
        for trial in range(self.args.trials):
            #
            meta_model = FSCILmodel(backbone=models.resnet18, pretrained=args.pretrained, args=args)
            para = torch.load(self.best_model_dir)
            meta_model.load_state_dict(para)
            logging.info('Meta testing (Support set: %d way %d shot):' % (self.args.way, self.args.shot))
            for session in range(1, self.num_sessions):
                updated_model = self.meta_test(session, trial, meta_model)
                meta_model = updated_model
            self.test_results_all_trial[trial] = self.test_results_one_trial.copy()
        meta_test_end_time = time.time()

        meta_spend_time = (meta_test_end_time - meta_test_start_time) / self.args.trials
        meta_spend_time = meta_spend_time / (self.num_sessions - 1)
        #
        avg_meta_test_time = format_time(meta_spend_time)
        logging.info('meta-testing is done! avg running time (raw) over sessions is {:8}.\n'.format(meta_spend_time))
        logging.info(
            'meta-testing is done! avg running time (format) over sessions is {:8}.\n'.format(avg_meta_test_time))

        results_save_path = os.path.join(self.args.dir_name, 'test_results_{}_trial.pth'.format(self.args.trials))
        torch.save(self.test_results_all_trial, results_save_path)
        print(f'All results have been saved to {results_save_path}')
        #
        get_results_assemble(results_save_path)

    def prtraining(self):
        pass

    def meta_train(self, current_session=0, current_trial=1):
        train_dataset = self.datasets['train'][current_session]
        val_dataset = self.datasets['val']
        session_class = self.args.base_class + self.args.way * current_session
        epochs = self.args.base_epochs

        # if current_session == 0:
        if self.args.val:
            para = torch.load(self.args.model_path)
            self.model.load_state_dict(para)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True)
            train_sampler = CategoriesSampler(train_dataset.sub_indexes,
                                              len(train_loader),
                                              self.args.way + self.args.batch_task,
                                              self.args.shot)
            train_fsl_loader = DataLoader(dataset=train_dataset,
                                          batch_sampler=train_sampler,
                                          num_workers=4,
                                          pin_memory=True)
            self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr, epochs,
                                          len(train_loader), args=self.args)
            for epoch in range(epochs):
                self.model.train()
                tbar = tqdm(train_loader)

                train_loss = 0.0
                num_iter = len(train_loader)

                for i, batch_samples in enumerate(zip(tbar, train_fsl_loader)):

                    query_samples, query_targets = batch_samples[0][0], batch_samples[0][1]
                    support_samples, support_targets = batch_samples[1][0], batch_samples[1][1]

                    if self.args.cuda:
                        query_samples, query_targets = query_samples.cuda(), query_targets.cuda()
                        support_samples, support_targets = support_samples.cuda(), support_targets.cuda()
                    if not self.args.no_lr_scheduler:
                        self.scheduler(self.optimizer, i, epoch, self.best_pred)
                    self.optimizer.zero_grad()
                    outputs = self.model(query_samples, support_samples, support_targets)[:, :session_class]

                    loss = self.criterion(outputs, (
                        query_targets.view(-1, 1).repeat(1, self.args.batch_task).view(-1)).long())

                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                val_loss = self.validation(val_dataset)
                self.keep_record_of_best_model(val_loss, epoch)

                logging.info('[Session: {}/{}, Epoch: {}/{},'
                             ' num. of training samples: {}.'
                             ' ==> training loss: {:.3f},'
                             ' , val loss: {:.3f}]\n'.format(current_session, self.num_sessions,
                                                             epoch + 1, epochs,
                                                             (num_iter - 1) * self.args.batch_size +
                                                             query_samples.data.shape[0],
                                                             train_loss / num_iter, val_loss)
                             )

                if self.early_stopping_count > self.args.early_stop_tol:
                    torch.save(self.best_result_dic['model'], self.best_model_dir)
                    logging.info('meta-training is done, the best model is saving to %s \n' % self.best_model_dir)

                    logging.info('Early stop condition is met at epoch {},'
                                 ' break the training. \n'.format(epoch))
                    # 这里可以读取最优模型进行session 0 的评估
                    self.evaluate(current_session, current_trial, self.model)
                    break

            torch.save(self.best_result_dic['model'], self.best_model_dir)
            logging.info('meta-training is done, the best model is saving to %s \n' % self.best_model_dir)

            #
            self.evaluate(current_session, current_trial, self.model)

    def validation(self, dataset):
        self.model.eval()

        val_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        vbar = tqdm(val_loader)
        session_class = self.args.base_class

        outputs = []
        targets = []
        for i, batch_samples in enumerate(vbar):
            sample, target = batch_samples[0], batch_samples[1]
            targets.append(target)
            if self.args.cuda:
                sample = sample.cuda()
            with torch.no_grad():
                batch_output = self.model.forward_test(sample)[:, :session_class]
                outputs.append(batch_output.data.cpu().numpy())

        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        val_loss = float(self.criterion(torch.Tensor(outputs), torch.LongTensor(targets)).numpy())

        return val_loss

    def keep_record_of_best_model(self, val_loss, epoch):
        self.early_stopping_count += 1
        if self.val_loss_min is None or val_loss < self.val_loss_min:
            logging.info('Update best model and reset counting.')

            self.early_stopping_count = 0
            self.val_loss_min = val_loss
            # undate result dic
            self.best_result_dic = {'val_loss': val_loss,
                                    'model': self.model.state_dict(),
                                    'epoch': epoch
                                    }

    def meta_test(self, current_session, current_trial, _trained_model):

        meta_test_datasets = fsc89_dataset_for_fscil(self.args)

        meta_loader = DataLoader(meta_test_datasets['train'][current_session], batch_size=2048,
                                 shuffle=False, num_workers=4,
                                 pin_memory=True)

        for i, batch_samples in enumerate(meta_loader):
            support_samples, support_targets = batch_samples[0], batch_samples[1]
            if self.args.cuda:
                support_samples, support_targets = support_samples.cuda(), support_targets.cuda()
            with torch.no_grad():
                _trained_model.calculate_means(support_samples)
        self.evaluate(current_session, current_trial, _trained_model)

        return _trained_model

    def evaluate(self, current_session, current_trial, trained_model):

        eval_model = trained_model
        eval_model.eval()

        test_dataset = self.datasets['test'][current_session]
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        session_class = self.args.base_class + self.args.way * current_session

        outputs = []
        targets = []
        for i, sample in enumerate(test_loader):
            image, target = sample[0], sample[1]
            targets.append(target)
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                batch_output = eval_model.forward_test(image)[:, :session_class]
                outputs.append(batch_output.data.cpu().numpy())

        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)

        audio_predictions = np.argmax(outputs, axis=-1)  # (audios_num,)
        # Evaluate
        classes_num = outputs.shape[-1]

        test_set_acc_overall = calculate_accuracy(targets, audio_predictions,
                                                  classes_num, average='macro')
        class_wise_acc = calculate_accuracy(targets, audio_predictions, classes_num)
        cf_matrix = calculate_confusion_matrix(targets, audio_predictions, classes_num)

        class_wise_acc_base = class_wise_acc[:self.base_class_num]
        # -
        class_wise_acc_all_novel = class_wise_acc[self.base_class_num:]
        #
        class_wise_acc_previous_novel = class_wise_acc[self.base_class_num:(self.base_class_num + self.args.way)]
        #
        class_wise_acc_current_novel = class_wise_acc[-self.args.way:]

        # Test
        logging.info('[Trial: %d, Session: %d, num. of seen classes: %d,'
                     ' num. test samples: %5d]' % (current_trial, current_session,
                                                   session_class, i * self.args.batch_size + image.data.shape[0]))

        if current_session == 0:
            logging.info("==> Average of class wise acc: {:.2f} (base)"
                         ", - (all novel)"
                         ", - (previous novel)"
                         ", - (current novel)"
                         ", {:.2f} (both)\n".format(np.mean(class_wise_acc_base) * 100,
                                                    test_set_acc_overall * 100)
                         )

            ave_acc_all_novel = None
            ave_acc_previous_novel = None
            ave_acc_current_novel = None
        else:

            ave_acc_all_novel = np.mean(class_wise_acc_all_novel)
            ave_acc_previous_novel = np.mean(class_wise_acc_previous_novel)
            ave_acc_current_novel = np.mean(class_wise_acc_current_novel)

            logging.info("==> Average of class wise acc: {:.2f} (base)"
                         ", {:.2f} (all novel)"
                         ", {:.2f} (previous novel)"
                         ", {:.2f} (current novel)"
                         ", {:.2f} (both)\n".format(np.mean(class_wise_acc_base) * 100,
                                                    ave_acc_all_novel * 100,
                                                    ave_acc_previous_novel * 100,
                                                    ave_acc_current_novel * 100,
                                                    test_set_acc_overall * 100)
                         )

        session_results_dict = {'Ave_class_wise_acc_base': np.mean(class_wise_acc_base),
                                'Ave_class_wise_acc_all_novel': ave_acc_all_novel,
                                'Ave_class_wise_acc_previous_novel': ave_acc_previous_novel,
                                'Ave_class_wise_acc_current_novel': ave_acc_current_novel,
                                'Ave_acc_of_both': test_set_acc_overall,
                                }
        self.test_results_one_trial[current_session] = session_results_dict.copy()

        if current_session == self.num_sessions - 1:
            self.show_results_summary(current_trial)

    def show_results_summary(self, current_trial):

        base_avg_over_sessions = []
        all_avg_novel_over_sessions = []
        pre_avg_novel_over_sessoins = []
        curr_avg_novel_over_sessions = []
        both_avg_over_sessions = []

        logging.info('=====> Trial {} results summary, '
                     '(Support set: {} way {} shot)'.format(current_trial, self.args.way, self.args.shot))
        print(f'-------------------- Average of class-wise acc (%)--------------------------------')
        print(f'\n Session         ', end=" ")
        for _, n in enumerate(self.test_results_one_trial.keys()):
            print(f'{n}', end="\t")
        print(f'Average', end="\t")

        print(f'\n Base      ', end="\t")
        for _, n in enumerate(self.test_results_one_trial.keys()):
            temp = self.test_results_one_trial[n]['Ave_class_wise_acc_base']
            print(f'{temp * 100:.2f}', end="\t")
            base_avg_over_sessions.append(temp)

        print(f'{np.mean(base_avg_over_sessions) * 100:.2f}', end="\t")

        print(f'\n All Novel       ', end=" ")
        for _, n in enumerate(self.test_results_one_trial.keys()):

            if n == 0:
                print(f'-', end="\t")
            else:
                temp = self.test_results_one_trial[n]['Ave_class_wise_acc_all_novel']
                print(f'{temp * 100:.2f}', end="\t")
                all_avg_novel_over_sessions.append(temp)
        print(f'{np.mean(all_avg_novel_over_sessions) * 100:.2f}', end="\t")
        print(f'\n Previous Novel  ', end=" ")
        for _, n in enumerate(self.test_results_one_trial.keys()):

            if n == 0:
                print(f'-', end="\t")
            else:
                temp = self.test_results_one_trial[n]['Ave_class_wise_acc_previous_novel']
                print(f'{temp * 100:.2f}', end="\t")
                pre_avg_novel_over_sessoins.append(temp)

        print(f'{np.mean(pre_avg_novel_over_sessoins) * 100:.2f}', end="\t")

        print(f'\n Current Novel   ', end=" ")
        for _, n in enumerate(self.test_results_one_trial.keys()):

            if n == 0:
                print(f'-', end="\t")
            else:
                temp = self.test_results_one_trial[n]['Ave_class_wise_acc_current_novel']
                print(f'{temp * 100:.2f}', end="\t")
                curr_avg_novel_over_sessions.append(temp)
        print(f'{np.mean(curr_avg_novel_over_sessions) * 100:.2f}', end="\t")

        print(f'\n Both     ', end="\t")
        for _, n in enumerate(self.test_results_one_trial.keys()):
            temp = self.test_results_one_trial[n]['Ave_acc_of_both']
            print(f'{temp * 100:.2f}', end="\t")
            both_avg_over_sessions.append(temp)
        print(f'{np.mean(both_avg_over_sessions) * 100:.2f}', end="\t")
        print(f'\n --------------------------------------------------------------------------------\n ')

        PD = self.test_results_one_trial[0]['Ave_acc_of_both'] - \
             self.test_results_one_trial[self.num_sessions - 1]['Ave_acc_of_both']

        temp2 = self.test_results_one_trial[0]['Ave_class_wise_acc_base'] - \
                self.test_results_one_trial[self.num_sessions - 1]['Ave_class_wise_acc_base']
        AR_overall = temp2 / self.test_results_one_trial[0]['Ave_class_wise_acc_base']
        AR_overall_avg = AR_overall / (self.num_sessions - 1)
        MR_overall = 1 - AR_overall

        AR_session_list = []
        AR_session_list_temp = []
        for _session in range(1, self.num_sessions):
            acc_previous = self.test_results_one_trial[_session - 1]['Ave_class_wise_acc_base']
            acc_current = self.test_results_one_trial[_session]['Ave_class_wise_acc_base']
            AR_session = (acc_previous - acc_current) / acc_previous
            AR_session_list.append(AR_session)
            AR_session_list_temp.append(AR_session * 100)
        AR_session_avg = np.mean(AR_session_list)
        MSR_session_avg = 1 - AR_session_avg

        CPI = 0.5 * MSR_session_avg + 0.5 * np.mean(all_avg_novel_over_sessions)
        CPI_V2 = 0.5 * MR_overall + 0.5 * np.mean(all_avg_novel_over_sessions)

        logging.info(' ==> PD: {:.2f} (define by CEC); \n'.format(PD * 100))

        logging.info(' =====> AR_overall: {:.2f}, AR_overall_avg: {:.2f},'
                     ' MR_overall: {:.2f}; \n'.format(AR_overall * 100, AR_overall_avg * 100, MR_overall * 100))

        logging.info(
            '  =====> AR_session_avg: {:.2f}, '
            'MSR_session_avg: {:.2f};'.format(AR_session_avg * 100, MSR_session_avg * 100))

        logging.info('  =====> Average of all novel acc over {} incremental sessions: {:.2f};'.format(
            self.num_sessions - 1, np.mean(all_avg_novel_over_sessions) * 100))
        logging.info('  =====> CPI: {:.2f} \n'.format(CPI * 100))
        logging.info('  =====> CPI_V2: {:.2f} \n'.format(CPI_V2 * 100))


def setup_parser():
    parser = argparse.ArgumentParser(description='FSCIL_for_audio')

    # dir
    parser.add_argument('--metapath', type=str, required=True, help='path to FSC-89-meta folder')
    parser.add_argument('--datapath', type=str, required=True, help='path to FSD-MIX-CLIPS_data folder)')
    parser.add_argument('--setup', type=str, required=True, help='mini or huge')
    parser.add_argument('--data_type', type=str, required=True, help='audio or openl3)')
    parser.add_argument('--num_class', type=int, default=89, help='Total number of classes in the dataset')

    # dataset option
    parser.add_argument('--dataset_name', type=str, default='FSC89',
                        help='dataset name (default: FSC89)')

    # dataset setting(class-division, way, shot)
    parser.add_argument('--base_class', type=int, default=59, help='number of base class (default: 60)')
    parser.add_argument('--way', type=int, default=5, help='class number of per task (default: 5)')
    parser.add_argument('--shot', type=int, default=5, help='shot of per class (default: 5)')

    # model option
    parser.add_argument('--fscil_method', type=str, default='SPPR', help='fscil method (default: None)')
    parser.add_argument('--model_path', type=str, default=None, help='model path (default: None)')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='pretrained model path (default: None)')

    parser.add_argument('--pretrained', action='store_true', default=False, help='pretrained model')
    parser.add_argument('--backbone_name', type=str, default='resnet18', help='backbone name (default: resnet18)')
    parser.add_argument('--init_fic', type=str, default='None', choices=['None', 'identical'],
                        help='init_fic name (default: None)')
    parser.add_argument('--batch_task', type=int, default=3, help='tasks per batch')
    parser.add_argument('--embedding', type=int, default=64, choices=[64, 128, 256, 512], help='channel of embedding')
    parser.add_argument('--latent_dim', type=int, default=512, help='channel of latent')

    # gpu option
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cudnn', action='store_true', default=True, help='enables CUDNN accelerate')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')

    # loss option
    parser.add_argument('--loss_type', type=str, default='mse', help='type of loss (default: ce)')
    parser.add_argument('--loss_weight', type=int, default=1, metavar='N', help='weight ratio of loss (default: 1)')

    # random seed
    parser.add_argument('--seed', type=int, default=1668, help='random seed (default: 1993)')

    # hyper option
    parser.add_argument('--session', type=int, default=11, metavar='N',
                        help='num. of sessions, including one base session and n incremental sessions (default:10)')
    parser.add_argument('--trials', type=int, default=100, metavar='N',
                        help='num. of trials for the incremental sessions (default:100)')
    parser.add_argument('--early_stop_tol', type=int, default=10, metavar='N',
                        help='tolerance for early stopping (default:10)')
    parser.add_argument('--base_epochs', type=int, default=90, metavar='N', help='base epochs (default:50)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch_size (default:128)')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--no-lr-scheduler', '-a', action='store_true', default=False,
                        help='avoid lr-schduler (default: False)')
    parser.add_argument('--lr_coefficient', nargs='+', type=float, default=[1, 1, 1, 1], help='list of lr_coefficient')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')

    # optimizer option
    parser.add_argument('--opt', type=str, default='opt1', choices=['opt1', 'opt2'],
                        help='type of learnable para (default: opt1)')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='type of optimizer (default: sgd)')
    #
    # evaluation options
    parser.add_argument('--val', action='store_true', default=False, help='val mode')

    _args = parser.parse_args()

    return _args


def set_device(args):
    # if args.cudnn:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


def update_param(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":

    args = setup_parser()
    set_device(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.data_path = os.path.join(args.data_dir, args.dataset_name)

    args.tasks = args.session - 1
    args.all_class = args.base_class + args.way * args.tasks
    args.now_time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

    args.dir_name = 'exp/' + str(args.dataset_name) + str(args.fscil_method) + '_' \
                    + str(args.way) + 'way' + '_' + str(args.shot) + 'shot' + '_' + str(args.now_time)
    args.pretrained_model_path = 'exp/' + str(args.dataset_name) + '-' + str(args.num_class) + '-FS_' + str(args.fscil_method) + '_' \
                    + str(args.way) + 'way' + '_' + str(args.shot) + 'shot' + '_' + 'Pretrain_V2'

    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)

    logging.basicConfig(level=logging.INFO,
                        filename=args.dir_name + '/output_logging_' + args.now_time + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('\nAll args of the experiment ====>')
    logging.info(args)
    logging.info('\n\n')

    start_time = time.time()
    trainer = Trainer(args)
    trainer.fit()
    end_time = time.time()

    time_spent = format_time(end_time - start_time)
    logging.info('All done! The entire process took {:8}.\n'.format(time_spent))
