import json
import medmnist
import numpy as np
import os.path
import random
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as transforms
from collections import OrderedDict
from medmnist import INFO, Evaluator
from medmnist.info import INFO, DEFAULT_ROOT
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import datasets
from tqdm import trange

import utils
from dataset import Dataset
from evaluation_measures import evaluate_measures

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0

#This class is for training and evaluation of the models
class Evaluate:
    def __init__(self, batch_size, dataset_name, medmnist_dataset):
        self.dataset = Dataset()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.medmnist_dataset = medmnist_dataset
        self.optimizer = None
        # Checking for the dataset
        if dataset_name == 'CIFAR-10':
            self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.get_dataset(
                self.batch_size)
        elif dataset_name == 'CIFAR-100':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_cifar100(self.batch_size)
        elif dataset_name == 'FASHIONMNIST':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_fashionmnist(
                self.batch_size)
        elif dataset_name == 'IMAGENET':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_imagenet(self.batch_size)
        elif dataset_name == 'MEDMNIST':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_medmnist(medmnist_dataset,
                                                                                                       self.batch_size)
        elif dataset_name == 'GASHISSDB':
            self.train_loader, self.valid_loader, self.test_loader, classes = self.dataset.get_gashisdb(self.batch_size)
        elif dataset_name == 'MHIST':
            self.train_loader, self.valid_loader, self.test_loader, classes = self.dataset.get_mhist(self.batch_size)
        # self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.get_dataset(self.batch_size)
        self.lr = 0.001
        self.scheduler = None

    def __train(self, model, train_loader, task, criterion, optimizer, device, writer):
        total_loss = []
        global iteration

        model.train()
        # Training the model
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, x = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)

            total_loss.append(loss.item())
            writer.add_scalar('train_loss_logs', loss.item(), iteration)
            iteration += 1

            loss.backward()
            optimizer.step()

        epoch_loss = sum(total_loss) / len(total_loss)
        return epoch_loss

    def __test(self, model, evaluator, data_loader, task, criterion, device, run, type_task, save_folder=None):
        # Testing the model
        check_evaluator = medmnist.Evaluator(self.medmnist_dataset, type_task)
        info = INFO[self.medmnist_dataset]
        task = info["task"]
        root = DEFAULT_ROOT
        npz_file = np.load(os.path.join(root, "{}.npz".format((self.medmnist_dataset))))
        if type_task == 'train':
            self.labels = npz_file['train_labels']
        elif type_task == 'val':
            self.labels = npz_file['val_labels']
        elif type_task == 'test':
            self.labels = npz_file['test_labels']
        else:
            raise ValueError

        model.eval()

        total_loss = []
        y_score = torch.tensor([]).to(device)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                outputs, x = model(inputs.to(device))

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).to(device)
                    loss = criterion(outputs, targets)
                    m = nn.Sigmoid()
                    outputs = m(outputs).to(device)
                else:
                    targets = torch.squeeze(targets, 1).long().to(device)
                    loss = criterion(outputs, targets)
                    m = nn.Softmax(dim=1)
                    outputs = m(outputs).to(device)
                    targets = targets.float().resize_(len(targets), 1)

                total_loss.append(loss.item())
                y_score = torch.cat((y_score, outputs), 0)

            y_score = y_score.detach().cpu().numpy()
            auc, acc = evaluator.evaluate(y_score, save_folder, run)
            f1 = evaluate_measures(self.labels, y_score, task)
            test_loss = sum(total_loss) / len(total_loss)

            return [test_loss, auc, acc, f1]

    # This function is for training
    def train(self, model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
              batch_size, download, run):
        # Setting the parameters
        as_rgb = True
        resize = False
        lr = 0.001
        gamma = 0.1
        milestones = [0.5 * num_epochs, 0.75 * num_epochs]

        info = INFO[data_flag]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])

        str_ids = gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

        output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        print('==> Preparing data...')

        data_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])

        train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
        val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
        test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)

        print('==> Building and training model...')

        model = model.to(device)

        train_evaluator = medmnist.Evaluator(data_flag, 'train')
        val_evaluator = medmnist.Evaluator(data_flag, 'val')
        test_evaluator = medmnist.Evaluator(data_flag, 'test')

        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, 'train',
                                    output_root)
        val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run, 'val', output_root)
        test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run, 'test',
                                   output_root)

        print('train  auc: %.5f  acc: %.5f\n  f1: %.5f\n' % (train_metrics[1], train_metrics[2], train_metrics[3]) + \
              'val  auc: %.5f  acc: %.5f\n f1: %.5f\n' % (val_metrics[1], val_metrics[2], val_metrics[3]) + \
              'test  auc: %.5f  acc: %.5f\n f1: %.5f\n' % (test_metrics[1], test_metrics[2], test_metrics[3]))

        if num_epochs == 0:
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        logs = ['loss', 'auc', 'acc']
        train_logs = ['train_' + log for log in logs]
        val_logs = ['val_' + log for log in logs]
        test_logs = ['test_' + log for log in logs]
        log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

        writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

        best_auc = 0
        best_epoch = 0
        best_model = model

        global iteration
        iteration = 0
        # Training the models till the given epochs
        for epoch in trange(num_epochs):
            train_loss = self.__train(model, train_loader, task, criterion, optimizer, device, writer)

            train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run,
                                        'train')
            val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run, 'val')
            test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run, 'test')

            scheduler.step()

            for i, key in enumerate(train_logs):
                log_dict[key] = train_metrics[i]
            for i, key in enumerate(val_logs):
                log_dict[key] = val_metrics[i]
            for i, key in enumerate(test_logs):
                log_dict[key] = test_metrics[i]

            for key, value in log_dict.items():
                writer.add_scalar(key, value, epoch)

            cur_auc = val_metrics[1]
            if cur_auc > best_auc:
                best_epoch = epoch
                best_auc = cur_auc
                best_model = model
                print('cur_best_auc:', best_auc)
                print('cur_best_epoch', best_epoch)

        state = {
            'net': best_model.state_dict(),
        }

        path = os.path.join(output_root, 'best_model.pth')
        torch.save(state, path)

        train_metrics = self.__test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run,
                                    'train',
                                    output_root)
        val_metrics = self.__test(best_model, val_evaluator, val_loader, task, criterion, device, run, 'val',
                                  output_root)
        test_metrics = self.__test(best_model, test_evaluator, test_loader, task, criterion, device, run, 'test',
                                   output_root)

        train_log = 'train  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (
        train_metrics[1], train_metrics[2], train_metrics[3])
        val_log = 'val  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (val_metrics[1], val_metrics[2], train_metrics[3])
        test_log = 'test  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (test_metrics[1], test_metrics[2], train_metrics[3])

        log = '%s\n' % (data_flag) + train_log + val_log + test_log
        print(log)

        with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
            f.write(log)

        writer.close()

        # Retuning the fitness values
        if evaluation == 'valid':
            return 1 - val_metrics[1]
        else:
            return 1 - test_metrics[1]
