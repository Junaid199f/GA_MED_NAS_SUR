import os.path
from collections import OrderedDict
import torch.utils.data as data


from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from dataset import Dataset
import random
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
import medmnist
from medmnist import INFO, Evaluator
import utils
import json
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
class Evaluate:
    def __init__(self, batch_size,dataset_name,medmnist_dataset):
        self.dataset = Dataset()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.medmnist_dataset = medmnist_dataset
        self.optimizer = None
        if dataset_name == 'CIFAR-10':
            self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.get_dataset(self.batch_size)
        elif dataset_name == 'CIFAR-100':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_cifar100(self.batch_size)
        elif dataset_name == 'FASHIONMNIST':
            self.train_loader, self.valid_loader, self.test_loader =  self.dataset.get_dataset_fashionmnist(self.batch_size)
        elif dataset_name == 'IMAGENET':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_imagenet(self.batch_size)
        elif dataset_name == 'MEDMNIST':
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_medmnist(medmnist_dataset,self.batch_size)
        elif dataset_name == 'GASHISSDB':
            self.train_loader, self.valid_loader, self.test_loader, classes = self.dataset.get_gashisdb(self.batch_size)
        elif dataset_name == 'MHIST':
            self.train_loader, self.valid_loader, self.test_loader, classes = self.dataset.get_mhist(self.batch_size)
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.get_dataset(self.batch_size)
        self.lr = 0.001
        self.scheduler = None
    #
    # # Training
    # def __train(self,net,epoch,grad_clip):
    #     print('\nEpoch: %d' % epoch)
    #     net.train()
    #     train_loss = 0
    #     correct = 0
    #     total = 0
    #     for batch_idx, (inputs, targets) in enumerate(self.train_loader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         self.optimizer.zero_grad()
    #         outputs,x = net(inputs)
    #         loss = self.criterion(outputs, targets)
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    #         self.optimizer.step()
    #
    #         train_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()
    #
    #         utils.progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #                      % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    #
    # def __test(self,net,epoch):
    #     global best_acc
    #     net.eval()
    #     test_loss = 0
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for batch_idx, (inputs, targets) in enumerate(self.test_loader):
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             outputs,x = net(inputs)
    #             loss = self.criterion(outputs, targets)
    #
    #             test_loss += loss.item()
    #             _, predicted = outputs.max(1)
    #             total += targets.size(0)
    #             correct += predicted.eq(targets).sum().item()
    #
    #             utils.progress_bar(batch_idx, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #                          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    #
    #     # Save checkpoint.
    #     acc = 100. * correct / total
    #     if acc > best_acc:
    #         print('Saving..')
    #         state = {
    #             'net': net.state_dict(),
    #             'acc': acc,
    #             'epoch': epoch,
    #         }
    #         if not os.path.isdir('checkpoint'):
    #             os.mkdir('checkpoint')
    #         torch.save(state, './checkpoint/ckpt.pth')
    #         best_acc = acc
    #     return acc
    # def train(self, model, epochs,hash_indv,grad_clip,warmup=False):
    #     model = model.to(device)
    #     self.optimizer = optim.SGD(model.parameters(), lr=0.0001,momentum=0.9, weight_decay=5e-4)
    #     #self.optimizer = optim.AdamW(model.parameters(),lr = 0.025,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(epochs))
    #     acc = 0
    #     for epoch in range(0, epochs):
    #         self.__train(model,epoch,grad_clip)
    #         acc = self.__test(model,epoch)
    #         self.scheduler.step()
    #     loss = 100- acc
    #     # with open(os.path.join(os.path.join(os.path.join(os.getcwd(),'checkpoints'),str(hash_indv)),'output.json'), 'w') as json_file:
    #     #     json.dump(state, json_file)
    #     return loss


    def __train(self,model, train_loader, task, criterion, optimizer, device, writer):
        total_loss = []
        global iteration

        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs,x = model(inputs.to(device))

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


    def __test(self,model, evaluator, data_loader, task, criterion, device, run, save_folder=None):
        model.eval()

        total_loss = []
        y_score = torch.tensor([]).to(device)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                outputs,x = model(inputs.to(device))

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

            test_loss = sum(total_loss) / len(total_loss)

            return [test_loss, auc, acc]


    def train(self,model,epochs,hash_indv,grad_clip,data_flag, output_root, num_epochs, gpu_ids, batch_size, download, run):
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

        #device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        print(device)
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
        train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
        val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
        test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

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

        for epoch in trange(num_epochs):
            train_loss = self.__train(model, train_loader, task, criterion, optimizer, device, writer)

            train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run)
            val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run)
            test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run)

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
                             output_root)
        val_metrics = self.__test(best_model, val_evaluator, val_loader, task, criterion, device, run, output_root)
        test_metrics = self.__test(best_model, test_evaluator, test_loader, task, criterion, device, run, output_root)

        train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
        val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
        test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

        log = '%s\n' % (data_flag) + train_log + val_log + test_log
        print(log)

        with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
            f.write(log)

        writer.close()

        return 1 - test_metrics[1]