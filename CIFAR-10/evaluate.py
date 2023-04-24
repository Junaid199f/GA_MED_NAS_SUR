# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:48:33 2022

@author: Muhammad Junaid Ali (IRMAS Lab, University Haute Alsace)
"""


import logging
import os.path

import torch
import torch.nn as nn
import torch.optim as optim

import utils
from dataset import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0


class Evaluate:
    def __init__(self, batch_size):
        self.dataset = Dataset()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.get_dataset(self.batch_size)
        self.lr = 0.001
        self.scheduler = None

    # Training
    def __train(self, net, epoch, grad_clip):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs, x = net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def __test(self, net, epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, x = net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                utils.progress_bar(batch_idx, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                   % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                logging.info("Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        return acc

    def __test_final(self, net, epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, x = net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                utils.progress_bar(batch_idx, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                   % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                logging.info("Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        return acc

    def train(self, model, epochs, hash_indv, grad_clip, warmup=False):
        model = model.to(device)
        self.optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.AdamW(model.parameters(),lr = 0.025,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(epochs))
        acc = 0
        for epoch in range(0, epochs):
            self.__train(model, epoch, grad_clip)
            acc = self.__test(model, epoch)
            self.scheduler.step()
        loss = 100 - acc
        # with open(os.path.join(os.path.join(os.path.join(os.getcwd(),'checkpoints'),str(hash_indv)),'output.json'), 'w') as json_file:
        #     json.dump(state, json_file)
        return loss

    def train_final(self, model, epochs, hash_indv, grad_clip, warmup=False):
        model = model.to(device)
        self.optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.AdamW(model.parameters(),lr = 0.025,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(epochs))
        acc = 0
        for epoch in range(0, epochs):
            self.__train(model, epoch, grad_clip)
            acc = self.__test_final(model, epoch)
            self.scheduler.step()
        loss = 100 - acc
        # with open(os.path.join(os.path.join(os.path.join(os.getcwd(),'checkpoints'),str(hash_indv)),'output.json'), 'w') as json_file:
        #     json.dump(state, json_file)
        return loss
