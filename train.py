import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torchvision

from conf import settings
from src.utils import get_training_dataloader, get_test_dataloader, WarmUpLR
from src.model import FuseNet

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        #print(outputs.size())
        #print(labels.size())
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        
        print('\rTraining Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ), end = "")
        sys.stdout.flush()

        #update training loss for each iteration


    finish = time.time()

    print('\repoch {} training time consumed: {:.2f}s'.format(epoch, finish - start), end = "")
    sys.stdout.flush()
@torch.no_grad()
def eval_training(epoch):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('\rGPU INFO.....', end = "")
        sys.stdout.flush()
        print(torch.cuda.memory_summary(), end='')
        sys.stdout.flush()
    print('\rEvaluating Network.....', end = "")
    sys.stdout.flush()
    print('\rTest set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ), end = "")
    sys.stdout.flush()
    print('\n')
    #add informations to tensorboard
    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-H', type=int, default=224, help='height of input images')
    parser.add_argument('-W', type=int, default=224, help='width of input images')
    parser.add_argument('-C', type=int, default=3, help='number of channels in input images')
    args = parser.parse_args()
    #print(args)
    net = FuseNet(args.H, args.W, args.C)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'fuse', settings.TIME_NOW)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net='fuse', epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net='fuse', epoch=epoch, type='regular'))


