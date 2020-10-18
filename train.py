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

import warnings
warnings.filterwarnings("ignore")

import wandb

from google.cloud import storage

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
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
        
        if epoch <= args.warm:
            warmup_scheduler.step()
        
        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        
        print('\rTraining Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(cifar100_training_loader.dataset)), end="")
        sys.stdout.flush()
        
        if batch_index%args.log_interval==0:
            wandb.log({
                "Train Loss": loss.item(),
                "LR":optimizer.param_groups[0]['lr']
                })
        
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

    example_images = []
    
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

    wandb.log({
        "Test Accuracy": 100. * correct / len(cifar100_test_loader.dataset),
        "Test Loss": test_loss/len(cifar100_test_loader.dataset)})   
 
    return correct.float() / len(cifar100_test_loader.dataset)

def save_model(args, name):
    """Saves the model to Google Cloud Storage
    Args:
      args: contains name for saved model.
    """
    scheme = 'gs://'
    bucket_name = args.job_dir[len(scheme):].split('/')[0]

    prefix = '{}{}/'.format(scheme, bucket_name)
    bucket_path = args.job_dir[len(prefix):].rstrip('/')

    datetime_ = datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S')

    if bucket_path:
        model_path = '{}/{}/{}'.format(bucket_path, datetime_, name)
    else:
        model_path = '{}/{}'.format(datetime_, name)

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(name)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', 
        type=str2bool, 
        nargs='?', 
        const=True, 
        default=False, 
        help='use gpu or not'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=128, 
        help='batch size for dataloader'
    )
    parser.add_argument(
        '--warm', 
        type=int, 
        default=1, 
        help='warm up training phase'
    )
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.1, 
        help='initial learning rate'
    )
    parser.add_argument(
        '--momentum', 
        type=float, 
        default=0.5, 
        help='momentum for SGD optimizer'
    )
    parser.add_argument(
        '--height', 
        type=int, 
        default=224, 
        help='height of input images'
    )
    parser.add_argument(
        '--width', 
        type=int, 
        default=224, 
        help='width of input images'
    )
    parser.add_argument(
        '--channels', 
        type=int, default=3,    
        help='number of channels in input images'
    )
    parser.add_argument(
        '--job-dir', 
        help='path for saving images in gcs'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=120, 
        help='number of epochs of training'
    )
    parser.add_argument(
        '--gamma', 
        type=float, 
        default=0.2, 
        help='learning rate decay rate'
    )
    parser.add_argument(
        '--weight-decay', 
        type=float, 
        default=5e-4, 
        help='weight decay rate'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='random manual seed'
    )
    parser.add_argument(
        '--log-interval', 
        type=int, 
        default=20, 
        help='wandb log interval'
    )
    args = parser.parse_args()
    print(args)
    net = FuseNet(args.height, args.width, args.channels)

    if args.gpu:
        net.cuda()    

    wandb.init(entity="shandilya1998", project="assignment3-pytorch", config=args)
    wandb.watch_called = False
    MILESTONES = [int(args.epochs/4), int(args.epochs/2), int(3*args.epochs/4)]
    
    torch.manual_seed(args.seed)
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True
    )

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=args.gamma) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'fuse', settings.TIME_NOW)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    wandb.watch(net, log="all")
    
    best_acc = 0.0
    for epoch in range(1, args.epochs):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        name = checkpoint_path.format(net='fuse', epoch=epoch, type='best')
        #start to save best performance model after learning rate decay to 0.01
        if epoch > MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), name)
            wandb.save(name)
            best_acc = acc
            save_model(args, name)
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), name)
            wandb.save(name)
            save_model(args, name)

