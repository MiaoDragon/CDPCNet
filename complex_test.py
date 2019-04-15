"""
part of this training pipeline is inspired by:
     https://github.com/fxia22/pointnet.pytorch
"""
from model.simple_siamesePointnet import SiamesePointNet as SimpleSiamesePointNet
from model.siamesePointnet import SiamesePointNet as SiamesePointNet
from dataset.complex_loader import create_split_loaders
from save_util import *
import argparse
import random
import numpy as np
import torch
import os
import torch.optim as optim
import torch.nn as nn
import sys
from tqdm import tqdm
def main(args):
    purple = lambda x: '\033[45m' + x + '\033[0m'
    blue = lambda x: '\033[44m' + x + '\033[0m'
    # load model if exists
    start_model = args.out_path+args.model+'_%d.pth' % (args.start_epoch)
    if os.path.exists(start_model):
        args.seed = load_seed(start_model)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    extras = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}
    trainloader, valloader, testloader = create_split_loaders(args.data_path, args.data_size,\
                                                              args.data_minibatch, \
                                                              args.batch_size, args.seed, \
                                                              p_val=0.1, p_test=0.2, \
                                                              extras=extras)
    try:
        os.makedirs(args.out_path)
    except OSError:
        pass
    if torch.cuda.is_available():
        computing_device = torch.device("cuda")
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        print("CUDA NOT supported")

    if args.model_type == 'simple':
        classifier = SimpleSiamesePointNet(feature_transform=args.feature_transform)
    elif args.model_type == 'deep':
        classifier = SiamesePointNet(feature_transform=args.feature_transform)
    if os.path.exists(start_model):
        load_net_state(classifier, start_model)
    # setup for training
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    if os.path.exists(start_model):
        load_opt_state(optimizer, start_model)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifer = classifier.to(computing_device)
    loss_func = nn.CrossEntropyLoss()
    num_batch = len(trainloader)
    # evaluation metrics
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    test_loss = []
    test_acc = []
    if os.path.exists(start_model):
        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = load_eval(start_model)


    total_correct = 0
    total_testset = 0
    for i,data in tqdm(enumerate(testloader, 1)):
        P1, P2, target = data
        points = torch.cat((P1, P2), dim=0)
        #target = target[:, 0]
        #points = points.transpose(2, 1)
        points, target = points.to(computing_device), target.to(computing_device)
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_acc = total_correct / len(testloader) / args.batch_size
    print("final accuracy {}".format(total_acc))
    sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type=int, required=True, help="dataset size")
parser.add_argument('--data_minibatch', type=int, help='minibatch size when saving data')
parser.add_argument(
    '--batch_size', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--num_epoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=-1)
parser.add_argument('--out_path', type=str, default='output/', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--pin_memory', type=int, default=1)
parser.add_argument('--val_batch', type=int, default=10)
parser.add_argument('--data_path', type=str, default='dataset/complex/')
parser.add_argument('--model_type', type=str, default='simple')
args = parser.parse_args()
print(args)
main(args)
