
"""
    This is a script that trains U-Nets from scratch
    Usage:
        python train.py --method 2D
"""

"""
    Necessary libraries
"""
import argparse
import datetime
import os
import torch
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

from data.datasets import *
from networks.cnn import CNN
from networks.cam import ModelCAM
from util.preprocessing import get_augmenters_2d
from util.losses import CrossEntropyLoss, MSELoss

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs")
parser.add_argument("--write_dir", help="Writing directory", type=str, default=None)
parser.add_argument("--data", help="Dataset for training", type=str, default="epfl") # options: 'epfl', 'embl_mito', 'embl_er', vnc, med
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses", type=int, default=1)

# network parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=str, default="128,128")
parser.add_argument("--augment_noise", help="Use noise augmentation", type=int, default=1)
parser.add_argument("--class_weight", help="Percentage of the reference class", type=float, default=(0.5))
parser.add_argument("--cam_maps", help="Number of activation maps", type=int, default=256)
parser.add_argument("--n_samples", help="Number of samples for training", type=int, default=500)

# optimization parameters
parser.add_argument("--pretrain_unsupervised", help="Flag whether to pre-train unsupervised", type=int, default=0)
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay", type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=1)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=200)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=32)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=16)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]
weight = torch.FloatTensor([1-args.class_weight, args.class_weight]).cuda()
loss_fn_seg = CrossEntropyLoss(weight=weight)
loss_fn_rec = MSELoss()

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if args.write_dir is not None:
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    os.mkdir(os.path.join(args.write_dir, 'segmentation_last_checkpoint'))
    os.mkdir(os.path.join(args.write_dir, 'segmentation_best_checkpoint'))

"""
    Load the data
"""
input_shape = (1, args.input_size[0], args.input_size[1])
# load supervised data
print('[%s] Loading data' % (datetime.datetime.now()))
train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters_2d(augment_noise=(args.augment_noise==1))
if args.data == 'epfl':
    train = EPFLPixelTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform, n_samples=args.n_samples)
    test = EPFLPixelTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
elif args.data == 'vnc':
    train = VNCPixelTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform, n_samples=args.n_samples)
    test = VNCPixelTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
elif args.data == 'med':
    train = MEDPixelTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform, n_samples=args.n_samples)
    test = MEDPixelTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
else:
    if args.data == 'embl_mito':
        train = EMBLMitoPixelTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform, n_samples=args.n_samples)
        test = EMBLMitoPixelTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
    else:
        train = EMBLERPixelTrainDataset(input_shape=input_shape, transform=train_xtransform, target_transform=train_ytransform, n_samples=args.n_samples)
        test = EMBLERPixelTestDataset(input_shape=input_shape, transform=test_xtransform, target_transform=test_ytransform)
train_loader = DataLoader(train, batch_size=args.train_batch_size)
test_loader = DataLoader(test, batch_size=args.test_batch_size)

"""
    Setup optimization for finetuning
"""
print('[%s] Setting up optimization for finetuning' % (datetime.datetime.now()))
# load best checkpoint
net = CNN()
cam_net = ModelCAM(net.encoder, net.encoder[-1].out_channels)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the network
"""
print('[%s] Training network' % (datetime.datetime.now()))
cam_net.train_net(train_loader=train_loader, test_loader=test_loader,
              loss_fn=loss_fn_seg, optimizer=optimizer, scheduler=scheduler,
              epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
              log_dir=args.log_dir)

print('[%s] Finished!' % (datetime.datetime.now()))