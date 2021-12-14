import sys
import os
import time
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensorboard_logger import configure, log_value
import torch.distributed as dist
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.distributions import Bernoulli
from tqdm import tqdm

sys.path.append('./')
#sys.path.append('../Complex-YOLOv4/src')

from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
from models.model_utils import create_model, make_data_parallel, get_num_parameters
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.train_utils import reduce_tensor, to_python_float, get_tensorboard_log
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.train_config import parse_train_configs
from evaluate import evaluate_mAP

import argparse

from utils import utils, utils_detector

parser = argparse.ArgumentParser(description='PolicyNetworkTraining')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--img_size', type=int, default=448, help='PN Image Size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=8, help='Number of Workers')
parser.add_argument('--test_epoch', type=int, default=10, help='At every N epoch test the network')
parser.add_argument('--parallel', action='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--beta', type=float, default=0.1, help='Coarse detector increment')
parser.add_argument('--sigma', type=float, default=0.5, help='cost for patch use')
args = parser.parse_args()

class SE(nn.Module):
    '''Squeeze-and-Excitation block.'''
 
    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)
 
    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = F.relu(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out
 
 
class Block(nn.Module):
    def __init__(self, w_in, w_out, stride, group_width, bottleneck_ratio, se_ratio):
        super(Block, self).__init__()
        # 1x1
        w_b = int(round(w_out * bottleneck_ratio))
        self.conv1 = nn.Conv2d(w_in, w_b, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w_b)
        # 3x3
        num_groups = w_b // group_width
        self.conv2 = nn.Conv2d(w_b, w_b, kernel_size=3,
                               stride=stride, padding=1, groups=num_groups, bias=False)
        self.bn2 = nn.BatchNorm2d(w_b)
        # se
        self.with_se = se_ratio > 0
        if self.with_se:
            w_se = int(round(w_in * se_ratio))
            self.se = SE(w_b, w_se)
        # 1x1
        self.conv3 = nn.Conv2d(w_b, w_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(w_out)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or w_in != w_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(w_in, w_out,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(w_out)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.with_se:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 

# output layer size = num_classes
class RegNet(nn.Module):
    def __init__(self, cfg, num_classes=2):
        super(RegNet, self).__init__()
        self.cfg = cfg
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(0)
        self.layer2 = self._make_layer(1)
        self.layer3 = self._make_layer(2)
        self.layer4 = self._make_layer(3)
        self.linear = nn.Linear(self.cfg['widths'][-1], num_classes)
 
    def _make_layer(self, idx):
        depth = self.cfg['depths'][idx]
        width = self.cfg['widths'][idx]
        stride = self.cfg['strides'][idx]
        group_width = self.cfg['group_width']
        bottleneck_ratio = self.cfg['bottleneck_ratio']
        se_ratio = self.cfg['se_ratio']
 
        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers.append(Block(self.in_planes, width,
                                s, group_width, bottleneck_ratio, se_ratio))
            self.in_planes = width
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
 
 
def RegNetX_200MF():
    cfg = {
        'depths': [1, 1, 4, 7],
        'widths': [24, 56, 152, 368],
        'strides': [1, 1, 2, 2],
        'group_width': 8,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg)
 
 
def RegNetX_400MF():
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg)
 
 
def RegNetY_400MF():
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0.25,
    }
    return RegNet(cfg)


agent = RegNetX_200MF()



configs = parse_train_configs()

# Create dataloader
trainloader, train_sampler = create_train_dataloader(configs)


num_actions = 9 # Hyperparameter, should be equal to num_windows * num_windows
num_windows = 3 # Number of windows in one dimension
epoch = 200

def train(epoch):
    agent.train()
    rewards, rewards_baseline, policies = [], [], []

    # iterate dataset
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs = Variable(inputs)
        if not args.parallel:
            inputs = inputs.cuda()

        # Actions by the Agent
        probs = F.sigmoid(agent.forward(inputs))
        alpha_hp = np.clip(args.alpha + epoch * 0.001, 0.6, 0.95)
        probs = probs*alpha_hp + (1-alpha_hp) * (1-probs)

        # Sample the policies from the Bernoulli distribution characterized by agent
        distr = Bernoulli(probs)
        policy_sample = distr.sample()

        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        # Get the batch wise metrics
        offset_fd, offset_cd = utils.read_offsets(targets, num_actions)

        # Find the reward for baseline and sampled policy
        # yolo models are part of the environment of the RL model
        reward_map = utils.compute_reward(offset_fd, offset_cd, policy_map.data, args.beta, args.sigma)
        reward_sample = utils.compute_reward(offset_fd, offset_cd, policy_sample.data, args.beta, args.sigma)
        advantage = reward_sample.cuda().float() - reward_map.cuda().float()

        # Find the loss for only the policy network
        loss = -distr.log_prob(policy_sample)
        loss = loss * Variable(advantage).expand_as(policy_sample)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_map.cpu())
        policies.append(policy_sample.data.cpu())

    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)

    print('Train: %d | Rw: %.2E | S: %.3f | V: %.3f | #: %d' % (epoch, reward, sparsity, variance, len(policy_set)))

    log_value('train_reward', reward, epoch)
    log_value('train_sparsity', sparsity, epoch)
    log_value('train_variance', variance, epoch)
    log_value('train_baseline_reward', torch.cat(rewards_baseline, 0).mean(), epoch)
    log_value('train_unique_policies', len(policy_set), epoch)

# ---- Load the pre-trained model ----------------------
start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print('loaded agent from %s' % args.load)


# Parallelize the models if multiple GPUs available - Important for Large Batch Size to Reduce Variance
if args.parallel:
    agent = nn.DataParallel(agent)
agent.cuda()

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# Save the args to the checkpoint directory
configure(args.cv_dir+'/log', flush_secs=5)

# Start training and testing
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    #if epoch % args.test_epoch == 0:
    #    test(epoch)