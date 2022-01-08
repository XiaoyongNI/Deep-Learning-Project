import sys
import os
import time
import numpy as np
#import csv
import random
from easydict import EasyDict as edict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from config.input_config import LR_IMAGE_CFG, HR_IMAGE_CFG, LR_PATCH_CFG, HR_PATCH_CFG
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboard_logger import configure, log_value
import torch.distributed as dist
import torch.optim as optim
from torch.autograd import Variable
#import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.distributions import Bernoulli
from tqdm import tqdm
from copy import deepcopy
sys.path.append('./')
from data_process.RL_dataloader import create_RL_dataloader, create_patch
import argparse
from utils import utils, utils_detector
from constants import num_windows, num_actions, img_size_fd, img_size_cd

# config for RL agent
parser = argparse.ArgumentParser(description='PolicyNetworkTraining')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='../RLsave/old', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--save_intervals',default = 10,help='At every N epoch save the checkpoint')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--img_size', type=int, default=448, help='PN Image Size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=200, help='total epochs to run')
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
 

# blocks of RegNet
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
    def __init__(self, cfg, num_classes=16,BEV_WIDTH=608):
        super(RegNet, self).__init__()
        self.cfg = cfg
        self.in_planes = BEV_WIDTH
        self.conv1 = nn.Conv2d(3, BEV_WIDTH, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(BEV_WIDTH)
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
 
 
def RegNetX_200MF(num_classes,BEV_WIDTH):
    cfg = {
        'depths': [1, 1, 4, 7],
        'widths': [24, 56, 152, 368],
        'strides': [1, 1, 2, 2],
        'group_width': 8,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg,num_classes,BEV_WIDTH)
 
 
def RegNetX_400MF(num_classes,BEV_WIDTH):
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg,num_classes,BEV_WIDTH)
 
 
def RegNetY_400MF(num_classes,BEV_WIDTH):
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0.25,
    }
    return RegNet(cfg,num_classes,BEV_WIDTH)


# config for Complex YOLO for dataloader
parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')
# parser.add_argument('--img_size', type=int, default=64,
#                     help='the size of input image')
parser.add_argument('--num_samples', type=int, default=None,
                    help='Take a subset of the dataset to run and debug')
parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of threads for loading data')
parser.add_argument('--batch_size', type=int, default=16,
                    help='mini-batch size (default: 1)')
configs = edict(vars(parser.parse_args()))
configs.distributed = False  # For testing
configs.pin_memory = False
configs.dataset_dir = os.path.join('../', 'dataset', 'kitti')

# load training dataset
configs_lr = deepcopy(configs)
configs_lr.input_cfg = LR_IMAGE_CFG
configs_hr = deepcopy(configs)
configs_hr.input_cfg = HR_IMAGE_CFG
trainloader, train_sampler = create_RL_dataloader(configs_lr, configs_hr)

#num_actions = num_windows * num_windows # Hyperparameter, should be equal to num_windows * num_windows
#num_windows = 4 # Number of windows in one dimension
# number of actions equals to the number of patches 

cudaFLAG = True

'''
RLtrain_ids = []
with open('../dataset/kitti/val_id.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile,delimiter=' ')
    for row in spamreader:
        id_int = int(float(row[0]))
        id_str = '00' + str(id_int)
        if id_str not in RLtrain_ids:
            RLtrain_ids.append(id_str)
'''


def train(epoch, agent):
    agent.train()
    rewards, rewards_baseline, policies = [], [], []

    for batch_idx, (img_paths, img_lr, img_hr, targets) in enumerate(tqdm(trainloader)):
        #patch_lr,patch_hr,patch_target = create_patch(img_lr, img_hr, target)
        #data_time.update(time.time() - start_time)

        #img_id = img_path[index].split('image_2/')[1].replace('.png', '.npy')

        inputs = img_lr
        inputs = Variable(inputs)
        if not args.parallel:
            if cudaFLAG:
                inputs = inputs.cuda()

        # Actions by the Agent's output
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
        # yolo models are part of the environment of the RL model
        # use pre calculated metric mARs
        # img_paths are the paths to images in a batch
        offset_fd, offset_cd = utils.read_offsets(img_paths, num_actions)

        # Find the reward for baseline and sampled policy
        reward_map = utils.compute_reward(offset_fd, offset_cd, policy_map.data, args.beta, args.sigma)
        reward_sample = utils.compute_reward(offset_fd, offset_cd, policy_sample.data, args.beta, args.sigma)
        if cudaFLAG:
            advantage = reward_sample.cuda().float() - reward_map.cuda().float()
        else:
            advantage = reward_sample.float() - reward_map.float()

        # config for detector 
        #lr_cfg = edict()
        #hr_cfg = edict()
        #lr_cfg.cfgfile = './config/cfg/complex_yolov4_lr.cfg'
        #hr_cfg.cfgfile = './config/cfg/complex_yolov4_hr.cfg'

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

    if epoch % args.save_intervals==0 and epoch != 0:
        agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
        state = {
            'agent': agent_state_dict,
            'epoch': epoch,
            'reward': reward,
        }
        torch.save(state, args.cv_dir + '/ckpt_E_%d_R_%.2E' % (epoch, reward))

# Save the args to the checkpoint directory
if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)
configure(args.cv_dir+'/log', flush_secs=5)

# create an agent with RegNetX_200MF
num_classes = num_actions
BEV_WIDTH = img_size_cd
if False:
    agent = RegNetX_200MF(num_classes,BEV_WIDTH)
else:
    agent = torchvision.models.resnet50(num_classes = num_classes)
    if True:
        pretrained = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
        # pretrained = 'https://download.pytorch.org/models/resnet34-b627a593.pth'
        pretrained_dict = torch.hub.load_state_dict_from_url(pretrained)
        model_dict = agent.state_dict()
        new_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model_dict.update(new_dict)
        agent.load_state_dict(model_dict)
    
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

if cudaFLAG:
    agent.cuda()

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)


# Start training and testing
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch,agent)
    #if epoch % args.test_epoch == 0:
    #    test(epoch)
