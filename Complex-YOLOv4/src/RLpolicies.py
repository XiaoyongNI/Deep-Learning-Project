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
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data.distributed
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
sys.path.append('./')
from data_process.RL_dataloader import create_RL_dataloader, create_patch
import argparse
from constants import num_windows, num_actions, img_size_fd, img_size_cd
from utils.evaluation_utils import get_patch_policies, ensemble

parser = argparse.ArgumentParser(description='RL policies generation')
parser.add_argument('--num_samples', type=int, default=1201,
                    help='Take a subset of the dataset to run and debug')
parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of threads for loading data')
parser.add_argument('--batch_size', type=int, default=1,
                    help='mini-batch size (default: 1)')
# parser.add_argument('--load',type = str,default='../RLsave/old_val/ckpt_E_200_R_6.11E-01',
#                     help='checkpoint to load agent from')
parser.add_argument('--savetxt', type = str, default='../RLsave/reg_policies.txt',
                    help='file path to save the policies')
configs = edict(vars(parser.parse_args()))
configs.distributed = False  # For testing
configs.pin_memory = False
configs.dataset_dir = os.path.join('../', 'dataset', 'kitti')


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
    def __init__(self, cfg, num_classes=16, BEV_WIDTH=608):
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


def RegNetX_200MF(num_classes, BEV_WIDTH):
    cfg = {
        'depths': [1, 1, 4, 7],
        'widths': [24, 56, 152, 368],
        'strides': [1, 1, 2, 2],
        'group_width': 8,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg, num_classes, BEV_WIDTH)


def RegNetX_400MF(num_classes, BEV_WIDTH):
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg, num_classes, BEV_WIDTH)


def RegNetY_400MF(num_classes, BEV_WIDTH):
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0.25,
    }
    return RegNet(cfg, num_classes, BEV_WIDTH)


# load training dataset
configs_lr = deepcopy(configs)
configs_lr.input_cfg = LR_IMAGE_CFG
configs_hr = deepcopy(configs)
configs_hr.input_cfg = HR_IMAGE_CFG
evalloader, eval_sampler = create_RL_dataloader(configs_lr, configs_hr,split_first = False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

patch_flag = True
num_classes = num_actions if not patch_flag else 1
BEV_WIDTH = img_size_cd if not patch_flag else 64

def load_agents(dir_path,model_list):
    m_list = []
    for m in model_list:
        if 'resnet18' in m:
            agent = torchvision.models.resnet18(num_classes = num_classes).to(device)
        elif 'resnet34' in m:
            agent = torchvision.models.resnet34(num_classes=num_classes).to(device)
        elif 'resnet50' in m:
            agent = torchvision.models.resnet50(num_classes=num_classes).to(device)
        elif 'reg' in m and '1.6gf' not in m:
            agent = RegNetX_200MF(num_classes,BEV_WIDTH).to(device)
        elif 'regnet1.6gf' in m:
            agent = torchvision.models.regnet_x_1_6gf(num_classes= num_classes,stem_width = BEV_WIDTH).to(device)
        #file_path = dir_path + m.split('_')[0] + '/' + m + '.pth'
        file_path = dir_path + m + '/ckpt_E_10.pth'
        checkpoint = torch.load(file_path,map_location=device)
        agent.load_state_dict(checkpoint['agent'])
        print('loaded agent from %s' % file_path)
        m_list.append(agent)
    return m_list


if not os.path.isfile(configs.savetxt):
    f = open(configs.savetxt, "x")
    f.close()
else:
    f= open(configs.savetxt, "w")
    f.close()

def policy(agents,device):
    for agent in agents:
        agent.eval()
    timels = []

    for batch_idx, (img_paths, img_lr, img_hr, targets) in enumerate(tqdm(evalloader)):
        with torch.no_grad():
            start = time.time()
            inputs = img_lr.to(device)
            patches_lr, _, _ = create_patch(img_lr, img_hr, targets)
            # Actions by the Agent's output
            probs_list = torch.zeros(inputs.size(0),num_actions,len(agents)).to(device)
            for i,agent in enumerate(agents):
                if not patch_flag:
                    probs = F.sigmoid(agent.forward(inputs))
                else:
                    probs = F.sigmoid(agent.forward(patches_lr.to(device)))
                    probs = probs.view(-1,num_actions).contiguous()
                probs_list[:,:,i] = probs[:,:]
            policies = ensemble(probs_list,method = 'boltzmann_multiplication')
            policies = get_patch_policies(policies,img_paths)
            end = time.time()
            t = end-start
            timels.append(t)
            t_array = t * np.ones(shape = (policies.size(0),1))
            save = np.concatenate((policies.data.cpu(),t_array),axis = 1)
            with open(configs.savetxt,'a') as f:
                np.savetxt(f,save)
    time_avr = sum(timels) / len(timels)
    print('average time: %.3f s' % time_avr)
    return time_avr

if __name__ == '__main__':
    # specify the RL checkpoint names, checkpoints should be stored in dir_path/model_list[i]/ckpt_E_10.pth
    dir_path = '../RLsave/'
    model_list = ['reg_' + str(i) for i in range(1,14)]
    agents = load_agents(dir_path,model_list)
    policy(agents,device)


