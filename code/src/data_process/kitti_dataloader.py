"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""

import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.kitti_dataset import KittiDataset
from data_process.transformation import Compose, OneOf, Random_Rotation, Random_Scaling, Horizontal_Flip, Cutout
from config.input_config import LR_IMAGE_CFG, LR_PATCH_CFG, HR_PATCH_CFG,HR_IMAGE_CFG


def create_train_dataloader(configs,split_first = True):
    """Create dataloader for training"""

    train_lidar_transforms = OneOf([
        Random_Rotation(limit_angle=20., p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0)
    ], p=0.66)

    train_aug_transforms = Compose([
        Horizontal_Flip(p=configs.hflip_prob),
        Cutout(n_holes=configs.cutout_nholes, ratio=configs.cutout_ratio, fill_value=configs.cutout_fill_value,
               p=configs.cutout_prob)
    ], p=1.)
    multiscale_training = configs.multiscale_training if configs.img_size > 4*32 else False
    train_dataset = KittiDataset(configs.dataset_dir,configs.input_cfg, mode='train', lidar_transforms=train_lidar_transforms,
                                 aug_transforms=train_aug_transforms, multiscale=multiscale_training,
                                 num_samples=configs.num_samples, mosaic=configs.mosaic,
                                 random_padding=configs.random_padding,split_first = split_first)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler,
                                  collate_fn=train_dataset.collate_fn)

    return train_dataloader, train_sampler


def create_val_dataloader(configs,split_first = True):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = KittiDataset(configs.dataset_dir, configs.input_cfg, mode='val', lidar_transforms=None, aug_transforms=None,
                               multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False,
                               split_first = split_first)
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler,
                                collate_fn=val_dataset.collate_fn)

    return val_dataloader


def create_test_dataloader(configs,split_first = True):
    """Create dataloader for testing phase"""

    test_dataset = KittiDataset(configs.dataset_dir, configs.input_cfg, mode='test', lidar_transforms=None, aug_transforms=None,
                                multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False,
                                split_first = split_first)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader


if __name__ == '__main__':
    import argparse
    import os

    import cv2
    import numpy as np
    from easydict import EasyDict as edict

    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from data_process.RL_dataloader import create_RL_dataloader
    from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target
    import config.kitti_config as cnf
    from models.model_utils import create_model
    from copy import deepcopy
    from utils.evaluation_utils import post_processing_v2

    parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')

    # parser.add_argument('--img_size', type=int, default=64,
    #                     help='the size of input image')
    parser.add_argument('--classnames-infor-path', type=str, default='../dataset/kitti/classes_names.txt',
                        metavar='PATH', help='The class names of objects in the task')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')
    parser.add_argument('--hflip_prob', type=float, default=0.,
                        help='The probability of horizontal flip')
    parser.add_argument('--cutout_prob', type=float, default=0.,
                        help='The probability of cutout augmentation')
    parser.add_argument('--cutout_nholes', type=int, default=1,
                        help='The number of cutout area')
    parser.add_argument('--cutout_ratio', type=float, default=0.3,
                        help='The max ratio of the cutout area')
    parser.add_argument('--cutout_fill_value', type=float, default=0.,
                        help='The fill value in the cut out area, default 0. (black)')
    parser.add_argument('--multiscale_training', action='store_true',
                        help='If true, use scaling data for training')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--mosaic', action='store_true',
                        help='If true, compose training samples as mosaics')
    parser.add_argument('--random-padding', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--show-train-data', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')
    parser.add_argument('--save_img', action='store_true',
                        help='If true, save the images')
    parser.add_argument('--detector_type',type = str, default = "coarse",
                        help="coarse or fine")
    parser.add_argument('--patch',action = "store_true",
                        help="if true, load patches as input, else use images")
    parser.add_argument('--policy_file', type = str, default = '../../RLsave/regnet_policies_3_bm.txt',
                        help='policy file path')

    configs = edict(vars(parser.parse_args()))
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')
    assert configs.detector_type in ["coarse","fine"]


    if configs.save_img:
        print('saving validation images')
        configs.saved_dir = os.path.join('imgs')
        if not os.path.isdir(configs.saved_dir):
            os.makedirs(configs.saved_dir)

    policy_mat = np.loadtxt(configs.policy_file)
    sample_start = min(policy_mat[:,0])
    configs.num_samples = int(sample_start)-6000

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')

    if not configs.patch:
        if configs.detector_type == "coarse":
            configs.input_cfg = LR_IMAGE_CFG
        elif configs.detector_type == "fine":
            configs.input_cfg = HR_IMAGE_CFG
    else:
        if configs.detector_type == "coarse":
            configs.input_cfg = LR_PATCH_CFG
        elif configs.detector_type == "fine":
            configs.input_cfg = HR_PATCH_CFG
    configs.img_size = configs.input_cfg.BEV_WIDTH

    if configs.show_train_data:
        dataloader, _ = create_train_dataloader(configs)
        print('len train dataloader: {}'.format(len(dataloader)))
    else:
        dataloader = create_val_dataloader(configs,split_first = False)
        print('len val dataloader: {}'.format(len(dataloader)))


    for batch_i, (img_files, imgs, targets) in enumerate(dataloader):
        if not (configs.mosaic and configs.show_train_data):
            img_file = img_files[0][0]
            patch_id = img_files[0][1]
            print(img_file,patch_id)
            img_rgb = cv2.imread(img_file)
            calib = kitti_data_utils.Calibration(img_file.replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = invert_target(targets[:, 1:], calib, img_rgb.shape, configs.input_cfg, patch_id, None)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, True)

        # Rescale target
        targets[:, 2:6] *= configs.img_size
        # Get yaw angle
        targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])

        img_bev = imgs.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))

        for c, x, y, w, l, yaw in targets[:, 1:7].numpy():
            # Draw rotated box
            bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])

        img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)

        if configs.mosaic and configs.show_train_data:
            if configs.save_img:
                fn = os.path.basename(img_file)
                cv2.imwrite(os.path.join(configs.saved_dir, fn), img_bev)
            else:
                cv2.imshow('mosaic_sample', img_bev)
        else:
            out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=configs.output_width)
            if configs.save_img:
                fn = os.path.basename(img_file).replace('.png','')
                mask = np.logical_and(policy_mat[:, 0] == float(fn), policy_mat[:,1] == patch_id)
                p = policy_mat[mask][0][2]
                save_path = os.path.join(configs.saved_dir, fn+'_'+str(patch_id)+'_'+str(p)+'.png')
                cv2.imwrite(save_path, out_img)
                cv2.imshow('single_sample', out_img)
                print('save image to %s' %save_path)
            else:
                cv2.imshow('single_sample', out_img)


        if cv2.waitKey(0) & 0xff == 27:
            break
