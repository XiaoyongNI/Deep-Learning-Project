"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.08
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from config.input_config import LR_IMAGE_CFG, LR_PATCH_CFG, HR_PATCH_CFG
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    # parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4_lr.cfg', metavar='PATH',
    #                     help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--detector_type', type=str, default="coarse",
                        help="coarse or fine")
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--load_coarse', type=str, default='../Model_coarse_1_epoch_300.pth',
                        help="coarse checkpoint path")
    parser.add_argument('--load_fine', type=str, default='../Model_fine_1_epoch_300.pth',
                        help="fine checkpoint path")
    parser.add_argument('--policy_file', type = str, default = '../RLsave/regnet_policies_3_bm.txt',
                        help='policy file path')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')
    configs.input_cfg = LR_PATCH_CFG if configs.detector_type == "coarse" else HR_PATCH_CFG

    if configs.save_test_output:
        configs.saved_dir = os.path.join('imgs')
        make_folder(configs.saved_dir)

    return configs


if __name__ == '__main__':
    from copy import deepcopy
    from data_process.RL_dataloader import create_RL_dataloader
    configs = parse_test_configs()
    configs.distributed = False  # For testing
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    policy_mat = np.loadtxt(policy_file)
    sample_start = min(policy_mat[:, 0])
    configs.num_samples = int(sample_start) - 6000

    model_c = None
    model_f = None
    if configs.load_coarse is not None:
        configs_lr = deepcopy(configs)
        configs_lr.input_cfg = LR_PATCH_CFG
        configs_lr.detector_type = 'coarse'
        configs_lr.cfgfile = 'config/cfg/complex_yolov4_lr.cfg'
        configs_lr.img_size = configs_lr.input_cfg.BEV_WIDTH
        model_c = create_model(configs_lr)
        model_c.to(device)
        model_c.load_state_dict(torch.load(configs.load_coarse,map_location=device))
        model_c.eval()

    if configs.load_fine is not None:
        configs_hr = deepcopy(configs)
        configs_hr.input_cfg = HR_PATCH_CFG
        configs_hr.detector_type = 'fine'
        configs_hr.cfgfile = 'config/cfg/complex_yolov4_hr.cfg'
        configs_hr.img_size = configs_hr.input_cfg.BEV_WIDTH
        model_f = create_model(configs_hr)
        model_f.to(device)
        model_f.load_state_dict(torch.load(configs.load_fine, map_location=device))
        model_f.eval()

    out_cap = None

    test_dataloader,_ = create_RL_dataloader(configs_lr,configs_hr,split_first = False)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            img_paths, imgs_bev_lr, imgs_bev_hr, targets = batch_data
            patch_id = img_paths[0][1]
            input_imgs_lr = imgs_bev_lr.to(device).float()
            input_imgs_hr = imgs_bev_hr.to(device).float()
            t1 = time_synchronized()
            outputs_c = model_c(input_imgs_lr)
            outputs_f = model_f(input_imgs_hr)
            t2 = time_synchronized()
            detections_c = post_processing_v2(outputs_c, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            detections_f = post_processing_v2(outputs_f, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            img_detections_c = []
            img_detections_f = []# Stores detections for each image index
            img_detections_c.extend(detections_c)
            img_detections_f.extend(detections_f)

            img_bev_hr = imgs_bev_hr.squeeze() * 255
            img_bev_hr = img_bev_hr.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev_hr = cv2.resize(img_bev_hr, (configs_hr.img_size, configs_hr.img_size))

            for detections_lr,detections_hr in zip(img_detections_c,img_detections_f):
                if detections_lr is None and detections_hr is None:
                    continue
                # Rescale boxes to original image
                if detections_lr is not None:
                    detections_lr = rescale_boxes(detections_lr, configs_lr.img_size, img_bev_hr.shape[:2])
                    for x, y, w, l, im, re, *_, cls_pred in detections_lr:
                        yaw = np.arctan2(im, re)
                        # Draw rotated box
                        kitti_bev_utils.drawRotatedBox(img_bev_hr, x, y, w, l, yaw,configs_lr.input_cfg.color)

                if detections_hr is not None:
                    detections_hr = rescale_boxes(detections_hr, configs_hr.img_size, img_bev_hr.shape[:2])
                    for x, y, w, l, im, re, *_, cls_pred in detections_hr:
                        yaw = np.arctan2(im, re)
                        # Draw rotated box
                        kitti_bev_utils.drawRotatedBox(img_bev_hr, x, y, w, l, yaw, configs_hr.input_cfg.color)

            img_rgb = cv2.imread(img_paths[0][0])
            calib = kitti_data_utils.Calibration(img_paths[0][0].replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred_c = predictions_to_kitti_format(img_detections_c, calib, img_rgb.shape,
                                                       configs_lr.input_cfg,patch_id,configs_hr.img_size)
            objects_pred_f = predictions_to_kitti_format(img_detections_f, calib, img_rgb.shape,
                                                         configs_hr.input_cfg, patch_id, configs_hr.img_size)

            img_rgb = show_image_with_boxes(img_rgb, objects_pred_c, calib, False,color = configs_lr.input_cfg.color)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred_f, calib,False, color = configs_hr.input_cfg.color)

            img_bev_hr = cv2.flip(cv2.flip(img_bev_hr, 0), 1)

            out_img = merge_rgb_to_bev(img_rgb, img_bev_hr, output_width=608)

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))

            if configs.save_test_output:
                if configs.output_format == 'image':
                    fn = os.path.basename(img_paths[0][0]).replace('.png', '')
                    mask = np.logical_and(policy_mat[:, 0] == float(fn), policy_mat[:, 1] == patch_id)
                    p = policy_mat[mask][0][2]
                    save_path = os.path.join(configs.saved_dir, fn + '_' + str(patch_id) + '_' + str(p)+'.png')
                    cv2.imwrite(save_path, out_img)
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.saved_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            if configs.show_image:
                cv2.imshow('test-img', out_img)
                print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                if cv2.waitKey(0) & 0xFF == 27:
                    break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
