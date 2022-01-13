import argparse
import os
import time
import numpy as np
import sys
import warnings
import time


warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.utils.data.distributed
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path.append('./')
from config.input_config import LR_IMAGE_CFG, LR_PATCH_CFG, HR_PATCH_CFG
from data_process.kitti_dataloader import create_val_dataloader
from models.model_utils import create_model
from utils.misc import AverageMeter, ProgressMeter
from utils.evaluation_utils import post_processing, get_batch_statistics_rotated_bbox, ap_per_class, load_classes, post_processing_v2, get_batch_statistics_iou


def evaluate_mAP(val_loader, model, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, w, l, im, re))
            targets[:, 2:6] *= configs.img_size
            imgs = imgs.to(configs.device, non_blocking=True)
            val_loader
            this_start = time.time()
            outputs = model(imgs)
            this_end = time.time()
            outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            temp_metric = get_batch_statistics_iou(outputs, targets, iou_threshold=configs.iou_thresh)
            gt_len, _ = np.shape(targets)
            gt_array = np.zeros((gt_len,3))
            for temp_ii in range(gt_len):
                gt_array[temp_ii,0] = batch_idx
                gt_array[temp_ii,1] = temp_ii
                gt_array[temp_ii,2] = targets[temp_ii,1]


            with open("coarse_gt.txt", "ab") as of:
                np.savetxt(of, gt_array)
            if len(temp_metric)>0:
                this_iou =  temp_metric[0][0]
                this_score =  temp_metric[0][1]
                this_class = temp_metric[0][2]
                this_len = len(this_class)
                out_numpy = np.zeros((this_len,6))

                for temp_i in range(this_len):
                    out_numpy[temp_i,0] = batch_idx
                    out_numpy[temp_i,1] = temp_i
                    out_numpy[temp_i,2] = this_iou[temp_i]
                    out_numpy[temp_i,3] = this_score[temp_i]
                    out_numpy[temp_i,4] = this_class[temp_i]
                    out_numpy[temp_i,5] = this_end - this_start



                with open("coarse_eval.txt", "ab") as f:
                    np.savetxt(f, out_numpy)


            sample_metrics += temp_metric 

            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def parse_eval_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--classnames-infor-path', type=str, default='../dataset/kitti/classes_names.txt',
                        metavar='PATH', help='The class names of objects in the task')
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
    # parser.add_argument('--img_size', type=int, default=64,
    #                     help='the size of input image')
    parser.add_argument('--detector_type', type=str, default="coarse",
                        help="coarse or fine")
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for IoU')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')
    assert configs.detector_type in ["coarse", "fine"]
    configs.input_cfg = LR_PATCH_CFG if configs.detector_type == "coarse" else HR_PATCH_CFG
    configs.cfgfile = './config/cfg/complex_yolov4_lr.cfg' \
        if configs.detector_type == "coarse" else './config/cfg/complex_yolov4_hr.cfg'
    configs.img_size = configs.input_cfg.BEV_WIDTH

    return configs


if __name__ == '__main__':
    configs = parse_eval_configs()
    configs.distributed = False  # For evaluation
    class_names = load_classes(configs.classnames_infor_path)

    model = create_model(configs)
    # model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    print(configs.device)
    model.load_state_dict(torch.load(configs.pretrained_path,
                                     map_location = configs.device if configs.no_cuda else None))


    model = model.to(device=configs.device)

    model.eval()
    print('Create the validation dataloader')
    val_dataloader = create_val_dataloader(configs)

    print("\nStart computing mAP...\n")
    precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
    print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))

    print("\nmAP: {}\n".format(AP.mean()))
