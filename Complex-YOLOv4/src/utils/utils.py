import os
import torch
import torchvision.transforms as transforms
import torchvision.models as torchmodels
import numpy as np
import shutil
import json
import pandas as pd

from utils import utils_detector
#from dataset.dataloader import CustomDatasetFromImages
from constants import base_dir_groundtruth, base_dir_detections_cd, base_dir_detections_fd, base_dir_metric_cd, base_dir_metric_fd
from constants import num_windows, img_size_fd, img_size_cd
from constants import iou_threshold

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def read_json(filename):
    with open(filename) as dt:
        data = json.load(dt)
    return data

def xywh2xyxy(x):
    y = np.zeros(x.shape)
    y[:,0] = x[:, 0] - x[:, 2] / 2.
    y[:,1] = x[:, 1] - x[:, 3] / 2.
    y[:,2] = x[:, 0] + x[:, 2] / 2.
    y[:,3] = x[:, 1] + x[:, 3] / 2.
    return y

def get_detected_boxes(policy, file_dirs, metrics, set_labels):
    for index, file_dir_st in enumerate(file_dirs):
        counter = 0
        for xind in range(num_windows):
            for yind in range(num_windows):
                # ---------------- Read Ground Truth ----------------------------------
                outputs_all = []
                gt_path = '{}/{}_{}_{}.txt'.format(base_dir_groundtruth, file_dir_st, xind, yind)
                if os.path.exists(gt_path):
                    gt = np.loadtxt(gt_path).reshape([-1, 5])
                    targets = np.hstack((np.zeros((gt.shape[0], 1)), gt))
                    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                    # ----------------- Read Detections -------------------------------
                    if policy[index, counter] == 1:
                        preds_dir = '{}/{}_{}_{}'.format(base_dir_detections_fd, file_dir_st, xind, yind)
                        targets[:, 2:] *= img_size_fd
                        if os.path.exists(preds_dir):
                            preds = np.loadtxt(preds_dir).reshape([-1,7])
                            outputs_all.append(torch.from_numpy(preds))
                    else:
                        preds_dir = '{}/{}_{}_{}'.format(base_dir_detections_cd, file_dir_st, xind, yind)
                        targets[:, 2:] *= img_size_cd
                        if os.path.exists(preds_dir):
                            preds = np.loadtxt(preds_dir).reshape([-1,7])
                            outputs_all.append(torch.from_numpy(preds))
                    set_labels += targets[:, 1].tolist()
                    metrics += utils_detector.get_batch_statistics(outputs_all, torch.from_numpy(targets), 0.5)
                else:
                    continue
                counter += 1

    return metrics, [[[[set_labels]]]]

cudaFLAG = True

def read_offsets(img_paths, num_actions):
    #iou_threshold = 0.5
    if cudaFLAG:
        offset_fd = torch.zeros((len(img_paths), num_actions)).cuda()
        offset_cd = torch.zeros((len(img_paths), num_actions)).cuda()
    else:
        offset_fd = torch.zeros((len(img_paths), num_actions))
        offset_cd = torch.zeros((len(img_paths), num_actions))
    for index, img_path in enumerate(img_paths):
        # get img_id from img_path, eg img_id = '002096.npy'
        img_id = int(img_path[0].split('image_2/')[1].replace('.png', ''))
        # read detector and ground truth
        df_val_id = pd.read_csv(base_dir_metric_fd+'val_id_16.txt', sep=" ", header=None)
        df_coarse = pd.read_csv(base_dir_metric_fd+'coarse_detector_boxid.txt', sep=" ", header=None)
        df_fine = pd.read_csv(base_dir_metric_fd+'fine_detector_boxid.txt', sep=" ", header=None)
        df_gt = pd.read_csv(base_dir_metric_fd+'ground_truth_16.txt', sep=" ", header=None)
        # find index(row number: patch id) of image
        select_indices = list(np.where(df_val_id[0] == img_id)[0])
        # empty lists
        patch_ids = []
        patch_coarse_iou = []
        patch_coarse_score = []
        patch_fine_iou = []
        patch_fine_score = []
        # line numbers
        select_indices = list(np.where(df_val_id[0] == img_id)[0])
        for id_patch in select_indices:
            # patch id of image
            patch_id = int(df_val_id.iloc[id_patch][1])
            patch_ids.append(patch_id)
            # corresponding predictions of coarse/fine detectors
            # row number
            coarse_index = list(np.where(df_coarse[0] == id_patch)[0])
            fine_index = list(np.where(df_fine[0] == id_patch)[0])
            gt_index = list(np.where(df_gt[0] == id_patch)[0])
            # ground truth conparation
            # create two mini dataframe of a patch
            df_coarse_patch = df_coarse.iloc[coarse_index]
            df_fine_patch = df_fine.iloc[fine_index]
            # iou value
            iou_patch_coarse = 0
            iou_patch_fine = 0
            # all ground truth
            for gt_id in gt_index:
                ## bounding_box_id of ground truth
                bounding_box_id = df_gt.iloc[gt_id][1]
                ## class of ground truth
                gt_class = df_gt.iloc[gt_id][2]
                # row number in mini dataframe of a bounding box
                coarse_patch_id = list(np.where(df_coarse_patch[5] == bounding_box_id)[0])
                fine_patch_id = list(np.where(df_fine_patch[5] == bounding_box_id)[0])
                # coarse
                if coarse_patch_id:
                    # mini data frame corresponding to bounding box
                    df_coarse_bounding_box = df_coarse_patch.iloc[coarse_patch_id]
                    # row number with max iou value
                    coarse_bounding_box_id = df_coarse_bounding_box.iloc[:,2].argmax()
                    # add iou value
                    iou_patch_coarse += df_coarse_bounding_box.iloc[coarse_bounding_box_id][2]
                else:
                    iou_patch_coarse += 0
                # fine
                if fine_patch_id:
                    # mini data frame corresponding to bounding box
                    df_fine_bounding_box = df_fine_patch.iloc[fine_patch_id]
                    # row number with max iou value
                    fine_bounding_box_id = df_fine_bounding_box.iloc[:,2].argmax()
                    # add iou value
                    iou_patch_fine += df_fine_bounding_box.iloc[fine_bounding_box_id][2]
                else:
                    iou_patch_fine += 0
            # iou value to bounding box number
            iou_patch_coarse = iou_patch_coarse / len(gt_index)
            iou_patch_fine = iou_patch_fine / len(gt_index)
            #print(iou_patch_coarse)
            # iou value
            if coarse_index:
                #patch_coarse_iou.append(np.mean(df_coarse.iloc[coarse_index][2]))
                patch_coarse_iou.append(iou_patch_coarse)
            else:
                patch_coarse_iou.append(0.0)
            if fine_index:
                #patch_fine_iou.append(np.mean(df_fine.iloc[fine_index][2]))
                patch_fine_iou.append(iou_patch_fine)
            else:
                patch_fine_iou.append(0.0)
            # # score value
            # patch_coarse_score.append(np.mean(df_coarse.iloc[coarse_index][3]))
            # patch_fine_score.append(np.mean(df_fine.iloc[fine_index][3]))
        # write offset_fd & offset_cd
        for i in range(0,len(patch_ids)):
            offset_fd[index,patch_ids[i]] = torch.from_numpy(np.array(patch_fine_iou[i]))
            offset_cd[index,patch_ids[i]] = torch.from_numpy(np.array(patch_coarse_iou[i]))
        #data = pd.read_csv(base_dir_metric_fd+'val_id.txt', sep=" ", header=None)
        #offset_fd[index, :] = torch.from_numpy(np.loadtxt('{}/{}'.format(base_dir_metric_fd, img_id)).flatten())
        #offset_cd[index, :] = torch.from_numpy(np.loadtxt('{}/{}'.format(base_dir_metric_cd, img_id)).flatten())

    return offset_fd, offset_cd

def performance_stats(policies, rewards):
    # Print the performace metrics including the average reward, average number
    # and variance of sampled num_patches, and number of unique policies
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return reward, num_unique_policy, variance, policy_set

def compute_reward(offset_fd, offset_cd, policy, beta, sigma):
    """
    Args:
        offset_fd: np.array, shape [batch_size, num_actions]
        offset_cd: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    offset_cd += beta
    # R_acc
    # reward_patch_diff = (offset_fd - offset_cd)*policy + -1*((offset_fd - offset_cd)*(1-policy))
    reward_patch_diff = (offset_fd - offset_cd)*policy 
    # R_cost
    reward_patch_acqcost = (policy.size(1) - policy.sum(dim=1)) / policy.size(1)
    # R_c
    reward_img = reward_patch_diff.sum(dim=1) + sigma * reward_patch_acqcost
    reward = reward_img.unsqueeze(1)
    return reward.float()

def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Scale(img_size),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Scale(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test

#def get_dataset(img_size, root='data/'):
#    transform_train, transform_test = get_transforms(img_size)
#    trainset = CustomDatasetFromImages(root+'train.csv', transform_train)
#    testset = CustomDatasetFromImages(root+'val.csv', transform_test)
#
#    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(num_output):
    agent = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)

    return agent
