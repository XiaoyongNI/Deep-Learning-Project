import sys

import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.append('../')

from data_process.kitti_dataset import KittiDataset
from config.input_config import HR_IMAGE_CFG, LR_IMAGE_CFG, LR_PATCH_CFG, HR_PATCH_CFG


class RL_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_lr, dataset_hr):
        self.dataset_lr = dataset_lr
        self.dataset_hr = dataset_hr
        assert len(self.dataset_lr) == len(self.dataset_hr)

    def __getitem__(self, index):
        path_lr, img_lr, target_lr = self.dataset_lr[index]
        path_hr, img_hr, target_hr = self.dataset_hr[index]
        assert path_lr == path_hr
        assert torch.equal(target_lr, target_hr)
        return path_lr, img_lr, img_hr, target_lr

    def __len__(self):
        return len(self.dataset_lr)

    def collate_fn(self, batch):
        paths, imgs_lr, imgs_hr, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs_lr = torch.stack(imgs_lr)
        imgs_hr = torch.stack(imgs_hr)
        return paths, imgs_lr, imgs_hr, targets


def create_RL_dataloader(configs_lr, configs_hr):
    dataset_lr = KittiDataset(configs_lr.dataset_dir, configs_lr.input_cfg, mode='train', lidar_transforms=None,
                              aug_transforms=None, multiscale=False,
                              num_samples=configs_lr.num_samples, mosaic=False,
                              random_padding=False)
    dataset_hr = KittiDataset(configs_hr.dataset_dir, configs_hr.input_cfg, mode='train', lidar_transforms=None,
                              aug_transforms=None, multiscale=False,
                              num_samples=configs_lr.num_samples, mosaic=False,
                              random_padding=False)
    train_dataset = RL_dataset(dataset_lr, dataset_hr)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler,
                                  collate_fn=train_dataset.collate_fn)

    return train_dataloader, train_sampler


def create_patch(img_lr, img_hr, target):
    '''
    :param
    img_lr: (batch_size,c,h,w)
    img_hr: (batch_size,c,H,W)
    target: (num_targets, 8), target[i,0] is the index of image in this batch which i-th target corresponds to
    :return
    patch_lr: (4*batch_size,c,h,w), low resolution patches
    patch_hr: (4*batch_size,c,H,W), high resolution patches
    *** for patch tensors, the indices 4*i,4*i+1,4*i+2,4*i+3 of the first axis correspond to the patches
        divided from the ith image in this batch , i = 0,1,..,batch_size-1

    patch_target: (new_num_targets, 8)
    *** patch_target[i,0] is the index of patch which i-th target corresponds to (i = 0,1..,new_num_targets),
    namely patch_target[i,0] == j where patch_lr[j,:,:,:] and patch_hr[j,:,:,:] contain the i-th target.
    *** Note that there might be more targets (i.e. new_num_target > num_targets) due to overlap between two patches.
    '''
    batch_size = img_lr.size(0)
    patch_size = [LR_PATCH_CFG.BEV_WIDTH, HR_PATCH_CFG.BEV_WIDTH]
    img_size = [img_lr.size(2), img_hr.size(2)]
    patch_num = LR_PATCH_CFG.PATCH_NUM
    ## calcalate overlap in pixel: may be not accuarate for other patch numbers
    overlap = [int((patch_num * patch_size[i] - img_size[i]) / (patch_num - 1)) for i in range(2)]
    target[:, 2:6] *= img_lr.size(2) # convert to pixel position in low resolution image

    patch_lr = []
    patch_hr = []
    patch_target = []
    permute = np.array_split(np.arange(batch_size * patch_num ** 2), patch_num ** 2)
    permute = np.concatenate(list(zip(*(permute))))
    for i in range(patch_num):
        for j in range(patch_num):
            patch_id = i * patch_num + j
            pos_i = [patch_size[k] * i - overlap[k] * i for k in range(2)]
            pos_j = [patch_size[k] * j - overlap[k] * j for k in range(2)]
            p_lr = img_lr[:, :, pos_i[0]:pos_i[0] + patch_size[0], pos_j[0]:pos_j[0] + patch_size[0]]
            patch_lr.append(p_lr)
            p_hr = img_hr[:, :, pos_i[1]:pos_i[1] + patch_size[1], pos_j[1]:pos_j[1] + patch_size[1]]
            patch_hr.append(p_hr)

            map_X = (pos_i[0] <= target[:, 2]) * (target[:, 2] <= pos_i[0] + patch_size[0])
            map_Y = (pos_j[0] <= target[:, 3]) * (target[:, 3] <= pos_j[0] + patch_size[0])
            valid_map = map_X * map_Y
            p_target = target[valid_map]
            # change the box id to patch id which ranges from 0 to patch_num*batch_size-1
            p_target[:, 0] = p_target[:, 0] * patch_num ** 2 + patch_id
            # scale the bounding box coordinate to be in [0,1], which is the relative position in the patch
            p_target[:, 2] = (p_target[:, 2] - pos_i[0]) / patch_size[0]
            p_target[:, 3] = (p_target[:, 3] - pos_j[0]) / patch_size[0]
            p_target[:, 4:6] = p_target[:, 4:6] / patch_size[0]
            patch_target.append(p_target)

    patch_lr = torch.cat(patch_lr, 0)[permute]
    patch_hr = torch.cat(patch_hr, 0)[permute]
    patch_target = torch.cat(patch_target, 0)
    patch_target = patch_target[patch_target[:, 0].sort()[1]]
    return patch_lr, patch_hr, patch_target


if __name__ == "__main__":
    import argparse
    from easydict import EasyDict as edict
    import os
    from copy import deepcopy

    parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')

    # parser.add_argument('--img_size', type=int, default=64,
    #                     help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='mini-batch size (default: 1)')

    configs = edict(vars(parser.parse_args()))
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')

    configs_lr = deepcopy(configs)
    configs_lr.input_cfg = LR_IMAGE_CFG
    configs_hr = deepcopy(configs)
    configs_hr.input_cfg = HR_IMAGE_CFG
    RL_dataloader, _ = create_RL_dataloader(configs_lr, configs_hr)

    for batch_id, (img_path, img_lr, img_hr, target) in enumerate(RL_dataloader):
        patch_lr,patch_hr,patch_target = create_patch(img_lr, img_hr, target)
        break

