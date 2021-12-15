import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.kitti_dataset import KittiDataset
from config.input_config import HR_IMAGE_CFG, LR_IMAGE_CFG


class RL_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_lr, dataset_hr):
        self.dataset_lr = dataset_lr
        self.dataset_hr = dataset_hr
        assert len(self.dataset_lr) == len(self.dataset_hr)

    def __getitem__(self, index):
        path_lr, img_lr, target_lr = self.dataset_lr[index]
        path_hr, img_hr, target_hr = self.dataset_hr[index]
        assert path_lr == path_hr
        assert torch.equal(target_lr,target_hr)
        return path_lr, img_lr, img_hr, target_lr

    def __len__(self):
        return len(self.dataset_lr)

    def collate_fn(self,batch):
        paths,imgs_lr,imgs_hr,targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs_lr = torch.stack(imgs_lr)
        imgs_hr = torch.stack(imgs_hr)
        return paths,imgs_lr,imgs_hr,targets

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

#def create_patch(image,target)

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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 1)')

    configs = edict(vars(parser.parse_args()))
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')

    configs_lr = deepcopy(configs)
    configs_lr.input_cfg = LR_IMAGE_CFG
    configs_hr = deepcopy(configs)
    configs_hr.input_cfg = HR_IMAGE_CFG
    RL_dataloader,_ = create_RL_dataloader(configs_lr, configs_hr)

    for batch_id, (img_path, img_lr, img_hr, target) in enumerate(RL_dataloader):
        break

