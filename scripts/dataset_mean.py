import set_paths
from common.train import safe_collate
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import numpy as np
import os.path as osp
import tqdm
"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

"""
Computes the mean and std of pixels in a dataset
"""

parser = argparse.ArgumentParser(description='Dataset images statistics')
parser.add_argument('--dataset', type=str, choices=('7Scenes',  'DeepLoc', 'RobotCar', 'AachenDayNight'
                                                   , 'CambridgeLandmarks', 'stylized_localization'),
                    help='Dataset', required=True)
parser.add_argument('--scene', type=str, default='', help='Scene name')
parser.add_argument('--augmentation', type=str, default='None', choices=['None', 'only', 'combined', 'stylized'], help='Use augmentation?')
args = parser.parse_args()

data_dir = osp.join('..', 'data', args.dataset)
crop_size_file = osp.join(data_dir, 'crop_size.txt')
crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(crop_size),
    transforms.ToTensor()])

# dataset loader
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=True, real=False,
              transform=data_transform, only_augmentation=args.augmentation=='only',
              night_augmentation=args.augmentation=='combined', use_stylization=args.augmentation=='stylized')
if args.dataset == '7Scenes':
    from dataset_loaders.seven_scenes import SevenScenes
    dset = SevenScenes(**kwargs)
elif args.dataset == 'DeepLoc':
    from dataset_loaders.deeploc import DeepLoc
    dset = DeepLoc(**kwargs)
elif args.dataset == 'AachenDayNight':
    from dataset_loaders.aachen import AachenDayNight
    dset = AachenDayNight(resize=None, **kwargs)
elif args.dataset == 'CambridgeLandmarks':
    from dataset_loaders.cambridge import Cambridge
    dset = Cambridge(**kwargs)
elif args.dataset == 'stylized_localization':
    from dataset_loaders.stylized_loader import StylizedCambridge
    dset = StylizedCambridge(styles = args.styles, **kwargs)
elif args.dataset == 'RobotCar':
    from dataset_loaders.robotcar import RobotCar
    dset = RobotCar(**kwargs)
else:
    raise NotImplementedError

# accumulate
batch_size = 90
num_workers = 8
loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers,
                    collate_fn=safe_collate)
acc = np.zeros((3, crop_size[0], crop_size[1]))
sq_acc = np.zeros((3, crop_size[0], crop_size[1]))
for batch_idx, (imgs, _) in tqdm.tqdm(enumerate(loader), total=len(loader)):
    imgs = imgs.numpy()
    acc += np.sum(imgs, axis=0)
    sq_acc += np.sum(imgs**2, axis=0)

    #if batch_idx % 50 == 0:
    #    print('Accumulated {:d} / {:d}'.format(
    #        batch_idx * batch_size, len(dset)))

N = len(dset) * acc.shape[1] * acc.shape[2]

mean_p = np.asarray([np.sum(acc[c]) for c in range(3)])
mean_p /= N
print('Mean pixel = ', mean_p)

# std = E[x^2] - E[x]^2
std_p = np.asarray([np.sum(sq_acc[c]) for c in range(3)])
std_p /= N
std_p -= (mean_p ** 2)
print('Std. pixel = ', std_p)

filename = 'stats'
#if args.styles >0:
#    filename = '{:s}_{:d}_styles'.format(filename, args.styles)
if args.augmentation == 'only':
    filename = '{:s}_only_aug'.format(filename)
elif args.augmentation == 'combined':
    filename = '{:s}_augm'.format(filename)
elif args.augmentation == 'stylized':
    filename = '{:s}_stylized'.format(filename)
output_filename = osp.join('..', 'data', 'deepslam_data', args.dataset, args.scene, '{:s}.txt'.format(filename))
np.savetxt(output_filename, np.vstack((mean_p, std_p)), fmt='%8.7f')
print('{:s} written'.format(output_filename))
