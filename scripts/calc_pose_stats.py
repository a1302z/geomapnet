import set_paths
import os.path as osp
import argparse


"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

"""
script to calculate pose translation stats (run first for every dataset)
"""

# config
parser = argparse.ArgumentParser(
    description='Calculate pose translation stats')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'DeepLoc', 'RobotCar', 'AachenDayNight',
                                                   'CambridgeLandmarks', 'stylized_localization'),
                    help='Dataset')
parser.add_argument('--scene', type=str, default='', help='Scene name')
args = parser.parse_args()
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)

# dataset loader
# creating the dataset with train=True and real=False saves the stats from the
# training split
kwargs = dict(scene=args.scene, data_path=data_dir, train=True, real=False, seed=7, train_split=20)

if args.dataset in ['DeepLoc', 'AachenDayNight', 'CambridgeLandmarks']:
    kwargs['input_types'] = []
else:
    kwargs['skip_images'] = True

if args.dataset == '7Scenes':
    from dataset_loaders.seven_scenes import SevenScenes
    dset = SevenScenes(**kwargs)
elif args.dataset == 'DeepLoc':
    from dataset_loaders.deeploc import DeepLoc
    dset = DeepLoc(**kwargs)
elif args.dataset == 'RobotCar':
    from dataset_loaders.robotcar import RobotCar
    dset = RobotCar(**kwargs)
elif args.dataset == 'AachenDayNight':
    from dataset_loaders.aachen import AachenDayNight
    dset = AachenDayNight(**kwargs)
elif args.dataset == 'CambridgeLandmarks':
    from dataset_loaders.cambridge import Cambridge
    dset = Cambridge(**kwargs)
elif args.dataset == 'stylized_localization':
    from dataset_loaders.stylized_loader import StylizedCambridge
    dset = StylizedCambridge(**kwargs)
else:
    raise NotImplementedError

print('Done')
