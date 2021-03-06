import set_paths

from torchvision import transforms
from torchvision import models as pretrainedmodels
from torch import nn
import torch
import json
import configparser
import argparse
import numpy as np
import os.path as osp
from dataset_loaders.composite import MF, MFOnline
from models.posenet import *
from common.criterion import PoseNetCriterion, MapNetCriterion,\
    MapNetOnlineCriterion, UncertainyCriterion, SemanticCriterion
from common.optimizer import Optimizer
from common.train import Trainer
import random

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

"""
Main training script for MapNet
"""

parser = argparse.ArgumentParser(description='Training script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'DeepLoc', 'RobotCar', 'AachenDayNight', 'CambridgeLandmarks', 'stylized_localization'),
                    help='Dataset')
parser.add_argument('--scene', type=str, default='', help='Scene name')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++', 'multitask', 'multitask3d', 'semanticOutput', 'semanticV0', 'semanticV1','semanticV2', 'semanticV3', 'semanticV4'),
                    help='Model to train')
parser.add_argument('--device', type=str, default=None,
                    help='value to be set to $CUDA_VISIBLE_DEVICES')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from',
                    default=None)
parser.add_argument('--learn_beta', action='store_true',
                    help='Learn the weight of translation loss')
parser.add_argument('--learn_gamma', action='store_true',
                    help='Learn the weight of rotation loss')
parser.add_argument('--learn_sigma', action='store_true',
                    help='Learn the relative weight of pose and semantics')
parser.add_argument('--resume_optim', action='store_true',
                    help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--suffix', type=str, default='',
                    help='Experiment name suffix (as is)')
parser.add_argument('--overfit', type=int, default=None, help='Reduce dataset to overfit few examples')
parser.add_argument('--uncertainty_criterion', action='store_true', help='Use criterion which weighs losses with uncertainty')
parser.add_argument('--learn_direct_sigma', action='store_true', help='Learn sigma directly instead of log of sigma')
parser.add_argument('--init_seed', type=int, default=0, help='Set seed for random initialization of model')
parser.add_argument('--server', type=str, default='http://localhost', help='Set visdom server address')
parser.add_argument('--port', type=int, default=8097, help='set visdom port')
#parser.add_argument('--crop_size_file', type=str, default='crop_size.txt', help='Specify crop size file')
parser.add_argument('--use_augmentation', type=str, default=None, choices=['combined', 'only'], help='Use augmented images. Needs to be supported by dataloader (currently only AachenDayNight)')
#parser.add_argument('--only_augmentation', action='store_true', help='Use only augmented images. Not in combination with use augmentation option!')
parser.add_argument('--use_stylization', default=None, type=int, help='Use stylized images as augmentation. Argument is number of used styles per database image. All styles are used equally.')
parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic images for training')
#parser.add_argument('--styles', type=int, default=0, help='Only for stylized dataset')

args = parser.parse_args()
print(args)

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
    settings.read_file(f)
section = settings['optimization']
optim_config = {k: json.loads(v) for k, v in list(section.items()) if k not in ['opt', 'loss_fn']}
opt_method = section['opt']
lr = optim_config.pop('lr')
weight_decay = optim_config.pop('weight_decay')
loss_fn_config = section.get('loss_fn', 'l1').lower()


data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
filename = 'stats'
"""if 'args.only_augmentation':
    filename = '{:s}_only_aug'.format(filename)
elif args.use_augmentation:
    filename = '{:s}_augm'.format(filename)
elif args.use_stylization:
    filename = '{:s}_stylized'.format(filename)
"""
stats_file = osp.join(data_dir, args.scene, '{:s}.txt'.format(filename))
print('Using {} as stats file'.format(stats_file))
stats = np.loadtxt(stats_file)
#crop_size_file = osp.join('..', 'data', args.dataset, args.crop_size_file)
#crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))

section = settings['hyperparameters']
crop_size = section.getint('crop_size', 224)
crop_size = (crop_size, crop_size)
freeze = section.getboolean('freeze_feature_extraction', False)
activation_function = section.get('activation_function', 'relu').lower()
feature_dim = section.getint('feature_dim', 2048)
base_poses = section.get('base_poses', 'None')
dropout = section.getfloat('dropout')
color_jitter = section.getfloat('color_jitter', 0)
sax = section.getfloat('beta_translation', 0.0)
saq = section.getfloat('beta')
train_split = section.getint('train_split', 6)
if args.model.find('mapnet') >= 0 or args.model.find('semantic') >= 0 or args.model.find('multitask') >= 0:
    skip = section.getint('skip')
    real = section.getboolean('real')
    variable_skip = section.getboolean('variable_skip')
    srx = section.getfloat('gamma_translation', 0.0)
    srq = section.getfloat('gamma')
    sas = section.getfloat('sigma')
    steps = section.getint('steps')
    
if args.model.find('++') >= 0:
    vo_lib = section.get('vo_lib', 'orbslam')
    print('Using {:s} VO'.format(vo_lib))

section = settings['training']
backbone_model = section.get('model', 'ResNet-34')
seed = section.getint('seed')
optim_config['epochs'] = section.getint('n_epochs')


det_seed = args.init_seed
if det_seed >= 0:
    random.seed(det_seed)
    torch.manual_seed(det_seed)
    np.random.seed(det_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#  --------- Model ----------
# activation function
af = torch.nn.functional.relu
if activation_function == 'sigmoid':
    af = torch.nn.functional.sigmoid
#base poses
set_base_poses = None
if base_poses in ['unit_vectors', 'unit']:
    feature_dim = 6
    set_base_poses = np.array([[1,0,0], [0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float)
    set_base_poses = torch.from_numpy(set_base_poses.T).float()
    print('Base poses set to unit vectors')
elif base_poses == 'gaussian':
    set_base_poses = np.random.normal(size=(3, feature_dim))
    set_base_poses = torch.from_numpy(set_base_poses).float()
    print('Base poses sampled from gaussian')
else:
    print('Standard initialization for base poses')
if backbone_model == 'ResNet-34':
    feature_extractor = pretrainedmodels.resnet34(pretrained=True)
elif backbone_model == 'ResNet-101':
    feature_extractor = pretrainedmodels.resnet101(pretrained=True)
elif backbone_model == 'ResNet-152':
    feature_extractor = pretrainedmodels.resnet152(pretrained=True)
#elif backbone_model == 'ResNext-101':
#    feature_extractor = pretrainedmodels.resnext101_32x8d(pretrained=True)
elif backbone_model == 'InceptionV3':
    feature_extractor = pretrainedmodels.inception_v3(pretrained=True)
else:
    raise NotImplementedError('Required model not implemented yet')
print('Use {:s} as backbone'.format(backbone_model))
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=True,
                  filter_nans=(args.model == 'mapnet++'), feat_dim=feature_dim, 
                 freeze_feature_extraction = freeze, activation_function=af, 
                 set_base_poses=set_base_poses)
if args.model in ['multitask', 'semanticOutput', 'multitask3d']:
    classes = None
    input_size = None
    if args.dataset == 'DeepLoc':
        classes = 10
        input_size = crop_size #(256, 455)
    elif args.dataset == 'AachenDayNight':
        classes = 65
        input_size = crop_size #(224,224)
    else:
        raise NotImplementedError('Classes for dataset not specified')
if args.model == 'posenet':
    model = posenet
elif 'mapnet' in args.model or args.model == 'semanticV0':
    model = MapNet(mapnet=posenet)
elif args.model == 'semanticV1':
    model = SemanticMapNet(mapnet=posenet)
elif args.model == 'semanticV2':
    feature_extractor_sem = pretrainedmodels.resnet34(pretrained=True)
    posenet_sem = PoseNet(feature_extractor_sem, droprate=dropout, pretrained=True,
                      filter_nans=(args.model=='mapnet++'))
    model = SemanticMapNetV2(mapnet_images=posenet, mapnet_semantics=posenet_sem)
elif args.model == 'semanticV3':
    feature_extractor_2 = pretrainedmodels.resnet34(pretrained=True)
    model = SemanticMapNetV3(feature_extractor_img=feature_extractor, feature_extractor_sem=feature_extractor_2, droprate=dropout, pretrained=True,
        feat_dim=2048, filter_nans=(args.model=='mapnet++'))
elif args.model == 'semanticV4':
    posenetv2 = PoseNetV2(feature_extractor, droprate=dropout, pretrained=True,
                  filter_nans=(args.model == 'mapnet++'))
    model = MapNet(mapnet=posenetv2)
elif args.model == 'multitask':
    model = MultiTask(posenet=posenet, classes=classes, input_size=input_size,feat_dim=feature_dim, 
                     freeze_feature_extraction=freeze, set_base_poses=set_base_poses)
elif args.model == 'multitask3d':
    model = MultiTask3D(posenet=posenet, classes=classes, input_size=input_size,feat_dim=feature_dim, 
                     freeze_feature_extraction=freeze, set_base_poses=set_base_poses)
elif args.model == 'semanticOutput':
    model = SemanticOutput(posenet=posenet, classes=classes, input_size=input_size,feat_dim=feature_dim)
else:
    raise NotImplementedError

# loss function
loss_fn = nn.L1Loss() #default regression loss
if loss_fn_config in ['l2', 'mse']:
    loss_fn = nn.MSELoss()
    print('Using MSE Loss')
elif loss_fn_config in ['smoothl1', 'huber']:
    loss_fn = nn.SmoothL1Loss()
    print('Using Smooth L1 / Huber Loss')
if args.model == 'posenet':
    train_criterion = PoseNetCriterion(t_loss_fn=loss_fn, q_loss_fn=loss_fn,
        sax=sax, saq=saq, learn_beta=args.learn_beta)
    val_criterion = PoseNetCriterion()
elif args.model == 'semanticOutput':
    train_criterion = SemanticCriterion()
    val_criterion = SemanticCriterion()
elif 'mapnet' in args.model or 'semantic' in args.model or 'multitask' in args.model:
    kwargs = dict(t_loss_fn=loss_fn, q_loss_fn=loss_fn,sax=sax, saq=saq, srx=srx, srq=srq,
                  learn_beta=args.learn_beta, learn_gamma=args.learn_gamma)
    
    if 'multitask' in args.model:
        kwargs = dict(kwargs, dual_target=True, 
                      sas=sas, learn_sigma=args.learn_sigma)
        
    if '++' in args.model:
        kwargs = dict(kwargs, gps_mode=(vo_lib == 'gps'))
        train_criterion = MapNetOnlineCriterion(**kwargs)
        val_criterion = MapNetOnlineCriterion()
    elif args.uncertainty_criterion:
        train_criterion = UncertainyCriterion(**kwargs, learn_log=not args.learn_direct_sigma)
        val_criterion = UncertainyCriterion(dual_target='multitask' in args.model, learn_log=not args.learn_direct_sigma)
    else:
        train_criterion = MapNetCriterion(**kwargs)
        val_criterion = MapNetCriterion(dual_target='multitask' in args.model)
else:
    raise NotImplementedError

# optimizer
param_list = [{'params': model.parameters()}]
if args.learn_beta and hasattr(train_criterion, 'sax') and \
        hasattr(train_criterion, 'saq'):
    param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if args.learn_gamma and hasattr(train_criterion, 'srx') and \
        hasattr(train_criterion, 'srq'):
    param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
if args.learn_sigma and hasattr(train_criterion, 'sas'):
    param_list.append({'params': [train_criterion.sas]})
optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
                      weight_decay=weight_decay, **optim_config)


# transformers
resize = int(max(crop_size))
tforms = [transforms.Resize(resize)]
print('Cropping and resizing to %s'%str(crop_size))
#if args.dataset == 'AachenDayNight':
tforms.append(transforms.CenterCrop(crop_size))
if color_jitter > 0:
    assert color_jitter <= 1.0
    print('Using ColorJitter data augmentation')
    tforms.append(transforms.ColorJitter(brightness=color_jitter,
                                         contrast=color_jitter, saturation=color_jitter, hue=0.5))
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

int_semantic_transform = transforms.Compose([
        transforms.Resize(resize,0), #Nearest interpolation
        transforms.CenterCrop(crop_size),
        transforms.Lambda(lambda pic: torch.from_numpy(np.array(pic, np.int64, copy=False)))
    ])
float_semantic_transform = transforms.Compose([
        transforms.Resize(resize,0), #Nearest interpolation
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])

# datasets
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, transform=data_transform,
              target_transform=target_transform, seed=seed)
#default
input_types = ['img']
output_types = ['pose']
if args.model == 'semanticV0':
    input_types = ['label_colorized']
elif args.model == 'semanticV4':
    input_types = ['img', 'label']
elif args.model == 'semanticOutput':
    output_types = 'label'
elif 'semantic' in args.model:
    input_types = ['img', 'label_colorized']
elif 'multitask' in args.model:
    output_types = ['pose', 'label']
if 'multitask3d' == args.model:
    assert args.dataset == 'AachenDayNight', 'Multitask3d only for AachenDayNight dataset available so far'
    input_types = ['img', 'point_cloud']
semantic_transform = (
            float_semantic_transform 
            if 'semanticV' in args.model else 
            int_semantic_transform)
if args.model == 'posenet':
    if args.dataset == '7Scenes':
        from dataset_loaders.seven_scenes import SevenScenes
        train_set = SevenScenes(train=True, **kwargs)
        val_set = SevenScenes(train=False, **kwargs)
    elif args.dataset == 'DeepLoc':
        

        
        semantic_transform = (
            float_semantic_transform 
            if 'semanticV' in args.model else 
            int_semantic_transform)
        
        kwargs = dict(kwargs,
                      overfit=args.overfit,
                      semantic_transform=semantic_transform,
                      semantic_colorized_transform=float_semantic_transform,
                      input_types=input_types, 
                      output_types=output_types,
                      concatenate_inputs=True)

        from dataset_loaders.deeploc import DeepLoc
        train_set = DeepLoc(train=True, **kwargs)
        val_set = DeepLoc(train=False, **kwargs)
    elif args.dataset == 'AachenDayNight':
        """
        data_path, train, train_split=0.7,    
                input_types='image', output_types='pose', real=False
        """
        kwargs = dict(kwargs, resize=resize,
                      overfit=args.overfit,
                      semantic_transform=semantic_transform,
                      #semantic_colorized_transform=float_semantic_transform,
                      input_types=input_types, 
                      output_types=output_types,
                      train_split=train_split,
                      #concatenate_inputs=True
                      night_augmentation=args.use_augmentation,
                      use_stylization = args.use_stylization, 
                      use_synthetic = args.use_synthetic
                     )
        from dataset_loaders.aachen import AachenDayNight
        train_set = AachenDayNight(train=True, **kwargs)
        val_set = AachenDayNight(train=False, **kwargs)
    elif args.dataset == 'CambridgeLandmarks':
        """
        data_path, train, train_split=0.7,    
                input_types='image', output_types='pose', real=False
        """
        kwargs = dict(kwargs,
                      overfit=args.overfit,
                      semantic_transform=semantic_transform,
                      #semantic_colorized_transform=float_semantic_transform,
                      input_types=input_types, 
                      output_types=output_types,
                      #concatenate_inputs=True
                     )
        from dataset_loaders.cambridge import Cambridge
        train_set = Cambridge(train=True, **kwargs)
        val_set = Cambridge(train=False, **kwargs)
    elif args.dataset == 'stylized_localization':
        kwargs = dict(kwargs,
                      overfit=args.overfit,
                      scene = args.scene,
                      styles = args.styles
                     )
        from dataset_loaders.stylized_loader import StylizedCambridge
        train_set = StylizedCambridge(train=True, **kwargs)
        val_set = StylizedCambridge(train=False, **kwargs)
    elif args.dataset == 'RobotCar':
        from dataset_loaders.robotcar import RobotCar
        train_set = RobotCar(train=True, **kwargs)
        val_set = RobotCar(train=False, **kwargs)
    else:
        raise NotImplementedError
elif 'mapnet' in args.model or 'semantic' in args.model or 'multitask' in args.model:
    kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
                  variable_skip=variable_skip)
    if args.dataset == 'DeepLoc':
                
        kwargs = dict(kwargs,
                      overfit=args.overfit,
                      semantic_transform=semantic_transform,
                      semantic_colorized_transform=float_semantic_transform,
                      input_types=input_types, 
                      output_types=output_types,
                      concatenate_inputs=True)
        
    elif args.dataset in ['AachenDayNight', 'CambridgeLandmarks']:
        kwargs = dict(kwargs, 
                      overfit=args.overfit,
                      semantic_transform=semantic_transform,
                      #semantic_colorized_transform=float_semantic_transform,
                      input_types=input_types, 
                      output_types=output_types,
                      train_split=train_split,
                      #concatenate_inputs=True
                     )
        if args.dataset == 'AachenDayNight':
            kwargs['resize'] = resize
            kwargs['night_augmentation']=args.use_augmentation
            kwargs['use_stylization'] = args.use_stylization
            kwargs['use_synthetic'] = args.use_synthetic
            kwargs['verbose'] = False
    elif args.dataset == 'stylized_localization':
        kwargs = dict(kwargs,
                      overfit=args.overfit,
                      scene=args.scene,
                      styles=args.styles
                     )
        
    if '++' in args.model:
        train_set = MFOnline(
            vo_lib=vo_lib, gps_mode=(
                vo_lib == 'gps'), **kwargs)
        val_set = None
    else:
        train_set = MF(train=True, real=real, **kwargs)
        val_set = MF(train=False, real=real, **kwargs)
else:
    raise NotImplementedError


print('Loaded {:d} images'.format(len(train_set)))
    
# trainer
config_name = args.config_file.split('/')[-1]
config_name = config_name.split('.')[0]
experiment_name = '{:s}_{:s}_{:s}_{:s}'.format(args.dataset, args.scene,
                                               args.model, config_name)
if args.learn_beta:
    experiment_name = '{:s}_learn_beta'.format(experiment_name)
if args.learn_gamma:
    experiment_name = '{:s}_learn_gamma'.format(experiment_name)
if args.learn_sigma:
    experiment_name = '{:s}_learn_sigma'.format(experiment_name)
if args.uncertainty_criterion:
    experiment_name = '{:s}_uncertainty_criterion'.format(experiment_name)
if args.use_augmentation == 'combined':
    experiment_name = '{:s}_augmented'.format(experiment_name)
elif args.use_augmentation == 'only':
    experiment_name = '{:s}_only_augmented'.format(experiment_name)
if args.use_stylization:
    experiment_name = '{:s}_stylized'.format(experiment_name)
    if args.use_stylization > 1:
        experiment_name = '{:s}_{:d}_styles'.format(experiment_name, args.use_stylization)
if args.use_synthetic:
    experiment_name = '{:s}_synthetic'.format(experiment_name)
if det_seed >= 0:
    experiment_name = '{:s}_seed{}'.format(experiment_name, det_seed) 
#if args.styles > 0:
#    experiment_name = '{:s}_{}_styles'.format(experiment_name, args.styles)
experiment_name += args.suffix
trainer = Trainer(model, optimizer, train_criterion, args.config_file,
                  experiment_name, train_set, val_set, device=args.device,
                  checkpoint_file=args.checkpoint, visdom_server = args.server, visdom_port = args.port,
                  resume_optim=args.resume_optim, val_criterion=val_criterion)
lstm = args.model == 'vidloc'
trainer.train_val(lstm=lstm, dual_target='multitask' in args.model)


