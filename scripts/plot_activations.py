"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import set_paths
from models.posenet import *
from common.train import load_state_dict
from dataset_loaders.composite import MF
from common.pose_utils import calc_vos_safe_fc, calc_vos_safe
import argparse
import os
import os.path as osp
import sys
import cv2
import tqdm
import numpy as np
import configparser
import torch.cuda
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, models
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# config
parser = argparse.ArgumentParser(description='Activation visualization script')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'DeepLoc', 'RobotCar', 'AachenDayNight'),
                    help='Dataset')
parser.add_argument('--scene', type=str, default='', help='Scene name')
parser.add_argument('--weights', type=str, help='trained weights to load')
#parser.add_argument('--config_file', type=str,help='configuration file used for training')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++', 'multitask', 'semanticOutput',
                                        'semanticV0', 'semanticV1','semanticV2', 'semanticV3', 'semanticV4'))
parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
parser.add_argument('--val', action='store_true', help='Use val split')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Output directory for video')
parser.add_argument('--no_video', action='store_false', help='Generate image instead of video. (Useful for unordered datasets such as AachenDayNight)', dest='generate_video')
parser.add_argument('--suffix', default='', type=str, help='Add suffix to figure name')
parser.add_argument('--num_images', type=int, default=5, help='Number of images generated if no video modus is active')
args = parser.parse_args()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

settings = configparser.ConfigParser()
config_file = os.path.join(os.path.dirname(args.weights), 'config.ini')
with open(config_file, 'r') as f:
    settings.read_file(f)
backbone_model = settings['training'].get('model', 'ResNet-34')
seed = settings.getint('training', 'seed')
section = settings['hyperparameters']
crop_size = section.getint('crop_size', 224)
crop_size = (crop_size, crop_size)
dropout = section.getfloat('dropout')
train_split = section.getint('train_split', 6)
sax = section.getfloat('beta_translation', 0.0)
saq = section.getfloat('beta')
if (args.model.find('mapnet') >= 0) or (args.model.find('semantic') >= 0) or (args.model.find('multitask') >= 0):
    steps = section.getint('steps')
    skip = section.getint('skip')
    real = section.getboolean('real')
    variable_skip = section.getboolean('variable_skip')
    srx = section.getfloat('gamma_translation', 0.0)
    srq = section.getfloat('gamma')
    sas = section.getfloat('sigma')
    fc_vos = args.dataset == 'RobotCar'

# modelfeature_extractor = models.resnet34(pretrained=False)
##WARNING: Changed dropout to be 0.0 
if backbone_model == 'ResNet-34':
    feature_extractor = models.resnet34(pretrained=False)
elif backbone_model == 'ResNet-101':
    feature_extractor = models.resnet101(pretrained=False)
elif backbone_model == 'ResNet-152':
    feature_extractor = models.resnet152(pretrained=False)
#elif backbone_model == 'ResNext-101':
#    feature_extractor = models.resnext101_32x8d(pretrained=False)
elif backbone_model == 'InceptionV3':
    feature_extractor = models.inception_v3(pretrained=False)
else:
    raise NotImplementedError('Required model not implemented yet')
posenet = PoseNet(feature_extractor, droprate=0.0, pretrained=False)
if (args.model.find('mapnet') >= 0) or args.model == 'semanticV0':
    model = MapNet(mapnet=posenet)
elif args.model == 'semanticV1':
    model = SemanticMapNet(mapnet=posenet)
elif args.model == 'semanticV2':
    feature_extractor_sem = models.resnet34(pretrained=False)
    posenet_sem = PoseNet(feature_extractor_sem, droprate=dropout, pretrained=False)
    model = SemanticMapNetV2(mapnet_images=posenet,mapnet_semantics=posenet_sem)
elif args.model == 'semanticV3':
    feature_extractor_2 = models.resnet34(pretrained=False)
    model = SemanticMapNetV3(feature_extractor_img=feature_extractor, feature_extractor_sem=feature_extractor_2, droprate=dropout, pretrained=False)
elif args.model == 'semanticV4':
    posenetv2 = PoseNetV2(feature_extractor, droprate=dropout, pretrained=False,filter_nans=(args.model == 'mapnet++'))
    model = MapNet(mapnet=posenetv2)
elif args.model == 'multitask':
    model = MultiTask(posenet=posenet, classes=10)
elif args.model == 'semanticOutput':
    model = SemanticOutput(posenet=posenet, classes=10)
else:
    model = posenet
model.eval()

# load weights
weights_filename = osp.expanduser(args.weights)
if osp.isfile(weights_filename):
    def loc_func(storage, loc): return storage
    checkpoint = torch.load(weights_filename, map_location=loc_func)
    load_state_dict(model, checkpoint['model_state_dict'])
    print('Loaded weights from {:s}'.format(weights_filename))
else:
    print('Could not load weights from {:s}'.format(weights_filename))
    sys.exit(-1)

data_dir = osp.join('..', 'data', args.dataset)
stats_file = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
resize = int(max(crop_size))

# transformer
data_transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=stats[1])])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
int_semantic_transform = transforms.Compose([
        transforms.Resize(resize,0), #Nearest interpolation
        transforms.Lambda(lambda pic: torch.from_numpy(np.array(pic, np.int64, copy=False)))
    ])
float_semantic_transform = transforms.Compose([
        transforms.Resize(resize,0), #Nearest interpolation
        transforms.ToTensor()
    ])
# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(data_dir, args.scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
# dataset
train = not args.val
if train:
    print('Visualizing TRAIN data')
else:
    print('Visualizing VAL data')
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=train,
              transform=data_transform, target_transform=target_transform, seed=seed)

#default
input_types = ['img'] if args.dataset != 'DeepLoc' else 'left'
output_types = ['pose']
        
if args.model == 'semanticV0':
    input_types = ['label_colorized']
elif args.model == 'semanticOutput':
    output_types = ['label']
elif args.model == 'semanticV4':
    input_types = ['left', 'label']
elif 'semantic' in args.model:
    input_types = ['left', 'label_colorized']
elif 'multitask' in args.model:
    output_types = ['pose', 'label']
if args.dataset in ['DeepLoc', ]:
        #print("Input types: %s\nOutput types: %s"%(input_types, output_types))
        
        semantic_transform = (
            float_semantic_transform 
            if 'semanticV' in args.model else 
            int_semantic_transform)
        
        kwargs = dict(kwargs,
                      semantic_transform=semantic_transform,
                      semantic_colorized_transform=float_semantic_transform,
                      input_types=input_types, 
                      output_types=output_types,
                      concatenate_inputs=True)
elif args.dataset in ['AachenDayNight', 'CambridgeLandmarks']:
    semantic_transform = (
            float_semantic_transform 
            if 'semanticV' in args.model else 
            int_semantic_transform)
        
    kwargs = dict(kwargs,
                      semantic_transform=semantic_transform,
                      #semantic_colorized_transform=float_semantic_transform,
                      input_types=input_types, 
                      output_types=output_types,
                      train_split=train_split,
                      #concatenate_inputs=True
                 )
    if args.dataset == 'AachenDayNight':
        kwargs['night_augmentation'] = None
        kwargs['resize'] = resize
if (args.model.find('mapnet') >= 0) or (args.model.find('semantic') >= 0) or (args.model.find('multitask') >= 0):
    vo_func = calc_vos_safe_fc if fc_vos else calc_vos_safe
    data_set = MF(dataset=args.dataset, steps=steps, skip=skip, real=real,
                  variable_skip=variable_skip, include_vos=False,
                  vo_func=vo_func, no_duplicates=False, **kwargs)
    L = len(data_set.dset)
if args.dataset == '7Scenes':
    from dataset_loaders.seven_scenes import SevenScenes
    data_set = SevenScenes(**kwargs)
elif args.dataset == 'DeepLoc':
    from dataset_loaders.deeploc import DeepLoc
    data_set = DeepLoc(**kwargs)
elif args.dataset == 'RobotCar':
    from dataset_loaders.robotcar import RobotCar
    data_set = RobotCar(**kwargs)
elif args.dataset == 'AachenDayNight':
    from dataset_loaders.aachen import AachenDayNight
    data_set = AachenDayNight(**kwargs)
    L = len(data_set)
else:
    raise NotImplementedError

# loader (batch_size MUST be 1)
loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=5,
                    pin_memory=True)

# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
    model.cuda()

# opencv init
if args.generate_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    model_name = 'posenet' if args.weights.find('posenet') >= 0 else 'vidvo'
    out_filename = osp.join(args.output_dir, '{:s}_{:s}_attention_{:s}.avi'.
                            format(args.dataset, args.scene, model_name))
    # get frame size
    img, _ = data_set[0]
    vwrite = cv2.VideoWriter(out_filename, fourcc=fourcc, fps=20.0,
                             frameSize=(img.size(2), img.size(1)))
    print('Initialized VideoWriter to {:s} with frames size {:d} x {:d}'.\
        format(out_filename, img.size(2), img.size(1)))
else:
    every = int(len(loader)/args.num_images)
    images = []
    
# inference
cm_jet = plt.cm.get_cmap('jet')
for batch_idx, (data, target) in tqdm.tqdm(enumerate(loader), total=len(loader)):
    if not args.generate_video and (batch_idx % every != 0):
        continue
    if CUDA:
        data = data.cuda()
    data_var = Variable(data.unsqueeze(0), requires_grad=True)

    model.zero_grad()
    pose = model(data_var)
    pose.mean().backward()

    act = data_var.grad.data.cpu().numpy()
    act = act.squeeze().transpose((1, 2, 0))
    img = data[0].cpu().numpy()
    img = img.transpose((1, 2, 0))

    # saliency map = data*gradient, comment next line if you want
    # saliency map = gradient
    act *= img
    act = np.amax(np.abs(act), axis=2)
    act -= act.min()
    act /= act.max()
    act = cm_jet(act)[:, :, :3]
    #act *= 255

    img *= stats[1]
    img += stats[0]
    #img *= 255
    img = img[:, :, ::-1]

    img = 0.5 * img + 0.5 * act
    img = np.clip(img, 0, resize)
    if args.generate_video:
        vwrite.write(img.astype(np.uint8))
    else:
        #img = img.astype(np.uint8)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[..., ::-1].copy()
        images.append(img)

    #if batch_idx % 200 == 0:
    #    print('{:d} / {:d}'.format(batch_idx, len(loader)))

if args.generate_video:
    vwrite.release()
    print('{:s} written'.format(out_filename))
else:
    print('Start generating final image')
    images = torch.stack([torch.from_numpy(img).transpose(0,2).transpose(1,2) for img in images])
    print(images.size())
    images = make_grid(images,nrow=int(np.sqrt(args.num_images)))
    output_path = os.path.join(args.output_dir, 'attention_{:s}{:s}.png'.format(args.model, args.suffix))
    save_image(images, output_path)
    print('saved to {:s}'.format(output_path))
    
    
