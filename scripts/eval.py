from __future__ import division
import pickle
from torchvision import transforms, models
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import torch.cuda
import configparser
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import set_paths
from models.posenet import *
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import optimize_poses, quaternion_angular_error, qexp,\
    calc_vos_safe_fc, calc_vos_safe

from common.vis_utils import show_images, sem_labels_to_rgb, normalize, one_hot_to_one_channel, sem_label_to_name
from common.stolen_utils import *
from dataset_loaders.composite import MF
import argparse
import os
import os.path as osp
import sys
import numpy as np
import matplotlib
DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')

# config
parser = argparse.ArgumentParser(description='Evaluation script for pose regression networks')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'DeepLoc', 'RobotCar', 'AachenDayNight', 
                                                   'CambridgeLandmarks'),
                    help='Dataset')
parser.add_argument('--scene', type=str, default='', help='Scene name')
parser.add_argument('--weights', type=str, help='trained weights to load')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++', 'multitask', 'semanticOutput',
                                        'semanticV0', 'semanticV1','semanticV2', 'semanticV3', 'semanticV4'),
                    help='Model to use (mapnet includes both MapNet and MapNet++ since their'
                    'evluation process is the same and they only differ in the input weights'
                    'file')
parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
#parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--val', action='store_true', help='Plot graph for val')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output image directory')
parser.add_argument('--pose_graph', action='store_true',
                    help='Turn on Pose Graph Optimization')
parser.add_argument('--overfit', type=int, default=None, help='Reduce dataset to overfit few examples')
parser.add_argument('--result_file', default=None, help='Give file where results should be stored')
parser.add_argument('--display_segmentation', type=int, default=0, help='Show n segmentation results')
parser.add_argument('--show_class_dist', action='store_true', help='Create histogram of pixel classes in semantics')
parser.add_argument('--print_every', type=int, default=200, help='Plot progress every n steps')
args = parser.parse_args()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

settings = configparser.ConfigParser()
config_file = os.path.join(os.path.dirname(args.weights), 'config.ini')
with open(config_file, 'r') as f:
    settings.read_file(f)
seed = settings.getint('training', 'seed')
section = settings['hyperparameters']
dropout = section.getfloat('dropout')
train_split = section.getint('train_split', 6)
if (args.model.find('mapnet') >= 0) or args.pose_graph or (args.model.find('semantic') >= 0) or (args.model.find('multitask') >= 0):
    steps = section.getint('steps')
    skip = section.getint('skip')
    real = section.getboolean('real')
    variable_skip = section.getboolean('variable_skip')
    fc_vos = args.dataset == 'RobotCar'
    if args.pose_graph:
        vo_lib = section.get('vo_lib')
        sax = section.getfloat('s_abs_trans', 1)
        saq = section.getfloat('s_abs_rot', 1)
        srx = section.getfloat('s_rel_trans', 20)
        srq = section.getfloat('s_rel_rot', 20)
        
data_dir = osp.join('..', 'data', args.dataset)
stats_filename = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_filename)
crop_size_file = osp.join(data_dir, 'crop_size.txt')
crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))
resize = int(max(crop_size))

# model
feature_extractor = models.resnet34(pretrained=False)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)
if args.model in ['multitask', 'semanticOutput']:
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
if (args.model.find('mapnet') >= 0) or args.pose_graph  or args.model == 'semanticV0':
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
    posenetv2 = PoseNetV2(feature_extractor, droprate=dropout, pretrained=False,
                  filter_nans=(args.model == 'mapnet++'))
    model = MapNet(mapnet=posenetv2)
elif args.model == 'multitask':
    model = MultiTask(posenet=posenet, classes=classes, input_size=input_size)
elif args.model == 'semanticOutput':
    model = SemanticOutput(posenet=posenet, classes=classes, input_size=input_size)
else:
    model = posenet
model.eval()

# loss functions


def t_criterion(t_pred, t_gt): return np.linalg.norm(t_pred - t_gt)


q_criterion = quaternion_angular_error

# load weights
weights_filename = osp.expanduser(args.weights)
if osp.isfile(weights_filename):
    def loc_func(storage, loc): return storage
    checkpoint = torch.load(weights_filename, map_location=loc_func)
    #for key in checkpoint['model_state_dict']:
    #    if 'conv1' in key:
    #        print(key)
    load_state_dict(model, checkpoint['model_state_dict'])
    print('Loaded weights from {:s}'.format(weights_filename))
else:
    print('Could not load weights from {:s}'.format(weights_filename))
    sys.exit(-1)


# transformer
data_transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
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
# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(data_dir, args.scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

# dataset
train = not args.val
if train:
    print('Running {:s} on TRAIN data'.format(args.model))
else:
    print('Running {:s} on VAL data'.format(args.model))
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=train,
              transform=data_transform, target_transform=target_transform, seed=seed)


#default
input_types = ['img']
output_types = ['pose']
        
if args.model == 'semanticV0':
    input_types = ['label_colorized']
elif args.model == 'semanticOutput':
    output_types = ['label']
elif args.model == 'semanticV4':
    input_types = ['img', 'label']
elif 'semantic' in args.model:
    input_types = ['img', 'label_colorized']
elif 'multitask' in args.model:
    output_types = ['pose', 'label']
if args.dataset == 'DeepLoc':
        
        #print("Input types: %s\nOutput types: %s"%(input_types, output_types))
        
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
elif args.dataset in ['AachenDayNight', 'CambridgeLandmarks']:
    semantic_transform = (
            float_semantic_transform 
            if 'semanticV' in args.model else 
            int_semantic_transform)
        
    kwargs = dict(kwargs,
                      overfit=args.overfit,
                      semantic_transform=semantic_transform,
                      #semantic_colorized_transform=float_semantic_transform,
                      input_types=input_types, 
                      output_types=output_types,
                      train_split=train_split,
                      #concatenate_inputs=True
                 )

if (args.model.find('mapnet') >= 0) or args.pose_graph or (args.model.find('semantic') >= 0) or (args.model.find('multitask') >= 0):
    if args.pose_graph:
        assert real
        kwargs = dict(kwargs, vo_lib=vo_lib)
    vo_func = calc_vos_safe_fc if fc_vos else calc_vos_safe
    data_set = MF(dataset=args.dataset, steps=steps, skip=skip, real=real,
                  variable_skip=variable_skip, include_vos=args.pose_graph,
                  vo_func=vo_func, no_duplicates=False, **kwargs)
    L = len(data_set.dset)
elif args.dataset == '7Scenes':
    from dataset_loaders.seven_scenes import SevenScenes
    data_set = SevenScenes(**kwargs)
    L = len(data_set)
elif args.dataset == 'DeepLoc':
    from dataset_loaders.deeploc import DeepLoc
    data_set = DeepLoc(**kwargs)
    L = len(data_set)
elif args.dataset == 'RobotCar':
    from dataset_loaders.robotcar import RobotCar
    data_set = RobotCar(**kwargs)
    L = len(data_set)
elif args.dataset == 'AachenDayNight':
    from dataset_loaders.aachen import AachenDayNight
    data_set = AachenDayNight(**kwargs)
    L = len(data_set)
elif args.dataset == 'CambridgeLandmarks':
    from dataset_loaders.cambridge import Cambridge
    data_set = Cambridge(**kwargs)
    L = len(data_set)
else:
    raise NotImplementedError

# loader (batch_size MUST be 1)
batch_size = 1
assert batch_size == 1
loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                    num_workers=settings.getint('training','num_workers'), pin_memory=True)

# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
    model.cuda()

pred_poses = np.zeros((L, 7))  # store all predicted poses
targ_poses = np.zeros((L, 7))  # store all target poses

sem_imgs = []
acc_meter = AverageMeter()
intersection_meter = AverageMeter()
union_meter = AverageMeter()
time_meter = AverageMeter()
count_stats = {x : 0 for x in range(10)}
# inference loop
for batch_idx, (data, target) in enumerate(loader):
    #print("Data size: %s"%str(data.size()))
    #print("Target size: %s"%str(target.shape))
    """
    dual output: Data size: torch.Size([1, 6, 256, 455]) Target size: torch.Size([1, 6])
    normal: Data size: torch.Size([1, 3, 3, 256, 455]) Target size: torch.Size([1, 3, 6])
    """
    
    if batch_idx % args.print_every == 0:
        print('Image {:d} / {:d}'.format(batch_idx, len(loader)))

    # indices into the global arrays storing poses
    if (args.model.find('vid') >= 0) or args.pose_graph:
        idx = data_set.get_indices(batch_idx)
    else:
        idx = [batch_idx]
    idx = idx[len(idx) // 2]

    # output : 1 x 6 or 1 x STEPS x 6
    #print("Data size: %s"%str(data.size()))
    _, output, _ = step_feedfwd(data, model, CUDA, train=False)
    
    

    if args.model == 'semanticOutput' or args.model.find('multitask') >= 0:
        if args.model.find('multitask') >= 0:
            seg_out = output[1].cpu().data.numpy()
            seg_targ = target[1].data.numpy()
            """
            (1, 3, 256, 455)
            Shape out: (256, 455)    min: 1 max: 9
            Shape targ: (256, 455)   min: 1 max: 9
            """
            target = target[0]
            output = output[0]
        else:
            seg_out = output.cpu().data.numpy()
            seg_targ = target.data.numpy()
        seg_out = one_hot_to_one_channel(seg_out[0])
        seg_targ = seg_targ[0,0]
        #print("Shape out: %s\t min: %f max: %f"%(str(seg_out.shape), np.amin(seg_out), np.amax(seg_out)))
        #print("Shape targ: %s\t min: %d max: %d"%(str(seg_targ.shape), np.amin(seg_targ), np.amax(seg_targ)))
        acc, pix = accuracy(seg_out, seg_targ)
        intersection, union = intersectionAndUnion(seg_out, seg_targ, 10)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        unique, counts = np.unique(seg_targ, return_counts=True)
        if args.show_class_dist:
            d = dict(zip(unique, counts))
            for k in d:
                count_stats[k] += d[k]
        
        if args.display_segmentation > 0:
            sem_imgs.append((seg_out, seg_targ, data))
        if args.model == 'semanticOutput':
            continue
    
    s = output.size()
    #print("Output size is %s"%str(s))
    #print("Target size is %s"%str(target.size()))
    
    output = output.cpu().data.numpy().reshape((-1, s[-1]))
    target = target.numpy().reshape((-1, s[-1]))

    # normalize the predicted quaternions
    q = [qexp(p[3:]) for p in output]
    output = np.hstack((output[:, :3], np.asarray(q)))
    q = [qexp(p[3:]) for p in target]
    target = np.hstack((target[:, :3], np.asarray(q)))

    if args.pose_graph:  # do pose graph optimization
        kwargs = {'sax': sax, 'saq': saq, 'srx': srx, 'srq': srq}
        # target includes both absolute poses and vos
        vos = target[len(output):]
        target = target[:len(output)]
        output = optimize_poses(
            pred_poses=output,
            vos=vos,
            fc_vos=fc_vos,
            **kwargs)

    # un-normalize the predicted and target translations
    #print('Unnormalize: Mean: %s\tStd: %s'%(str(pose_m), str(pose_s)))
    output[:, :3] = (output[:, :3] * pose_s) + pose_m
    target[:, :3] = (target[:, :3] * pose_s) + pose_m
    #print(output.shape)
    #print(target.shape)

    # take the middle prediction
    pred_poses[idx, :] = output[len(output) // 2]
    targ_poses[idx, :] = target[len(target) // 2]

if (args.model == 'semanticOutput' or args.model.find('multitask') >= 0) and args.show_class_dist:
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('%s (%d)\t\tIoU: %.4f'%(sem_label_to_name(i), i, _iou))
    print('Mean IoU: %.4f, Accuracy: %.2f%%'%(iou[1:].mean(), acc_meter.average()*100))
    plt.bar([sem_label_to_name(i) for i in count_stats.keys()], count_stats.values())
    plt.xticks(rotation='vertical')
    plt.show()
if not args.model == 'semanticOutput':
    # calculate losses
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                                           targ_poses[:, :3])])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                                           targ_poses[:, 3:])])
    #eval_func = np.mean if args.dataset == 'RobotCar' else np.median
    #eval_str  = 'Mean' if args.dataset == 'RobotCar' else 'Median'
    #t_loss = eval_func(t_loss)
    #q_loss = eval_func(q_loss)
    #print '{:s} error in translation = {:3.2f} m\n' \
    #      '{:s} error in rotation    = {:3.2f} degrees'.format(eval_str, t_loss,
    med_trans, mean_trans = np.median(t_loss), np.mean(t_loss)
    med_rot, mean_rot = np.median(q_loss), np.mean(q_loss)
    report_string = 'Error in translation: median {:3.2f} m,  mean {:3.2f} m\n' \
        'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree'.format(med_trans, mean_trans,
                                                                            med_rot, mean_rot)
    print(report_string)

    if not args.result_file is None:
        store_string = args.weights.split('/')[-2] + '\n'
        if args.val:
            store_string += 'Val:'
        else:
            store_string += 'Train:'
        store_string += '\n'+report_string+'\n'
        with open(args.result_file, 'a+') as result_log:
            result_log.write(store_string)

if args.display_segmentation > 0 and len(sem_imgs) > 0:
    figures = []
    titles = []
    n_img = args.display_segmentation
    n_img = n_img if len(sem_imgs) > n_img else len(sem_imgs)
    plot_every = int(round(len(sem_imgs)/n_img))
    print("Plot every: %d"%plot_every)
    show_only = -1
    if show_only >= 0:
        print("Showing only class %s"% sem_label_to_name(show_only) )
    for i in sem_imgs[::plot_every]:
        out = i[0]
        targ = i[1]
        if show_only >= 0:
            out = out*(out == show_only)
            targ = targ*(targ == show_only)
        out = sem_labels_to_rgb(out)
        targ = sem_labels_to_rgb(targ)
        inp_img = np.moveaxis(i[2].data.numpy()[0,0], 0, -1)
        """
        This is ugly but whatever
        """
        out = np.moveaxis(normalize(out), -1, 0)#.astype(np.uint8)
        targ = np.moveaxis(normalize(targ), -1, 0)#.astype(np.uint8)
        inp_img = np.moveaxis(normalize(inp_img), -1, 0)
        figures.append(torch.from_numpy(inp_img.astype(np.float32)))
        figures.append(torch.from_numpy(out.astype(np.float32)))
        figures.append(torch.from_numpy(targ.astype(np.float32)))
        #titles.append("Input")
        #titles.append("Predicted")
        #titles.append("Target")
    #print("#img: %d\t#titles: %d"%(len(figures), len(titles)))
    figures = [make_grid(figures[i], nrow=1) for i in range(len(figures))]
    output_dir = '../figures/'
    images = make_grid(figures,nrow=3)
    os.makedirs(output_dir, exist_ok=True)
    output_name = 'semantic_results_'+args.model+'.png'
    output_path = os.path.join(output_dir, output_name)
    save_image(images, output_path)
    final = plt.imread(output_path)
    plt.imshow(final)
    plt.show()
    
    #show_images(figures, len(figures)/3, titles=titles)

# create figure object

if args.output_dir is not None and not args.model == 'semanticOutput':
    fig = plt.figure()
    if args.dataset != '7Scenes':
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    # plot on the figure object
    ss = max(1, int(len(data_set) / 1000))  # 100 for stairs
    # scatter the points and draw connecting line
    x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
    y = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
    #for i in range(25):
    #    print('Predicted: %s\tTarget: %s\Distance: %.2f'%(str(pred_poses[i,:3]), str(targ_poses[i,:3]), t_criterion(pred_poses[i,:3], targ_poses[i,:3])))
    if args.dataset != '7Scenes':  # 2D drawing
        ax.plot(x, y, c='b')
        ax.scatter(x[0, :], y[0, :], c='r', s = 0.5)
        ax.scatter(x[1, :], y[1, :], c='g', s = 0.5)
    else:
        z = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
        for xx, yy, zz in zip(x.T, y.T, z.T):
            ax.plot(xx, yy, zs=zz, c='b')
        ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0)
        ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0)
        ax.view_init(azim=119, elev=13)

    if DISPLAY:
        plt.show(block=True)

    if args.output_dir is not None:
        model_name = args.model
        if args.weights.find('++') >= 0:
            model_name += '++'
        if args.pose_graph:
            model_name += '_pgo_{:s}'.format(vo_lib)
        experiment_name = '{:s}_{:s}_{:s}'.format(
            args.dataset, args.scene, model_name)
        image_filename = osp.join(osp.expanduser(args.output_dir),
                                  '{:s}.png'.format(experiment_name))
        fig.savefig(image_filename)
        print('{:s} saved'.format(image_filename))
        result_filename = osp.join(osp.expanduser(args.output_dir), '{:s}.pkl'.
                                   format(experiment_name))
        with open(result_filename, 'wb') as f:
            pickle.dump({'targ_poses': targ_poses, 'pred_poses': pred_poses}, f)
        print('{:s} written'.format(result_filename))

