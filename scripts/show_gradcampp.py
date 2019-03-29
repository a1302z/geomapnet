from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.cuda
import configparser
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import set_paths
from models.posenet import *
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import optimize_poses, quaternion_angular_error, qexp,\
    calc_vos_safe_fc, calc_vos_safe
from common.criterion import *
from common.vis_utils import show_images, sem_labels_to_rgb, normalize, one_hot_to_one_channel, sem_label_to_name
from common.stolen_utils import *
from dataset_loaders.composite import MF
import argparse
import os
import os.path as osp
import sys
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
import PIL
from common.gradcam_pp import *

parser = argparse.ArgumentParser(description='Create activation maps for pose regression networks')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'DeepLoc', 'RobotCar'),help='Dataset')
parser.add_argument('--scene', type=str, default='', help='Scene name')
parser.add_argument('--weights', type=str, help='trained weights to load')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++', 'multitask', 'semanticOutput',
                                        'semanticV0', 'semanticV1','semanticV2', 'semanticV3', 'semanticV4'),
                    help='Model to use (mapnet includes both MapNet and MapNet++ since their'
                    'evluation process is the same and they only differ in the input weights'
                    'file')
parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
parser.add_argument('--val', action='store_true', help='Plot graph for val')
parser.add_argument('--output_dir', type=str, default=None, help='Output image directory')
parser.add_argument('--overfit', type=int, default=None, help='Reduce dataset to overfit few examples')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--uncertainty_criterion', action='store_true', help='Use criterion which weighs losses with uncertainty')
parser.add_argument('--n_activation_maps', type=int, default=5, help='Number of activation maps plotted')
parser.add_argument('--layer_name', type=str, default='layer4', help='Which layer to use for activation maps')
parser.add_argument('--plot_per_layer', action='store_true', help='If given creates for each layer a seperate image. Else it stores all in one image') 
parser.add_argument('--image_num', type=str, default=None, help='Only plot for these images e.g.: 10,20,30 Deactivates n_activation_maps option') 
args = parser.parse_args()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    
settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
    settings.read_file(f)
seed = settings.getint('training', 'seed')
section = settings['hyperparameters']
dropout = section.getfloat('dropout')
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
    
feature_extractor = models.resnet34(pretrained=False)
##WARNING: Changed dropout to be 0.0 
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
    #for key in checkpoint['model_state_dict']:
    #    if 'conv1' in key:
    #        print(key)
    load_state_dict(model, checkpoint['model_state_dict'])
    print('Loaded weights from {:s}'.format(weights_filename))
else:
    print('Could not load weights from {:s}'.format(weights_filename))
    sys.exit(-1)

data_dir = osp.join('..', 'data', args.dataset)
stats_filename = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_filename)
# transformer
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

int_semantic_transform = transforms.Compose([
        transforms.Resize(256,0), #Nearest interpolation
        transforms.Lambda(lambda pic: torch.from_numpy(np.array(pic, np.int64, copy=False)))
    ])
float_semantic_transform = transforms.Compose([
        transforms.Resize(256,0), #Nearest interpolation
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
## WARNING: Fixed seed
kwargs = dict(scene=args.scene, data_path=data_dir, train=train,
              transform=data_transform, target_transform=target_transform, seed=seed)

if args.dataset == 'DeepLoc':
        #default
        input_types = ['left']
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
if (args.model.find('mapnet') >= 0) or (args.model.find('semantic') >= 0) or (args.model.find('multitask') >= 0):
    vo_func = calc_vos_safe_fc if fc_vos else calc_vos_safe
    data_set = MF(dataset=args.dataset, steps=steps, skip=skip, real=real,
                  variable_skip=variable_skip, include_vos=False,
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
else:
    raise NotImplementedError
    
# loss function
if args.model == 'posenet':
    train_criterion = PoseNetCriterion(
        sax=sax, saq=saq, learn_beta=False)
    val_criterion = PoseNetCriterion()
elif args.model == 'semanticOutput':
    train_criterion = SemanticCriterion()
    val_criterion = SemanticCriterion()
elif 'mapnet' in args.model or 'semantic' in args.model or 'multitask' in args.model:
    kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq,
                  learn_beta=False, learn_gamma=False)
    
    if 'multitask' in args.model:
        kwargs = dict(kwargs, dual_target=True, 
                      sas=sas, learn_sigma=False)
        
    if '++' in args.model:
        kwargs = dict(kwargs, gps_mode=(vo_lib == 'gps'))
        train_criterion = MapNetOnlineCriterion(**kwargs)
        val_criterion = MapNetOnlineCriterion()
    elif args.uncertainty_criterion:
        train_criterion = UncertainyCriterion(**kwargs, learn_log=True)
        val_criterion = UncertainyCriterion(dual_target='multitask' in args.model, learn_log=True)
    else:
        train_criterion = MapNetCriterion(**kwargs)
        val_criterion = MapNetCriterion(dual_target='multitask' in args.model)
else:
    raise NotImplementedError

# loader (batch_size MUST be 1)
batch_size = 1
assert batch_size == 1
loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                    num_workers=5, pin_memory=True)

# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
    model.cuda()
print("Created dataset loader")



layers = None
if ',' in args.layer_name:
    layers = args.layer_name.split(',')
else:
    layers = [args.layer_name]


model_dicts = [dict(model_type=args.model, arch=model, layer_name=layer, input_size=(224, 224)) for layer in layers]
gradcams = [GradCAMpp(d) for d in model_dicts]

# get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
#normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

images = [[] for i in range(len(layers))]  
every = int(len(loader)/args.n_activation_maps)
num_imgs = None if args.image_num is None else [int(x) for x in args.image_num.split(',')]
for batch_idx, (data, target) in enumerate(loader):
    if num_imgs is None:
        if batch_idx % every != 0:
            continue
    else:
        if batch_idx not in num_imgs:
            continue
    
    print('Image {:d} / {:d}'.format(batch_idx, len(loader)))
    #normed_img = normalizer.do(data)

    for i, g in enumerate(gradcams):
        # get a GradCAM saliency map on the class index 10.
        mask, logit = g(data, target, val_criterion)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        img = data[:,0]
        heatmap, cam_result = visualize_cam(mask, img)
        images[i].append(torch.stack([img.squeeze().cpu(), heatmap, cam_result], 0))
    
images = [make_grid(torch.cat(images[i], 0), nrow=1) for i in range(len(images))]
output_dir = '../figures/'
if not args.plot_per_layer:
    images = make_grid(images,nrow=len(layers))
    os.makedirs(output_dir, exist_ok=True)
    output_name = 'activation_maps_'+args.model+'_'+args.layer_name+'.png'
    output_path = os.path.join(output_dir, output_name)
    save_image(images, output_path)
    final = plt.imread(output_path)
    plt.imshow(final)
    plt.show()
else:
    for i in range(len(images)):
        output_name = 'activation_maps_'+args.model+'_'+layers[i]+'.png'
        output_path = os.path.join(output_dir, output_name)
        save_image(images[i], output_path)
        final = plt.imread(output_path)
        plt.imshow(final)
        plt.show()
