from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.cuda
import configparser
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import set_paths
from models.posenet import *
from common.train import load_state_dict
from dataset_loaders.composite import MF
import argparse
import os.path as osp
from common.pose_utils import calc_vos_safe_fc, calc_vos_safe, qexp
from torch.autograd import Variable
import time
import tqdm



parser = argparse.ArgumentParser(description='TSNE visualization of localization network')
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
parser.add_argument('--val', action='store_true', help='Use validation data')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output image directory')
parser.add_argument('--overfit', type=int, default=None, help='Reduce dataset to overfit few examples')
parser.add_argument('--use_augmentation', action='store_true', help='Use augmented images. Needs to be supported by dataloader (currently only AachenDayNight)')
parser.add_argument('--load_previous', default=None, type=str, help='File of previous embedding to restore')
parser.add_argument('--arp_vis', action='store_true', help='Create visualization like in understanding ARP paper')
parser.add_argument('--compare_query', action='store_true', help='Find closest images to query images')
parser.add_argument('--suffix', default='', type=str, help='Suffix for output image')
parser.add_argument('--no_tsne', action='store_false', help='Show no tsne representation')
args = parser.parse_args()


if args.load_previous:
    print('Load saved embedding')
    embedding = np.load(args.load_previous)
else:
    settings = configparser.ConfigParser()
    config_file = os.path.join(os.path.dirname(args.weights), 'config.ini')
    with open(config_file, 'r') as f:
        settings.read_file(f)
    seed = settings.getint('training', 'seed')
    section = settings['hyperparameters']
    activation_function = section.get('activation_function', 'relu').lower()
    feature_dim = section.getint('feature_dim', 2048)
    base_poses = section.get('base_poses', 'None')
    dropout = section.getfloat('dropout')
    train_split = section.getint('train_split', 6)
    if (args.model.find('mapnet') >= 0) or (args.model.find('semantic') >= 0) or (args.model.find('multitask') >= 0):
        steps = section.getint('steps')
        skip = section.getint('skip')
        real = section.getboolean('real')
        variable_skip = section.getboolean('variable_skip')
        fc_vos = args.dataset == 'RobotCar'

    data_dir = osp.join('..', 'data', args.dataset)
    stats_filename = osp.join(data_dir, args.scene, 'stats.txt')
    stats = np.loadtxt(stats_filename)
    crop_size_file = osp.join(data_dir, 'crop_size.txt')
    crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))
    resize = int(max(crop_size))

    # modelset_base_poses = None
    set_base_poses = None
    if base_poses in ['unit_vectors', 'unit']:
        feature_dim = 6
    """    set_base_poses = np.array([[1,0,0], [0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float)
        set_base_poses = torch.from_numpy(set_base_poses.T).float()
        print('Base poses set to unit vectors')
    elif base_poses == 'gaussian':
        set_base_poses = np.random.normal(size=(3, feature_dim))
        set_base_poses = torch.from_numpy(set_base_poses).float()
        print('Base poses sampled from gaussian')
    else:
        print('Standard initialization for base poses')
    """
    af = torch.nn.functional.relu
    if activation_function == 'sigmoid':
        af = torch.nn.functional.sigmoid
        print('Using sigmoid as activation function')
    feature_extractor = models.resnet34(pretrained=False)
    posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False,
                  feat_dim=feature_dim, activation_function=af, 
                 set_base_poses=set_base_poses)
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
    if (args.model.find('mapnet') >= 0)  or args.model == 'semanticV0':
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
        model = MultiTask(posenet=posenet, classes=classes, input_size=input_size, feat_dim=feature_dim, 
                 set_base_poses=set_base_poses)
    elif args.model == 'semanticOutput':
        model = SemanticOutput(posenet=posenet, classes=classes, input_size=input_size, feat_dim=feature_dim)
    else:
        model = posenet
    model.eval()

    # loss functions


    #def t_criterion(t_pred, t_gt): return np.linalg.norm(t_pred - t_gt)
    #q_criterion = quaternion_angular_error

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
    
    if args.arp_vis:
        print('Calculate understanding arp visualization')
        if args.model == 'multitask':
            layer = model.fc_xyz.weight.data.numpy()
        elif args.model == 'mapnet':
            layer = model.mapnet.fc_xyz.weight.data.numpy()
        else:
            raise NotImplementedError('No other models implemented yet')
        #print(layer)
        print(layer.shape)
        #layer = np.multiply(layer.T, pose_s.flatten())
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(layer[0], layer[2], layer[1], alpha=0.3)
        if args.output_dir:
            name = 'arp_visualization_{:s}'.format(args.model)
            if len(args.suffix) > 0:
                name = '{:s}_{:s}'.format(name, args.suffix)
            plt.savefig(osp.join(args.output_dir, '{:s}.png'.format(name)))
        plt.show()
        #print(layer.shape)
        #print('Do you want to exit the script [yes|NO]')
        #if input().lower() in ['yes', 'y']:
        #    exit()
            
    if args.no_tsne:

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
            if args.dataset == 'AachenDayNight':
                kwargs['night_augmentation'] = args.use_augmentation

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

        feature_vectors = []
        distance = []
        assert args.model in ['multitask', 'mapnet'], 'TSNE not implemented for this model yet'
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(loader), total = L):
            #if batch_idx % 200 == 0:
            #    print('Image {:d} / {:d}'.format(batch_idx, len(loader)))
            idx = [batch_idx]
            idx = idx[len(idx) // 2]
            with torch.set_grad_enabled(False):
                data_var = Variable(data, requires_grad=False)
                if CUDA:
                    data_var = data_var.cuda(async=True)
                output = model.__feature_vector__(data_var)
                if args.model == 'multitask':
                    output = output[0]
                vector = output.detach().cpu().numpy()
                if len(vector.shape) > 1:
                    vector = vector[vector.shape[0]//2]
                feature_vectors.append(vector)
                distance.append(np.linalg.norm(vector))
            target = target[0]
            target = target.numpy().reshape((-1, 6))
            q = [qexp(p[3:]) for p in target]
            target = np.hstack((target[:, :3], np.asarray(q)))
            target[:, :3] = (target[:, :3] * pose_s) + pose_m
            targ_poses[idx, :] = target[len(target) // 2]


        feature_vectors = np.vstack(feature_vectors)

        #distance = np.stack(distance)
        distance = np.stack([np.linalg.norm(targ_poses[i, :3]) for i in range(targ_poses.shape[0])])
        print(feature_vectors.shape)
        t1 = time.time()
        embedding = TSNE(n_components=2).fit_transform(feature_vectors)
        t = time.time() - t1
        print('TSNE took %d seconds'%t)
        print(embedding.shape)
        pairwise_distance_night_day = []
        length_half_dataset = len(feature_vectors)//2
        if args.use_augmentation:
            for i in range(length_half_dataset):
                pairwise_distance_night_day.append(np.linalg.norm(feature_vectors[i]- feature_vectors[i+length_half_dataset]))
            pairwise_distance_night_day = np.stack(pairwise_distance_night_day)
            print('Distances between day and corresponding night image: ')
            print(' Min: %.5f\t Max: %.5f\t Mean: %.5f\tMedian: %.5f'%(pairwise_distance_night_day.min(), pairwise_distance_night_day.max(), pairwise_distance_night_day.mean(), np.median(pairwise_distance_night_day)))
            print('Distance random day to day image: %.5f'% np.linalg.norm(feature_vectors[0]-feature_vectors[1]))



    #np.save('logs/embedding.npy', embedding)

#plt.scatter(embedding[:,0], embedding[:,1])

if args.no_tsne:
    med, std = np.median(distance), np.std(distance)
    #print('Clipping distances to max {:1f}'.format(med+std))
    #distance = np.clip(distance, min(abs(med-std), 0), med+std)
    #distance = np.clip(distance, 0, 12)

    fig, ax = plt.subplots()
    border = len(embedding) // 2 if args.use_augmentation else len(embedding)
    if args.use_augmentation:
        ax.scatter(embedding[border:,0], embedding[border:,1], color = 'black', label = 'Night', alpha=0.9)
        ax.scatter(embedding[:border,0], embedding[:border,1], color = 'orange', label = 'Day', alpha=0.5)
        ax.legend()
    else:
        #im = ax.scatter(embedding[:,0], embedding[:,1],c=distance, cmap=plt.cm.get_cmap('hot'), alpha=0.95)
        #fig.colorbar(im)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    if args.output_dir:
        id_str = '_val' if args.val else '_train'
        id_str = '{:s}_{:s}'.format(id_str, args.model)
        if len(args.suffix) > 0:
            id_str = '{:s}_{:s}'.format(id_str, args.suffix)
        plt.savefig(osp.join(args.output_dir, 'tsne_embedding'+id_str+'.png'))
    plt.show()


if args.compare_query and args.no_tsne:
    from sklearn.neighbors import NearestNeighbors
    knn = 5
    nbrs = NearestNeighbors(n_neighbors=knn).fit(feature_vectors)
    distance_feat_space, indices = nbrs.kneighbors(feature_vectors)
    distances = np.zeros(indices.shape,dtype=np.float)
    print(distances.shape)
    calc_dist = lambda i, j: np.linalg.norm(targ_poses[i,:3]-targ_poses[j,:3])        
    for i, idc in enumerate(indices):
        #idx = idc[2] if args.use_augmentation else idc[1]
        #distance = calc_dist(i,j)
        #print('Point %d to %d\tfeat space dist: %.2f\treal distance: %.2f'%(i, idx, distance_feat_space[i, 2], distance))
        for j, idx in enumerate(idc):
            d = calc_dist(i,idx)
            distances[i,j] = d
    print(distances)
    num = 2
    #fig, ax = plt.subplots(10,2)
    plt.figure(figsize=(knn*2,num*2))
    prefix = '../data/deepslam_data/AachenDayNight/'
    for k, i in enumerate(np.random.randint(0, len(loader), size=num)):
        ax1 = plt.subplot(num,knn,k*knn+1)
        ax1.axis('off')
        border = len(loader)//2 if args.use_augmentation else len(loader)
        if i < border:
            img1 = prefix+'images_upright/db/'+str(i+1)+'.jpg'
        else:
            img1 = prefix+'AugmentedNightImages/'+str(i//2+1)+'.png'
        ax1.imshow(plt.imread(img1))
        
        for x in range(1, knn):
            ax2 = plt.subplot(num,knn,k*knn+1+x)
            ax2.axis('off')
            idx = indices[i,x]
            if idx < border:
                img2 = prefix+'images_upright/db/'+str(idx+1)+'.jpg'
            else:
                img2 = prefix+'AugmentedNightImages/'+str(idx//2+1)+'.png'
            ax2.imshow(plt.imread(img2))
    plt.subplots_adjust(wspace=0, hspace=0)
    if args.output_dir:
        plt.savefig(osp.join(args.output_dir, 'similar_images1.png'), bbox_inches='tight')
    plt.show()

    #nbrs_compare = NearestNeighbors(n_neighbors=5).fit(targ_poses[:,:3])
    """
    This was for comparison to query images
    from dataset_loaders.utils import load_image
    files = []
    feature_vectors_query = []
    for dirpath, dirnames, filenames in os.walk('../data/deepslam_data/AachenDayNight/images_upright/query'):
        files += [(f, os.path.join(dirpath, f)) for f in filenames if f.endswith('.jpg')]
    print('Found %d files'%len(files))
    if args.overfit: 
        files = files[:args.overfit]
    L = len(files)
    pred_poses = np.zeros((L, 7))  # store all predicted poses

    for i, file in enumerate(files):
        if i % 200 == 0:
            print('Image {:d} / {:d}'.format(i, len(files)))

        #print('Load file %s'%file)
        data = load_image(file[1])
        data = data_transform(data).unsqueeze(0).unsqueeze(0)
        
        with torch.set_grad_enabled(False):
            data_var = Variable(data, requires_grad=False)
            if CUDA:
                data_var = data_var.cuda(async=True)
            output = model.__feature_vector__(data_var)[0]
            vector = output.detach().cpu().numpy()
            if len(vector.shape) > 1:
                vector = vector[vector.shape[0]//2]
            feature_vectors_query.append(vector)
    feature_vectors_query = np.vstack(feature_vectors_query)     
    print(feature_vectors_query.shape)
    t1 = time.time()
    nbrs = NearestNeighbors(n_neighbors=1).fit(feature_vectors)
    distances, indices = nbrs.kneighbors(feature_vectors_query)
    t2 = time.time()-t1
    print('k-NN took %d seconds'%t2)
    print(distances.shape)
    min_distances = distances[:,0]
    print('Min distances between training set and query images: \n Min: %.2f\tMax: %.2f\tMean: %.2f\tMedian: %.2f'%(min_distances.min(), min_distances.max(), min_distances.mean(), np.median(min_distances)))
    """