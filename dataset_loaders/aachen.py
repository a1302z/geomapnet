from __future__ import division
import sys
sys.path.insert(0, '../')
from torch.utils import data
import numpy as np
import ntpath
import random
from common.pose_utils import process_poses_quaternion
import transforms3d.quaternions as txq
from dataset_loaders.utils import load_image, single_channel_loader
import os
import tqdm

def identity(x):
    return x

available_styles = [
    'goeritz',
    'asheville',
    'mondrian',
    'scene_de_rue',
    'flower_of_life',
    'antimonocromatismo',
    'woman_with_hat_matisse',
    'trial',
    'sketch',
    'picasso_self_portrait',
    'picasso_seated_nude_hr',
    'la_muse',
    'contrast_of_forms',
    'brushstrokes',
    'the_resevoir_at_poitiers',
    'woman_in_peasant_dress'
]

class AachenDayNight(data.Dataset):
    
    def __init__(self, data_path, train, resize, train_split=20, overfit=None,
                seed=7,input_types='img', output_types='pose', real=False
                ,transform=identity, semantic_transform=identity, scene='', target_transform=identity, 
                night_augmentation=False, only_augmentation=False, use_stylization=False,
                verbose=False):
        """
        seed=7, overfit=None, reduce_data=True,
        transform=identity, 
        semantic_colorized_transform=identity, target_transform=identity, 
        vo_lib='orbslam', scene='', concatenate_inputs=False):
        """
        np.random.seed(seed)
        self.data_path = data_path
        self.input_types = input_types if type(input_types) is list else [input_types]
        self.output_types = output_types if type(output_types) is list else [output_types]
        self.train = train
        self.night_augmentation = night_augmentation
        self.only_augmentation = only_augmentation
        self.use_stylization = use_stylization
        self.resize = resize
        assert not (night_augmentation and only_augmentation), 'Not both augmentation options possible'
        if verbose:
            if self.night_augmentation:
                print('Use augmented images too')
            elif self.only_augmentation:
                print('Use only augmented images')
        self.train_split = train_split
        self.train_idcs = []
        self.val_idcs = []
        self.transforms = {
            'img' : transform, 
            #'right' : transform,
            #'label_colorized' : semantic_colorized_transform,
            'label' : semantic_transform,
            #'depth' : transform,
            'pose' : target_transform,
            'point_cloud' : target_transform
        }
        point_cloud = 'point_cloud' in self.input_types or 'point_cloud' in self.output_types
        filename = os.path.join(self.data_path, 'aachen_cvpr2018_db.nvm')
        f = open(filename, 'r')
        lines = f.readlines()
        num_points = int(lines[2].strip())
        if point_cloud:
            sift_lines = lines[num_points+4:]
        lines = lines[:num_points+3]
        lines[3:] = [x.strip().split(' ') for x in lines[3:]]
        self.img_paths = []
        poses = []
        for i in range(3, len(lines)):
            l = lines[i]
            q = [float(x) for x in l[2:6]]
            c = [float(x) for x in l[6:9]]
            poses.append(np.asarray(c+q))
            self.img_paths.append(os.path.join('images_upright', l[0]))
        self.poses = np.vstack(poses)
        del poses
        self.sem_labels = None
        if 'label' in self.input_types or 'label' in self.output_types:
            self.sem_labels = []
            for i in self.img_paths:
                sem_path = os.path.join(self.data_path, 'sem_labels', ntpath.basename(i).replace('.jpg','.png'))
                if os.path.isfile(sem_path):
                    self.sem_labels.append(sem_path)
                else:
                    raise AssertionError('WARNING: SEMANTIC LABEL NOT FOUND')
            if verbose:
                print('All semantic labels successfully loaded')
        self.night_img_paths = None
        if self.night_augmentation or self.only_augmentation:
            self.night_img_paths = []
            for i in self.img_paths:
                img_name = ntpath.basename(i).replace('.jpg','.png')
                aug_path = os.path.join(self.data_path, 'AugmentedNightImages_high_res', img_name)
                assert os.path.isfile(aug_path), 'Augmented file not found'
                self.night_img_paths.append(aug_path)
        if self.use_stylization:
            self.stylized_paths = []
            for i, path in enumerate(self.img_paths):
                img_dir, base_name = os.path.split(path)
                base_name = base_name.split('.')
                base_name = base_name[0] + '-stylized-'+available_styles[i % len(available_styles)] + '.' + base_name[1]
                file = os.path.join(self.data_path, img_dir, 'stylized', base_name)
                assert os.path.isfile(file), 'Stylized image not found: {:s}'.format(file)
                self.stylized_paths.append(file)
                
        for i in range(num_points):
            if i % self.train_split == 0:
                self.val_idcs.append(i)
            else:
                self.train_idcs.append(i)
        if verbose:
            print('Read %d points'%self.poses.shape[0])
            print('%d training points\n%d validation points'%(len(self.train_idcs), len(self.val_idcs)))
        """
        <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
        <Measurement> = <Image index> <Feature Index> <xy>
        """
        if point_cloud:
            num_sifts = int(sift_lines[0])
            sift_lines = [x.strip().split(' ') for x in sift_lines[1:]]
            if verbose:
                print('Stripped sift lines')
            sift_points = []
            self.points_per_img = {i:[] for i in range(num_points)}
            if verbose:
                print('Created dict for img-point correspondances')
            for i in range(num_sifts):
                if i % 500000 == 0:
                    print('%d/%d'%(i,num_sifts))
                line = sift_lines[i]
                num_imgs = int(line[6])
                for j in range(num_imgs):
                    idx = int(line[7+j*4])
                    self.points_per_img[idx].append(i)
                xyz = np.array([float(line[0]), float(line[1]), float(line[2])])
                sift_points.append(xyz)
            if verbose:
                print('Done loading %d points'%len(sift_points))
            self.sift_points = np.vstack(sift_points)
            if verbose:
                print("Found %d/%d sift points in threshold"%(self.sift_points.shape[0], num_sifts))
                
        if overfit is not None:
            self.train_idcs = self.train_idcs[:overfit]
            
        ##Reduce data
        self.poses = self.poses[self.train_idcs if self.train else self.val_idcs]
        self.img_paths = [self.img_paths[x] for x in (self.train_idcs if self.train else self.val_idcs)]
        if self.night_img_paths is not None:
            self.night_img_paths = [self.night_img_paths[x] for x in (self.train_idcs if self.train else self.val_idcs)]
        if self.sem_labels is not None:
            self.sem_labels = [self.sem_labels[x] for x in (self.train_idcs if self.train else self.val_idcs)]
        if point_cloud:
            self.points_per_img = {j: self.points_per_img[i] for j, i in enumerate(self.train_idcs if self.train else self.val_idcs)}
        if verbose:
            print('Reduced data')
            print(self.poses.shape)
            print(len(self.img_paths))
            if self.night_img_paths:
                print(len(self.night_img_paths))
            if self.sem_labels:
                print(len(self.sem_labels))
            if point_cloud:
                print(len(self.points_per_img))
                
        pose_stats_filename = os.path.join(self.data_path, 'pose_stats.txt')
        if train and not real:
            # optionally, use the ps dictionary to calc stats
            mean_t = self.poses[:, :3].mean(axis=0)
            std_t = self.poses[:, :3].std(axis=0)
            print('Stats {:s} saved to {:s}'.format('{:s}/{:s}'.format(str(mean_t), str(std_t)), pose_stats_filename))
            np.savetxt(
                pose_stats_filename, np.vstack(
                    (mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
            print('Use stats: {:s}/{:s}'.format(str(mean_t), str(std_t)))
        self.poses = process_poses_quaternion(self.poses[:,:3], self.poses[:,3:], mean_t, std_t, np.eye(3), np.zeros(3), np.ones(1))
        if verbose:
            print('Process poses: %s'%str(self.poses.shape))
            
        if resize is not None:
            self.__resize__(resolution=resize)
            
    def __len__(self):
        factor = 2 if self.night_augmentation or self.use_stylization else 1
        return len(self.poses)*factor
    
    def __resize__(self, resolution=256):
        import torchvision.transforms as tf
        transform = tf.Resize(resolution)
        L = self.__len__()
        for index in tqdm.tqdm(range(L), total=L, desc='Resize', leave=False):
            augmentation_index = 0
            if self.night_augmentation or self.use_stylization:
                augmentation_index = index // len(self.poses)
                index = index % len(self.poses)
            elif self.only_augmentation:
                augmentation_index = 1
            if augmentation_index == 0:
                img_path = os.path.join(self.data_path,self.img_paths[index])
                #direct, name = self.data_path, self.img_paths[index]
            else:
                if self.night_augmentation:
                    img_path = self.night_img_paths[index]
                elif self.use_stylization:
                    img_path = self.stylized_paths[index]
            direct, name = os.path.split(img_path)
            new_dir = os.path.join(direct, 'resized')
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            new_name = 'resized_{}px_{:s}'.format(resolution, name)
            new_path = os.path.join(new_dir, new_name)
            if not os.path.isfile(new_path):
                img = load_image(img_path)
                try:
                    img = transform(img)
                except:
                    print('Problem with image {}'.format(img_path))
                    exit()
                img.save(new_path)
            if augmentation_index == 0:
                self.img_paths[index] = os.path.join('images_upright', 'db', new_name)
            else:
                if self.night_augmentation:
                    self.night_img_paths[index] = new_path
                elif self.use_stylization:
                    self.stylized_paths[index] = new_path
            #print('New image path: {:s}'.format(new_path))
        
            
            
    def __getitem__(self, index):
        augmentation_index = 0
        if self.night_augmentation or self.use_stylization:
            augmentation_index = index // len(self.poses)
            index = index % len(self.poses)
        elif self.only_augmentation:
            augmentation_index = 1
        inps = []
        outs = []
        for inp in self.input_types:
            if inp == 'img':
                if augmentation_index == 0:
                    img_path = os.path.join(self.data_path,self.img_paths[index])
                    if self.resize:
                        d, n = os.path.split(img_path)
                        img_path = os.path.join(d, 'resized', n)
                else:
                    if self.night_augmentation:
                        img_path = self.night_img_paths[index]
                    elif self.use_stylization:
                        img_path = self.stylized_paths[index]
                img = load_image(img_path)
                img = None if img is None else self.transforms['img'](img)
                inps.append(img)
            elif inp == 'label':
                img = single_channel_loader(self.sem_labels[index])
                img = None if img is None else self.transforms['label'](img)
                inps.append(img)
            elif inp == 'point_cloud':
                pc = self.sift_points[self.points_per_img[index]]
                #Normalize 
                #print('----------- NORMALIZE ---------')
                #print(pc.shape)
                #print(pc.mean(axis=1).shape)
                pc = pc - pc.mean(axis=0)
                pc = pc / pc.std(axis=0)
                pc = pc.T
                #print(pc.mean(axis=1))
                #print(pc.std(axis=1))
                pc = None if pc is None else self.transforms['point_cloud'](pc)
                inps.append(pc)
            else:
                raise NotImplementedError('Not implemented yet (%s)'%str(inp))
        for out in self.output_types:
            if out == 'pose':
                p = self.poses[index]
                p = None if p is None else self.transforms['pose'](p).squeeze()
                outs.append(p)
            elif out == 'label':
                img = single_channel_loader(self.sem_labels[index])
                img = None if img is None else self.transforms['label'](img)
                outs.append(img)
            else:
                raise NotImplementedError('Not implemented yet')
        return inps[0] if len(inps) <= 1 else tuple(inps), outs[0] if len(outs) <= 1 else outs
            
if __name__ == '__main__':
    from torchvision import transforms, utils
    import torch
    import matplotlib.pyplot as plt
    test = AachenDayNight('../data/deepslam_data/AachenDayNight/', True, 224, use_stylization=True)
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    L = len(test)
    print('Dataset has {:d} entries'.format(L))
    imgs = torch.stack([tf(test[i][0]) for i in range(L//2-7, L//2+7)])
    grid = utils.make_grid(imgs, 7)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    """
    loader = AachenDayNight('../data/deepslam_data/AachenDayNight/', True, verbose=True)
    print(len(loader))
    loader = AachenDayNight('../data/deepslam_data/AachenDayNight/', False, input_types=['img', 'label', 'point_cloud'], verbose = True, night_augmentation=True)
    print(len(loader))
    """"""
    for i, (inp, target) in enumerate(loader):
        if i % 100 == 0:
            print('%d/%d loaded data points'%(i, len(loader)))
        if i > len(loader):
            break
    """"""
    x, c = loader[0]
    pose_stats_filename = os.path.join('../data/deepslam_data/AachenDayNight/', 'pose_stats.txt')
    mean_t, std_t = np.loadtxt(pose_stats_filename)
    c = c[:3]*std_t +mean_t
    print('len inps: %d'%len(x))
    for x_i in x:
        if type(x_i) is np.ndarray:
            print(x_i.shape)
        else:
            print(type(x_i))
    pc = x[2]
    plt.imshow(x[0])
    plt.show()
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')
    thresh = 2
    median = np.median(pc, axis=1)
    #ax.scatter3D(c[0],c[2],c[1], color='red')
    ax.scatter3D(pc[0],pc[2],pc[1], s = 0.5, alpha = 0.2)
    ax.set_xlim3d(-thresh,thresh)
    ax.set_ylim3d(-thresh,thresh)
    ax.set_zlim3d(-thresh,thresh)
    plt.show()
    """