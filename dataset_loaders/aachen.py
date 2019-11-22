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
from PIL import Image

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

class Sample:
    def __init__(self, index, path, pose, nighttime=None, stylized=None, semantic=None, point_cloud=None):
        self.index = index
        self.path = path
        self.pose = pose
        self.nighttime = nighttime
        self.stylized=stylized
        self.semantic = semantic
        self.point_cloud = point_cloud
        
class RangeCheck:
    def __init__(self, range_tuples=[], labels=[]):
        assert len(range_tuples) == len(labels), 'Tuples and labels do not have same size'
        for t in range_tuples:
            assert t[0] < t[1], 'Left border not smaller than right'
        self.range_tuples = range_tuples
        self.labels = labels
    
    def add_range(self, range_tuple, label):
        assert range_tuple[0] < range_tuple[1], 'Left border not smaller than right'
        self.range_tuples.append(range_tuple)
        self.labels.append(label)
        
    def __getitem__(self, value):
        for t, l in zip(self.range_tuples, self.labels):
            if t[0] <= value < t[1]:
                return l
        return None
    
    def __str__(self):
        s = '----- Range Dict -----\n' 
        for t, l in zip(self.range_tuples, self.labels):
            s += '{:d} - {:d} \t: {:s}\n'.format(t[0], t[1], str(l))
        s += '----------------------'
        return s
            
        

class AachenDayNight(data.Dataset):
    
    def __init__(self, data_path, train, resize, train_split=20, overfit=None,
                seed=7,input_types='img', output_types='pose', real=False
                ,transform=identity, semantic_transform=identity, scene='', target_transform=identity, 
                night_augmentation=None, use_stylization=False,
                verbose=False, augmentation_directory='AugmentedNightImages_high_res'):
        
        np.random.seed(seed)
        self.data_path = data_path
        self.input_types = input_types if type(input_types) is list else [input_types]
        self.output_types = output_types if type(output_types) is list else [output_types]
        self.train = train
        self.night_augmentation = night_augmentation
        assert self.night_augmentation in [None, 'only', 'combined'], 'Only None, only combined allowed'
        self.use_stylization = use_stylization
        self.resize = resize
        if verbose:
            if self.night_augmentation == 'combined':
                print('Use augmented images too')
            elif self.only_augmentation == 'only':
                print('Use only augmented images')
        self.train_split = train_split
        self.images = []
        self.train_idcs = []
        self.val_idcs = []
        self.transforms = {
            'img' : transform, 
            'label' : semantic_transform,
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
        poses = []
        index = 0
        for i in range(3, len(lines)):
            l = lines[i]
            q = [float(x) for x in l[2:6]]
            c = [float(x) for x in l[6:9]]
            pose = np.asarray(c+q)
            poses.append(pose)
            #self.img_paths.append(os.path.join('images_upright', l[0]))
            self.images.append(Sample(index, os.path.join(self.data_path, 'images_upright', l[0]), pose))
            index += 1
        poses = np.vstack(poses)
        #del poses
        self.sem_labels = None
        if 'label' in self.input_types or 'label' in self.output_types:
            for i in range(len(self.images)):
                img_path = self.images[i].path
                sem_path = os.path.join(self.data_path, 'sem_labels', ntpath.basename(img_path).replace('.jpg','.png'))
                if os.path.isfile(sem_path):
                    self.images[i].semantic = sem_path
                else:
                    raise AssertionError('WARNING: SEMANTIC LABEL NOT FOUND')
            if verbose:
                print('All semantic labels successfully loaded')
        #self.night_img_paths = None
        if self.night_augmentation is not None:
            self.night_img_paths = []
            for i in range(len(self.images)):
                img_name = ntpath.basename(self.images[i].path).replace('.jpg','.png')
                aug_path = os.path.join(self.data_path, augmentation_directory, img_name)
                assert os.path.isfile(aug_path), 'Augmented file not found'
                self.images[i].nighttime = aug_path
                
        if self.use_stylization:
            for i in range(len(self.images)):
                img_dir, base_name = os.path.split(self.images[i].path)
                base_name = base_name.split('.')
                base_name = base_name[0] + '-stylized-'+available_styles[i % len(available_styles)] + '.' + base_name[1]
                file = os.path.join(img_dir, 'stylized', base_name)
                assert os.path.isfile(file), 'Stylized image not found: {:s}'.format(file)
                self.images[i].stylized = file
                
        for i in range(len(self.images)):
            if i % self.train_split == 0:
                self.val_idcs.append(i)
            else:
                self.train_idcs.append(i)
        if verbose:
            print('Processed {:d} images'.format(len(self.images)))
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
            for i in range(len(self.images)):
                self.images[i].point_cloud = []
            if verbose:
                print('Created dict for img-point correspondances')
            for i in range(num_sifts):
                if i % 500000 == 0:
                    print('%d/%d'%(i,num_sifts))
                line = sift_lines[i]
                num_imgs = int(line[6])
                for j in range(num_imgs):
                    idx = int(line[7+j*4])
                    self.images[idx].point_cloud.append(i)
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
        self.images = [self.images[i] for i in (self.train_idcs if self.train else self.val_idcs)]
        """self.poses = self.poses[self.train_idcs if self.train else self.val_idcs]
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
        """
                
        pose_stats_filename = os.path.join(self.data_path, 'pose_stats.txt')
        if train and not real:
            # optionally, use the ps dictionary to calc stats
            mean_t = poses[:, :3].mean(axis=0)
            std_t = poses[:, :3].std(axis=0)
            print('Stats {:s} saved to {:s}'.format('{:s}/{:s}'.format(str(mean_t), str(std_t)), pose_stats_filename))
            np.savetxt(
                pose_stats_filename, np.vstack(
                    (mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
            print('Use stats: {:s}/{:s}'.format(str(mean_t), str(std_t)))
        poses = process_poses_quaternion(poses[:,:3], poses[:,3:], mean_t, std_t, np.eye(3), np.zeros(3), np.ones(1))
        for i in range(len(self.images)):
            j = self.train_idcs[i] if self.train else self.val_idcs[i]
            self.images[i].pose = poses[j]
        if verbose:
            print('Process poses: %s'%str(poses.shape))
            
        if resize is not None:
            self.__resize__(resolution=resize)
            
        self.types = RangeCheck()
        self.types.add_range((0, len(self.images)), 1 if self.night_augmentation == 'only' else 0)
        index = len(self.images)
        if self.night_augmentation == 'combined':
            self.types.add_range((index, index+len(self.images)), 1) 
            index += len(self.images)
        if self.use_stylization:
            self.types.add_range((index, index+len(self.images)), 2)
        #print(str(self.types))
            
    def __len__(self):
        L = len(self.images)
        if self.night_augmentation == 'combined':
            L += len(self.images)
        if self.use_stylization:
            L += len(self.images) #*num_styles
        return L
    
    def __img_resize__(self, img_path, transform, resolution, semantic=False):
        direct, name = os.path.split(img_path)
        new_dir = os.path.join(direct, 'resized')
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        new_name = 'resized_{}px_{:s}'.format(resolution, name)
        new_path = os.path.join(new_dir, new_name)
        if not os.path.isfile(new_path):
            if semantic:
                img = Image.open(img_path)
            else:
                img = load_image(img_path)
            try:
                img = transform(img)
            except:
                print('Problem with image {}'.format(img_path))
                exit()
            img.save(new_path)
        return new_path
    
    def __resize__(self, resolution=256):
        import torchvision.transforms as tf
        resize = tf.Compose([
            tf.Resize(resolution), 
            tf.CenterCrop(resolution)
        ])
        sm_resize = tf.Compose([
            tf.Resize(resolution, 0),
            tf.CenterCrop(resolution),
            tf.Lambda(lambda pic: torch.from_numpy(np.array(pic, np.int32, copy=False))),
            tf.ToPILImage()
        ])
        semantic = 'label' in self.input_types or 'label' in self.output_types
        L = len(self.images)
        for index in tqdm.tqdm(range(len(self.images)), total=L, desc='Resize', leave=False):
            img = self.images[index]
            img.path = self.__img_resize__(img.path, resize, resolution=resolution)
            if semantic:
                img.semantic = self.__img_resize__(img.semantic, sm_resize, resolution=resolution, semantic=True)
            if self.night_augmentation is not None:
                img.nighttime = self.__img_resize__(img.nighttime, resize, resolution=resolution)
            if self.use_stylization:
                img.stylized = self.__img_resize__(img.stylized, resize, resolution=resolution)
            self.images[index] = img
        
            
            
    def __getitem__(self, index):
        """
        Augmentation index defines which type of images is returned:
        0: standard daytime image
        1: artificial nighttime images
        2: stylized images
        """
        augmentation_index = self.types[index]
        inps = []
        outs = []
        img_sample = self.images[index % len(self.images)]
        for inp in self.input_types:
            if inp == 'img':
                if augmentation_index == 0:
                    img_path = img_sample.path
                elif augmentation_index == 1:
                    img_path = img_sample.nighttime
                elif augmentation_index == 2:
                    img_path = img_sample.stylized
                else:
                    raise NotImplementedError('Whats going on? Augmentation index: {:s}\t Index: {:d}'.format(str(augmentation_index), index))
                img = load_image(img_path)
                img = None if img is None else self.transforms['img'](img)
                inps.append(img)
            elif inp == 'label':
                img = single_channel_loader(img_sample.semantic)
                img = None if img is None else self.transforms['label'](img)
                inps.append(img)
            elif inp == 'point_cloud':
                pc = self.sift_points[img_sample.point_cloud]
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
                p = img_sample.pose
                p = None if p is None else self.transforms['pose'](p).squeeze()
                outs.append(p)
            elif out == 'label':
                img = single_channel_loader(img_sample.semantic)
                img = None if img is None else self.transforms['label'](img)
                outs.append(img)
            else:
                raise NotImplementedError('Not implemented yet')
        return inps[0] if len(inps) <= 1 else tuple(inps), outs[0] if len(outs) <= 1 else outs
            
if __name__ == '__main__':
    from torchvision import transforms, utils
    import torch
    import matplotlib.pyplot as plt
    test = AachenDayNight('../data/deepslam_data/AachenDayNight/', False, 224,
                          use_stylization=True, night_augmentation='combined', 
                         output_types=['pose', 'label'])
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    L = len(test)
    print('Dataset has {:d} entries'.format(L))
    for i in tqdm.tqdm(range(len(test)), total=L, desc='All there?', leave=False):
        test[i]
    imgs = torch.stack([tf(test[i][0]) for i in range(L//3-7, L//3+7)] 
                       + [tf(test[i][0]) for i in range(2*(L//3)-7, 2*(L//3)+7)])
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