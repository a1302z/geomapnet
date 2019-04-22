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

def identity(x):
    return x

class camerapoint:
    def __init__(self, position = np.zeros(3), rotation = np.zeros(4), img_path = None):
        self.position = position
        self.rotation = rotation
        self.img_path = img_path
        self.pose = None
        self.sem_path = None
        
    def set_pose(self, pose):
        self.pose = pose
        
    def set_sem_path(self, sem_path):
        self.sem_path = sem_path
        
    def __str__(self):
        return self.img_path+str(self.pose)#+str(self.rotation)+str(self.position)


class AachenDayNight(data.Dataset):
    
    def __init__(self, data_path, train, train_split=6, overfit=None,
                seed=7,input_types='img', output_types='pose', real=False
                ,transform=identity, semantic_transform=identity, scene='', target_transform=identity):
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
        self.train_split = train_split
        print_debugging_info = True
        self.transforms = {
            'img' : transform, 
            #'right' : transform,
            #'label_colorized' : semantic_colorized_transform,
            'label' : semantic_transform,
            #'depth' : transform,
            'pose' : target_transform
        }
        
        ##TODO: remove hardcoded
        if overfit:
            print('Overfitting to %d points'%overfit)
        filename = 'train_step%d.txt'%train_split if self.train or overfit else 'val_step%d.txt'%train_split
        f = open(os.path.join(data_path,filename), 'r')
        lines = f.readlines()
        lines[3:] = [x.strip().split(' ') for x in lines[3:]]
        """
        Format is 
        <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
        """

        self.points = []
        for i in range(len(lines)-3):
            l = lines[i+3]
            q = np.asarray([float(x) for x in l[1:5]])
            c = np.asarray([float(x) for x in l[5:8]])
            p = camerapoint(img_path = l[0], rotation=q, position=c)
            self.points.append(p)
            
        if print_debugging_info:
            print('Totals to %d points'%len(self.points))
            
        
        
        if overfit is not None:
            self.points = self.points[overfit]
        
        pose_stats_filename = os.path.join(self.data_path, 'pose_stats.txt')
        if train and not real:
            # optionally, use the ps dictionary to calc stats
            pos_list = []
            for p in self.points:
                pos = p.position
                pos_list.append(pos)
            pos_list = np.asarray(pos_list)
            mean_t = pos_list.mean(axis=0)
            print('Mean: %s'%str(mean_t))
            std_t = pos_list.std(axis=0)
            np.savetxt(
                pose_stats_filename, np.vstack(
                    (mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        for p in self.points:
            pose = process_poses_quaternion(np.asarray([p.position]), np.asarray([p.rotation]), mean_t, std_t, np.eye(3), np.zeros(3), np.ones(1))
            p.set_pose(pose)
            
        self.sem_labels = None
        if 'label' in self.input_types or 'label' in self.output_types:
            for p in self.points:
                sem_path = os.path.join(self.data_path, 'sem_labels', ntpath.basename(p.img_path).replace('.jpg','.png'))
                if os.path.isfile(sem_path):
                    p.set_sem_path(sem_path)
                else:
                    raise AssertionError('WARNING: SEMANTIC LABEL NOT FOUND')
            print('All semantic labels successfully loaded')
        
    """    
    def _split_dataset(self, train_split, data_path):
        print('Warning: Create new data split file')
        self.train_keys = random.sample(self.points.keys(), int(train_split*len(self.points.keys())))
        self.test_keys = [x for x in self.points.keys() if x not in self.train_keys]
        train_file = open(data_path+ 'train_'+str(train_split)+'.txt', 'w+')
        train_file.write(str(train_split)+"\n")
        for p in self.train_keys:
            train_file.write(str(p)+'\n')
        train_file.close()
        test_file = open(data_path+'test_'+str(train_split)+'.txt', 'w+')
        for p in self.test_keys:
            test_file.write(str(p)+'\n')
        test_file.close()
    """
    def __getitem__(self, index):
        point = self.points[index]
        inps = []
        outs = []
        for inp in self.input_types:
            if inp == 'img':
                img_path = os.path.join(self.data_path,point.img_path)
                img = load_image(img_path)
                img = None if img is None else self.transforms['img'](img)
                inps.append(img)
            elif inp == 'label':
                img = single_channel_loader(point.sem_path)
                img = None if img is None else self.transforms['label'](img)
                inps.append(img)
            else:
                raise NotImplementedError('Not implemented yet (%s)'%str(inp))
        for out in self.output_types:
            if out == 'pose':
                p = point.pose
                p = None if p is None else self.transforms['pose'](p).squeeze()
                outs.append(p)
            elif out == 'label':
                #print('GET SEMANTIC LAB')
                img = single_channel_loader(point.sem_path)
                #print(img.shape)
                img = None if img is None else self.transforms['label'](img)
                #print(img.shape)
                outs.append(img)
            else:
                raise NotImplementedError('Not implemented yet')
                
        
        
        return inps[0] if len(inps) <= 1 else inps, outs[0] if len(outs) <= 1 else outs
        
        
    
    def __len__(self):
        return len(self.points)
    
    
    def print_point(self, index):
        print(str(self.points[index]))

    
def main():
    loader = AachenDayNight('../data/deepslam_data/AachenDayNight/', True)
    loader = AachenDayNight('../data/deepslam_data/AachenDayNight/', False, input_types=['img', 'label'])
    x = loader[20]
    for i in x:
        print(type(i))
    loader = AachenDayNight('../data/deepslam_data/AachenDayNight/', True)
    loader[100]
    loader[0]
    print(len(loader))
    loader = AachenDayNight('../data/deepslam_data/AachenDayNight/', True)
    x = loader[100]
    print(type(x))
    for i in x:
        print(type(i))
    loader[0]
    print(len(loader))
    
if __name__ == '__main__':
    main()