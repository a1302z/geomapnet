from __future__ import division
import sys
sys.path.insert(0, '../')
from torch.utils import data
import numpy as np
from common.pose_utils import process_poses_quaternion
import transforms3d.quaternions as txq
import os
from dataset_loaders.utils import load_image, single_channel_loader


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
        del self.position, self.rotation
        
    def set_sem_path(self, sem_path):
        self.sem_path = sem_path
        
    def __str__(self):
        return self.img_path+' '+str(self.pose)


class Cambridge(data.Dataset):
    
    def __init__(self, data_path, train, train_split=0.7, overfit=None, scene='KingsCollege',
                seed=7,input_types='img', output_types='pose', real=False
                ,transform=identity, semantic_transform=identity, target_transform=identity):
 
        np.random.seed(seed)
        self.data_path = data_path
        self.scene = scene
        self.input_types = input_types if type(input_types) is list else [input_types]
        self.output_types = output_types if type(output_types) is list else [output_types]
        self.train = train
        print_debugging_info = True
        self.transforms = {
            'img' : transform, 
            #'right' : transform,
            #'label_colorized' : semantic_colorized_transform,
            'label' : semantic_transform,
            #'depth' : transform,
            'pose' : target_transform
        }
        
        
        
        if self.train or overfit is not None:
            training_file = open(os.path.join(self.data_path, self.scene, 'dataset_train.txt'), 'r')
        else:
            training_file = open(os.path.join(self.data_path, self.scene, 'dataset_test.txt'), 'r')
            
        lines = training_file.readlines()
        lines = [x.strip() for x in lines]
        
        self.points = []
        for l in lines[3:]:
            ls = l.split(' ')
            pose = [float(x) for x in ls[1:]]
            p = camerapoint(position = pose[:3], rotation = pose[3:], img_path=ls[0])
            self.points.append(p)
        print('Loaded %d points'%len(self.points))
        
        if overfit is not None:
            self.points = self.points[:overfit]
            
        pose_stats_filename = os.path.join(self.data_path, 'pose_stats.txt')
        if train and not real:
            # optionally, use the ps dictionary to calc stats
            pos_list = []
            for i in self.points:
                pos_list.append(i.position)
            pos_list = np.asarray(pos_list)
            mean_t = pos_list.mean(axis=0)
            std_t = pos_list.std(axis=0)
            np.savetxt(
                pose_stats_filename, np.vstack(
                    (mean_t, std_t)), fmt='%8.7f')
            #print('Saved pose stats to %s'%pose_stats_filename)
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        for p in self.points:
            pose = process_poses_quaternion(np.asarray([p.position]), np.asarray([p.rotation]), mean_t, std_t, np.eye(3), np.zeros(3), np.ones(1))
            p.set_pose(pose.flatten())

            
    def __getitem__(self, index):
        point = self.points[index]
        inps = []
        outs = []
        for inp in self.input_types:
            if inp == 'img':
                img_path = os.path.join(self.data_path,self.scene,point.img_path)
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
                p = None if p is None else self.transforms['pose'](p)
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
    
    
def main():
    loader = Cambridge('../data/deepslam_data/CambridgeLandmarks/', True)
    print(len(loader))
    print(loader[10])
    
if __name__ == '__main__':
    main()