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


class StylizedCambridge(data.Dataset):
    
    def __init__(self, data_path, train, train_split=0.7, overfit=None, scene='ShopFacade',
                styles = 4, seed=7, transform=identity, target_transform=identity, real=True, **kwargs):
        
        print('Start')
 
        np.random.seed(seed)
        self.data_path = data_path
        self.scene = scene
        self.train = train
        self.transforms = {
            'img' : transform, 
            'pose' : target_transform
        }
        
        
        if self.train or overfit is not None:
            if styles == 0:
                training_file = open(os.path.join(self.data_path, self.scene, 'dataset_train.txt'), 'r')
            else:
                training_file = open(os.path.join(self.data_path, self.scene, 'dataset_train_{}_styles.txt'.format(styles)), 'r')
        else:
            training_file = open(os.path.join(self.data_path, self.scene, 'dataset_test.txt'), 'r')
            
        lines = training_file.readlines()
        lines = [x.strip() for x in lines]
        
        print('Read points')
        
        self.points = []
        for l in lines[3:]:
            ls = l.split(' ')
            pose = [float(x) for x in ls[1:]]
            p = camerapoint(position = pose[:3], rotation = pose[3:], img_path=ls[0])
            self.points.append(p)
        print('Loaded %d points'%len(self.points))
        
        if overfit is not None:
            self.points = self.points[:overfit]
            
        print('Process points')
            
        pose_stats_filename = os.path.join(self.data_path, self.scene, 'pose_stats.txt')
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
            print('Saved pose stats to %s'%pose_stats_filename)
        else:
            print('Using pose stats from {}'.format(pose_stats_filename))
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        for p in self.points:
            pose = process_poses_quaternion(np.asarray([p.position]), np.asarray([p.rotation]), mean_t, std_t, np.eye(3), np.zeros(3), np.ones(1))
            p.set_pose(pose.flatten())
        print('Finish setup')

            
    def __getitem__(self, index):
        point = self.points[index]
        inps = []
        outs = []
        img_path = os.path.join(self.data_path,self.scene,point.img_path)
        img = load_image(img_path)
        img = self.transforms['img'](img)
        p = point.pose
        p = self.transforms['pose'](p)
        
        return img, p
        
        
    
    def __len__(self):
        return len(self.points)
    
    
def main():
    loader = StylizedCambridge('/storage/user/zillera/stylized_localization/', True, scene='ShopFacade224', 
                              styles = 16)
    print(len(loader))
    print(loader[10])
    
if __name__ == '__main__':
    main()