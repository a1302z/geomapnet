from __future__ import division
import sys
sys.path.insert(0, '../')
from torch.utils import data
import numpy as np
import ntpath
import random
from common.pose_utils import process_poses_quaternion
import transforms3d.quaternions as txq
from dataset_loaders.utils import load_image
import os

def identity(x):
    return x

class camerapoint:
    def __init__(self, rotation_matrix, translation):
        self.rotation_matrix = rotation_matrix
        self.translation = translation
        self.path = None
        self.pose = None
        
    def __str__(self):
        s = ''
        if self.path is not None:
            s+='Path: '+self.path+'\n'
        s += 'Rotation:\n'+str(self.rotation_matrix)+'\nPosition:\n'+str(self.translation)
        return s
    
    def to_pose_format(self):
        s = ""
        s+= str(self.path)+" "
        for f in self.translation:
            s+= str(f)+" "
        q = quat.mat2quat(self.rotation_matrix)
        for f in q:
            s+=str(f)+" "
        return s
        
    
    def add_path(self, path):
        self.path = path
    
    def set_pose(self,pose):
        self.pose=pose


class AachenDayNight(data.Dataset):
    
    def __init__(self, data_path, train, train_split=0.7, overfit=None,
                seed=7,input_types='image', output_types='pose', real=False
                ,transform=identity, scene='', target_transform=identity):
        """
        seed=7, overfit=None, reduce_data=True,
        transform=identity, semantic_transform=identity, 
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
            #'label' : semantic_transform,
            #'depth' : transform,
            'pose' : target_transform
        }
        
        ##TODO: remove hardcoded
        filename = 'aachen_cvpr2018_db.out'
        f = open(os.path.join(data_path,filename), 'r')
        lines = f.readlines()
        num_cams = int(lines[1].split(' ')[0])
        last_line = num_cams*5+1
        lines = lines[:last_line+1]
        lines = [x.strip().split(' ') for x in lines]
        lines[2:] = [[float(y) for y in x] for x in lines[2:]]
        if print_debugging_info:
            print('Read %d lines'%len(lines))
        self.points = {}
        for i in range(num_cams):
            #5 Lines per Object:
            #print('New object')
            x0 = i*5+2   #first line of descriptor 
            x1 = i*5+6   #last line of descriptor
            cam_params = np.asarray(lines[x0])
            rot_x = np.asarray(lines[x0+1])
            rot_y = np.asarray(lines[x0+2])
            rot_z = np.asarray(lines[x0+3])
            pos = np.asarray(lines[x0+4])
            point = camerapoint(np.asarray([rot_x,rot_y,rot_z]), pos)
            self.points[i] = point
        if print_debugging_info:
            print('Totals to %d points'%len(self.points))
            
        #TODO: Hardcoded
        list_file = open(os.path.join(data_path,'aachen_cvpr2018_db.list.txt'),'r')
        name_lines = [ntpath.basename(x.strip()) for x in list_file.readlines()]
        for i, line in enumerate(name_lines):
            self.points[i].add_path(line)
        assert len(self.points) == len(name_lines), 'Not same number of points and images'
        self.train_keys = None
        self.test_keys = None
        files_exist = os.path.isfile('train_'+str(self.train_split)+'.txt')
        if not files_exist:
            self._split_dataset(train_split, data_path)
        else:
            train_file = open(data_path+'train_'+str(self.train_split)+'.txt', 'r')
            train_lines = train_file.readlines()
            train_split_read = float(train_lines[0])
            #If train split of file and given trainsplit do not match create new split files
            if abs(train_split-train_split_read) > 1e-3:
                if self.train:
                    print('Train split and split files do not match -> Create new split files')
                    self._split_dataset(train_split, data_path)
                else:
                    raise AssertionError('Not the same train split used as in training')
                train_file.close()
            else:
                self.train_keys = [int(x) for x in train_lines[1:]]
                train_file.close()
                test_file = open(data_path+'test_'+str(train_split)+'.txt', 'r')
                test_lines = test_file.readlines()
                self.test_keys = [int(x) for x in test_lines]
                test_file.close()
        if print_debugging_info:
            print('%d training points\n%d test points'%(len(self.train_keys), len(self.test_keys)))
        
        if overfit is not None:
            self.train_keys = self.train_keys[:overfit]
            self.test_keys = self.train_keys
        
        pose_stats_filename = os.path.join(self.data_path, 'pose_stats.txt')
        if train and not real:
            # optionally, use the ps dictionary to calc stats
            pos_list = []
            for i in self.train_keys:
                pos_list.append(self.points[i].translation)
            pos_list = np.asarray(pos_list)
            mean_t = pos_list.mean(axis=0)
            std_t = pos_list.std(axis=0)
            np.savetxt(
                pose_stats_filename, np.vstack(
                    (mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        indexes = self.train_keys if self.train else self.test_keys
        for i in indexes:
            p = self.points[i]
            q = txq.mat2quat(p.rotation_matrix)
            pose = process_poses_quaternion(np.asarray([p.translation]), np.asarray([q]), mean_t, std_t, np.eye(3), np.zeros(3), np.ones(1))
            p.set_pose(pose)
        
        
    def _split_dataset(self, train_split, data_path):
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
    
    def __getitem__(self, index):
        i = self.train_keys[index] if self.train else self.test_keys[index]
        point = self.points[i]
        inps = []
        outs = []
        for inp in self.input_types:
            if inp == 'image':
                img_path = os.path.join(self.data_path,'images_upright/db',point.path)
                img = load_image(img_path)
                img = None if img is None else self.transforms['img'](img)
                inps.append(img)
            else:
                raise NotImplementedError('Not implemented yet')
        for out in self.output_types:
            if out == 'pose':
                p = point.pose
                p = None if p is None else self.transforms['pose'](p)
                outs.append(p)
            else:
                raise NotImplementedError('Not implemented yet')
                
        
        
        return inps[0], outs[0]
        
        
    
    def __len__(self):
        return len(self.train_keys) if self.train else len(self.test_keys)

    
def main():
    loader = AachenDayNight('../data/deepslam_data/AachenDayNight/', True)
    loader = AachenDayNight('../data/deepslam_data/AachenDayNight/', False)
    loader = AachenDayNight('../data/deepslam_data/AachenDayNight/', True, train_split=0.8)
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