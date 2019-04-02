from __future__ import division
import sys
sys.path.insert(0, '../')

from common.pose_utils import process_poses_quaternion
import torch
from dataset_loaders.utils import load_image, pfm_loader, single_channel_loader
from torchvision.datasets.folder import default_loader
from torch.utils import data
import numpy as np
import pandas as pd
import os.path as osp
import os
"""
Adapted DataLoader to DeepLoc dataset (ADL4CV - Project)

Based on 7Scenes DataLoader by NVidia
"""

"""
pytorch data loader for the DeepLoc dataset

"""
def identity(x):
    return x

class DeepLoc(data.Dataset):
    
    LOADERS = {  
            'left' : default_loader,
            'right' : default_loader,
            'label_colorized' : default_loader,
            'label' : single_channel_loader,
            'depth' : pfm_loader
        }

    def _loader(self, name, index):
        """
        Loads an image stored in the internal dataframe or gets an pose from the poses list
        
        :param name : string (valid names: 'left', 'right', 'label_colorized', 'label', 'depth')
            used to select the type of data to load 
        :param index: : int, the index of a certain image
        
        :returns : the loaded data
            if no path is given for the certain image, None is returned
        """
        if name == 'pose':
            return self.poses[index]
        else:
            path = self.df[name].iloc[index]
            if pd.isna(path):
                return None
            else:
                return load_image(path, self.LOADERS[name])
            
    def _loader_transformer(self, name, index):
        """
        Loads and transformes a data of a certain type and of a certian index
        
        :param name : string (valid names: 'left', 'right', 'label_colorized', 'label', 'depth')
            used to select the type of data to load 
        :param index: : int, the index of a certain image
        
        :returns : the loaded and transformed data
            if no path is given for the certain image, None is returned
        """
        loaded_data = self._loader(name, index)
        return None if loaded_data is None else self.transforms[name](loaded_data)
    
    
    def __init__(self, data_path, train, seed=7, real=False, overfit=None, reduce_data=True,
                 transform=identity, semantic_transform=identity, 
                 semantic_colorized_transform=identity, target_transform=identity, 
                 vo_lib='orbslam', scene='',    
                 input_types='left', output_types='pose', concatenate_inputs=False):
        """
        :param data_path: root DeepLoc data directory.
        Usually '../data/deepslam_data/DeepLoc'
        :param train: if True, return the training images. If False, returns the
        testing images
        :param transform: transform to apply to the images
        :param semantic_transform: transform to apply to the semantic labels
        :param semantic_colorized_transform: transform to apply to the colorized semantic labels
        :param target_transform: transform to apply to the poses
        :param real: If True, load poses from SLAM/integration of VO
        :param vo_lib: Library to use for VO (currently only 'dso')
        :param input_types: str or list of str, default = 'left'
            names of the data types which should be returned in the first part of the tuple
        :param output_types_ str or list of str, default = 'pose'
            names of the data types which should be returned in the second part of the tuple
        :param concatenate_inputs: bool, default = False
            indicates whether the inputs should be concatenated to one bigger tensor
            otherwise it is returend as a tuple of tensors
        """
        np.random.seed(seed)
        reduce_data = reduce_data or self.semantic or self.dual_output 
        
        self.transforms = {
            'left' : transform, 
            'right' : transform,
            'label_colorized' : semantic_colorized_transform,
            'label' : semantic_transform,
            'depth' : transform,
            'pose' : target_transform
        }
        
        self.concatenate_inputs = concatenate_inputs
        self.input_types = input_types if type(input_types) is list else [input_types]
        self.output_types = output_types if type(output_types) is list else [output_types]
        
        assert (all(data_type in self.transforms for data_type in self.input_types))
        assert (all(data_type in self.transforms for data_type in self.output_types))
        
        # generate the pathes to the directories contianing images
        base_dir = osp.expanduser(osp.join(data_path, scene))
        data_dir = osp.expanduser(osp.join('..', 'data', 'DeepLoc', scene))
        subset_name = 'train' if train else 'test'
        base_dir_left = osp.join(base_dir, subset_name)
        base_dir_right = osp.join(base_dir, 'DeepLoc_stereo', subset_name)
        img_base_left = osp.join(base_dir_left, 'LeftImages')
        labels_dir = osp.join(base_dir_left, 'labels')
        labels_colorized_dir = osp.join(base_dir_left, 'labels_colorized')

        # generate tuples which contain the name of the image sets and the path to the folder
        image_directories = [
            ('left', img_base_left),
            ('label', labels_dir),
            ('label_colorized', labels_colorized_dir)
        ]
        
        if not {'right', 'depth'}.isdisjoint(self.input_types + self.output_types):
            assert osp.exists(base_dir_right)
            img_base_right = osp.join(base_dir_right, 'RightImages')
            img_base_depth = osp.join(base_dir_right, 'DepthImages')
            image_directories += [
                ('right', img_base_right),
                ('depth', img_base_depth)
            ]
        
        # read the poses from the pose file
        pose_file = osp.join(base_dir, subset_name, 'poses.txt')
        poses_raw = pd.read_csv(pose_file, sep=' ', skiprows=1)
        poses_raw['id'] = poses_raw.img_name.str.extract(r"Image_(?P<id>\d*)\S*?").iloc[:,0].astype(int)
        poses = poses_raw.drop('img_name', axis=1).set_index('id').sort_index()
        
        def get_pathes_dataframe(image_type_name, path):
            """
            get a pandas dataframe containing the image number as index and the image pathes in an column
            
            :param image_type_name: the name the column containing the full pathes of the images
            :param train: if True, return the training images. If False, returns the
            
            :returns: a dataframe with one column
            """
            existing_files = pd.Series(os.listdir(path))
            df_extract = existing_files.str.extract(r"(?P<content>Image_(?P<id>\d*)\S*)", expand=True).dropna()
            pathes = df_extract.content.apply(lambda file_name: osp.join(path, file_name)).values
            index = df_extract.id.astype(int).values
            return pd.DataFrame({image_type_name : pathes}, index=index)
        
        # get dataframes for every type of image in the list image_directories
        df_columns = [get_pathes_dataframe(image_type_name, path) 
                      for image_type_name, path in image_directories]
        
        # concat the images and the poses to an single data frame
        # images with different types but matching ids are going to be in one row
        # missing images will be missing values in the data frame
        self.df = pd.concat([poses] + df_columns, axis=1, sort=True, join='outer')

        # remove the rows with missing images
        if reduce_data:
            self.df.dropna(inplace=True)
            
        # reduce the amout of data if overfit is true
        if overfit is not None:
            print("Shape before overfit: {} ".format(len(self.df)))
            print("Overfit parameter is {}".format(overfit))
            self.df = self.df.iloc[:overfit]
            print("Shape after overfit: {}".format(len(self.df)))

        vo_stats = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train and not real:
            # optionally, use the ps dictionary to calc stats
            mean_t = self.df[list('xyz')].mean().values
            std_t = self.df[list('xyz')].std().values
            np.savetxt(
                pose_stats_filename, np.vstack(
                    (mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion
        self.poses = process_poses_quaternion(xyz_in=self.df[list('xyz')].values,
                                              q_in=self.df[['qw', 'qx', 'qy', 'qz']].values,
                                              mean_t=mean_t,
                                              std_t=std_t,
                                              align_R=vo_stats['R'],
                                              align_t=vo_stats['t'],
                                              align_s=vo_stats['s'])
        

    def __getitem__(self, index):

        inputs = tuple(self._loader_transformer(image_type, index) for image_type in self.input_types)
        outputs = tuple(self._loader_transformer(image_type, index) for image_type in self.output_types)
        
        if len(inputs) == 1:
            inputs = inputs[0]
        elif len(inputs) > 1 and self.concatenate_inputs:
            new_size = (-1, *(inputs[0].size()[-2:]))
            inputs = torch.cat([input.view(*new_size) for input in inputs])
            
        if len(outputs) == 1:
            outputs = outputs[0]
        #print([x.shape for x in outputs])
        #print(np.amax(outputs[1].numpy()))
        #print(self.output_types)
        return inputs, outputs

    
    def __len__(self):
        return len(self.df)


def main():
    """
    visualizes the dataset
    """
    from common.vis_utils import show_batch, show_stereo_batch
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    mode = int(sys.argv[1])
    num_workers = 6
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])         
            
    concatenate_inputs = (mode == 3)

    if mode == 0:
        input_types = 'left'
    elif mode == 1 and not dual_output:
        input_types= 'depth'
    elif mode == 2 and not dual_output:
        input_types = ['left', 'depth']
    elif mode == 3:
        input_types = ['left', 'label_colorized']
           
    dset = DeepLoc('../data/deepslam_data/DeepLoc', True, transform,
                   input_types=input_types, output_types=[], concatenate_inputs=concatenate_inputs)
                 
    print('Loaded DeepLoc, length = {:d}'.format(len(dset)))

    data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
                                  num_workers=num_workers)

    batch_count = 0
    N = 2
    for batch in data_loader:
        print('Minibatch {:d}'.format(batch_count))
        if mode < 2:
            show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
        elif mode == 2:
            lb = make_grid(batch[0][0], nrow=1, padding=25, normalize=True)
            rb = make_grid(batch[0][1], nrow=1, padding=25, normalize=True)
            show_stereo_batch(lb, rb)
        elif mode == 3:
            print(len(batch))
            for elm in batch:
                print(elm.shape)
            lb = make_grid(batch[0][0][:3], nrow=1, padding=25, normalize=True)
            rb = make_grid(batch[0][0][3:], nrow=1, padding=25, normalize=True)
            show_stereo_batch(lb, rb)


        batch_count += 1
        if batch_count >= N:
            break


if __name__ == '__main__':
    main()
