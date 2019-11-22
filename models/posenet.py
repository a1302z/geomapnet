import sys
sys.path.insert(0, '../')
import os
import numpy as np
import torch.nn.init
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import models.pointnet as pointnet
"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

"""
implementation of PoseNet and MapNet networks
"""

os.environ['TORCH_MODEL_ZOO'] = os.path.join('..', 'data', 'models')

# def trace_hook(m, g_in, g_out):
#  for idx,g in enumerate(g_in):
#    g = g.cpu().data.numpy()
#    if np.isnan(g).any():
#      set_trace()
#  return None


def filter_hook(m, g_in, g_out):
    g_filtered = []
    for g in g_in:
        g = g.clone()
        g[g != g] = 0
        g_filtered.append(g)
    return tuple(g_filtered)


class PoseNet(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=2048, filter_nans=False, freeze_feature_extraction=False,
                 activation_function=F.relu, set_base_poses=None):
        super(PoseNet, self).__init__()
        print('Freeze features: {}\nActivation function: {}\nSet base poses: {}'.format(freeze_feature_extraction, activation_function, set_base_poses))
        self.droprate = droprate
        self.activation_function=activation_function

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)
        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            init_modules = [
                self.feature_extractor.fc,
                self.fc_xyz,
                self.fc_wpqr]
        else:
            init_modules = self.modules()
        if freeze_feature_extraction:
            for m in self.feature_extractor.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    print('Freeze grad for %s'%str(m))
                    m.weight.requires_grad = False

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                #print('Unfreeze grad for %s'%str(m))
                if freeze_feature_extraction:
                    m.weight.requires_grad = False
                else:
                    m.weight.requires_grad = True
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        if set_base_poses is not None:
            self.fc_xyz.weight.data = set_base_poses
            self.fc_xyz.weight.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.activation_function(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)
                    
                    
class PoseNetV2(nn.Module):
    """
    Extends the input of posenet by one channel.
    The new channel is initialized by the average of the weights of the other channels.
    """
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=2048, filter_nans=False):
        super(PoseNetV2, self).__init__()
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor

        first_layer = self.feature_extractor.conv1
        new_first_layer = nn.Conv2d(4, 
                             out_channels=first_layer.out_channels, 
                             kernel_size=first_layer.kernel_size, 
                             stride=first_layer.stride, 
                             padding=first_layer.padding,
                             bias=first_layer.bias)
          
        #trained_kernel = first_layer.weight # this is the pretrained RGB kernel
        #new_first_layer.weight = nn.Parameter(torch.Tensor(trained_kernel.detach().numpy()))
        #new_first_layer.weight = nn.Parameter(trained_kernel)
        
        trained_kernel = first_layer.weight # this is the pretrained RGB kernel
        new_first_layer.weight = nn.Parameter(torch.cat((
                trained_kernel, 
                torch.mean(trained_kernel, dim=1, keepdim=True)
            ), dim=1))
        
        self.feature_extractor.conv1 = new_first_layer
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
        self.activation_function = activation_function
        
        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)
        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            init_modules = [
                self.feature_extractor.fc,
                self.fc_xyz,
                self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)

class MapNet(nn.Module):
    """
    Implements the MapNet model (green block in Fig. 2 of paper)
    """

    def __init__(self, mapnet):
        """
        :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
        of paper). Not to be confused with MapNet, the model!
        """
        super(MapNet, self).__init__()
        self.mapnet = mapnet

    def forward(self, x):
        """
        :param x: image blob (N x T x C x H x W)
        :return: pose outputs
         (N x T x 6)
        """
        s = x.size()
        x = x.view(-1, *s[2:])
        poses = self.mapnet(x)  # Shape(30, 6)
        poses = poses.view(s[0], s[1], -1)  # Shape(10, 3, 6)
        return poses
    
    def __feature_vector__(self, x):
        s = x.size()
        x = x.view(-1, 3, s[3], s[4])
        x = self.mapnet.feature_extractor(x)
        return x
        


class SemanticMapNet(nn.Module):
    """
    Adaption of MapNet with additional semantic input
    """

    def __init__(self, mapnet):
        super(SemanticMapNet, self).__init__()
        self.mapnet = mapnet
        self.linear1 = nn.Linear(12, 6)

    def forward(self, x):
        """
        :param x: image blob (N x T x 2 * C x H x W)
        :return: pose outputs
         (N x T x 6)
        """
        s = x.size()  # should be (NxTx2xCxHxW) is [20, 3, 6, 256, 455]
        x_c = x[:, :, 0:3, :, :].view(-1, 3, 256, 455)
        x_s = x[:, :, 3:6, :, :].view(-1, 3, 256, 455)
        poses_c = self.mapnet(x_c)
        poses_s = self.mapnet(x_s)
        poses = torch.cat((poses_c, poses_s), -1)
        poses = self.linear1(poses)
        poses = poses.view(s[0], s[1], -1)
        return poses


class SemanticMapNetV2(nn.Module):
    """
    Adaption of MapNet with additional semantic input
    """

    def __init__(self, mapnet_images, mapnet_semantics):
        super(SemanticMapNetV2, self).__init__()
        self.mapnet_images = mapnet_images
        self.mapnet_semantics = mapnet_semantics
        self.linear1 = nn.Linear(12, 6)

    def forward(self, x):
        """
        :param x: image blob (N x T x 2 * C x H x W)
        :return: pose outputs
         (N x T x 6)
        """
        s = x.size()  # should be (NxTx2xCxHxW) is [20, 3, 6, 256, 455]
        x_c = x[:, :, 0:3, :, :].view(-1, 3, 256, 455)
        x_s = x[:, :, 3:6, :, :].view(-1, 3, 256, 455)
        poses_c = self.mapnet_images(x_c)
        poses_s = self.mapnet_semantics(x_s)
        poses = torch.cat((poses_c, poses_s), -1)
        poses = self.linear1(poses)
        poses = poses.view(s[0], s[1], -1)
        return poses


class SemanticMapNetV3(nn.Module):
    """
    Adaption of MapNet with additional semantic input
    """

    def __init__(self, feature_extractor_img, feature_extractor_sem,
                 droprate=0.5, pretrained=True, feat_dim=2048, filter_nans=False):
        super(SemanticMapNetV3, self).__init__()
        self.droprate = droprate
        self.feature_extractor_img = feature_extractor_img
        self.feature_extractor_sem = feature_extractor_sem
        # replace the last FC layer in feature extractor
        self.feature_extractor_img.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor_sem.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor_img.fc.in_features
        self.feature_extractor_img.fc = nn.Linear(fe_out_planes, feat_dim)
        self.feature_extractor_sem.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz = nn.Linear(feat_dim * 2, 3)
        self.fc_wpqr = nn.Linear(feat_dim * 2, 3)
        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            init_modules = [
                self.feature_extractor_img.fc,
                self.feature_extractor_sem.fc,
                self.fc_xyz,
                self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        :param x: image blob (N x T x 2 * C x H x W)
        :return: pose outputs
         (N x T x 6)
        """
        s = x.size()
        x_c = x[:, :, 0:3, :, :].view(-1, 3, 256, 455)
        x_s = x[:, :, 3:6, :, :].view(-1, 3, 256, 455)
        x_c = self.feature_extractor_img(x_c)
        x_s = self.feature_extractor_sem(x_s)
        x_c = F.relu(x_c)
        x_s = F.relu(x_s)
        if self.droprate > 0:
            x_c = F.dropout(x_c, p=self.droprate, training=self.training)
            x_s = F.dropout(x_s, p=self.droprate, training=self.training)

        x = torch.cat((x_c, x_s), 1)  # Shape(30, 4096)
        
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        x = torch.cat((xyz, wpqr), 1)

        poses = x
        poses = poses.view(s[0], s[1], -1)  # Shape(30, 6, 1)
        return poses
        
class SemanticOutput(nn.Module):
    """
    SemanticOutput model to only learn semantic segmentation
    
    The padding of the upconvolutions is fixed to a certain input and output size of the Images.
    """
    def __init__(self, posenet, classes=3, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False, input_size=(224,224)):
        super(SemanticOutput, self).__init__()
        self.input_size=input_size
        self.posenet = posenet
        self.semantic_layer0 = nn.ConvTranspose2d(64, classes, kernel_size=(6,5), stride=4, padding=1, bias=False)
        self.semantic_layer1 = nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.semantic_layer2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.semantic_layer3 = nn.ConvTranspose2d(256, 128, stride=2, kernel_size=(4,3), padding=1)
        self.semantic_layer4 = nn.ConvTranspose2d(512, 256, stride=2, kernel_size=(4,3), padding=1)
        
        
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = self.posenet.feature_extractor
        #self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        
    def forward(self, x):
        s = x.size()
        #print(s)
        x = x.view(-1, 3, self.input_size[0], self.input_size[1])
        x0 = x
        s1 = x.size()
        #print(s1)
        x = self.posenet.feature_extractor.conv1(x)
        x = self.posenet.feature_extractor.bn1(x)
        x = self.posenet.feature_extractor.relu(x)
        x = self.posenet.feature_extractor.maxpool(x)
        x1 = x
        s2 = x.size()
        #print(s2)
        x = self.posenet.feature_extractor.layer1(x)
        x2 = x
        s3 = x.size()
        #print(s3)
        x = self.posenet.feature_extractor.layer2(x)
        x3 = x
        s4 = x.size()
        #print(s4)
        x = self.posenet.feature_extractor.layer3(x)
        x4 = x
        s5 = x.size()
        #print(s5)
        x = self.posenet.feature_extractor.layer4(x)
        
        semantic = self.semantic_layer4(x)
        semantic = nn.functional.interpolate(semantic, size=s5[2:4])
        semantic = self.semantic_layer3(semantic + x4)
        semantic = nn.functional.interpolate(semantic, size=s4[2:4])
        #print(semantic.size())
        semantic = self.semantic_layer2(semantic + x3)
        semantic = nn.functional.interpolate(semantic, size=s3[2:4])
        #print(semantic.size())
        semantic = self.semantic_layer1(semantic + x2)
        semantic = nn.functional.interpolate(semantic, size=s2[2:4])
        #print(semantic.size())
        semantic = self.semantic_layer0(semantic + x1)
        semantic = nn.functional.interpolate(semantic, size=s1[2:4])
        semantic.view(s[0],s[1],-1)
        #print(semantic.size())
        #print('Forward pass completed')

        return semantic
    

class MultiTask(nn.Module):
    """
    Multi-task model to learn poses and semantic segmentation

    The padding of the upconvolutions is fixed to a certain input and output size of the Images.
    """
    def __init__(self, posenet, classes=3, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False, input_size=(224,224), set_base_poses=None, freeze_feature_extraction=False):
        super(MultiTask, self).__init__()
        self.posenet = posenet
        """
        TODO Experiment with model architecture
        Inspired by U-Net
        """
        self.input_size = input_size
        self.semantic_layer0 = nn.ConvTranspose2d(64, classes, kernel_size=(6,5), stride=4, padding=1, bias=False)
        self.semantic_layer1 = nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.semantic_layer2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.semantic_layer3 = nn.ConvTranspose2d(256, 128, stride=2, kernel_size=(4,3), padding=1)
        self.semantic_layer4 = nn.ConvTranspose2d(512, 256, stride=2, kernel_size=(4,3), padding=1)
        
        
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = self.posenet.feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz  = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)
        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        """
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        """
                    
        
        if freeze_feature_extraction:
            for m in self.feature_extractor.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    print('Freeze grad for %s'%str(m))
                    m.weight.requires_grad = False
                    
        if set_base_poses is not None:
            self.fc_xyz.weight.data = set_base_poses
            self.fc_xyz.weight.requires_grad = False
            #print('Set base poses to %s'%str(set_base_poses))
        
                    
    def __feature_vector__(self, x):
        s = x.size()
        x = x.view(-1, 3, self.input_size[0], self.input_size[1])
        x0 = x

        s1 = x.size()
        #print(s1)
        x = self.posenet.feature_extractor.conv1(x)
        x = self.posenet.feature_extractor.bn1(x)
        x = self.posenet.feature_extractor.relu(x)
        x = self.posenet.feature_extractor.maxpool(x)
        x1 = x
        s2 = x.size()
        #print(s2)
        x = self.posenet.feature_extractor.layer1(x)
        x2 = x
        s3 = x.size()
        #print(s3)
        x = self.posenet.feature_extractor.layer2(x)
        x3 = x
        s4 = x.size()
        #print(s4)
        x = self.posenet.feature_extractor.layer3(x)
        x4 = x
        s5 = x.size()
        #print(s5)
        x = self.posenet.feature_extractor.layer4(x)
        
        semantic = self.semantic_layer4(x)
        semantic = nn.functional.interpolate(semantic, size=s5[2:4])
        semantic = self.semantic_layer3(semantic + x4)
        semantic = nn.functional.interpolate(semantic, size=s4[2:4])
        #print(semantic.size())
        semantic = self.semantic_layer2(semantic + x3)
        semantic = nn.functional.interpolate(semantic, size=s3[2:4])
        #print(semantic.size())
        semantic = self.semantic_layer1(semantic + x2)
        semantic = nn.functional.interpolate(semantic, size=s2[2:4])
        #print(semantic.size())
        semantic = self.semantic_layer0(semantic + x1)
        semantic = nn.functional.interpolate(semantic, size=s1[2:4])
        semantic.view(s[0],s[1],-1)
        #print(semantic.size())
        #print('Forward pass completed')
        
        poses = self.posenet.feature_extractor.avgpool(x)
        poses = poses.view(poses.size(0), -1)
        poses = self.posenet.feature_extractor.fc(poses)
        poses = F.relu(poses)
        if self.droprate > 0:
            poses = F.dropout(poses, p=self.droprate, training=self.training)
        return poses, semantic
        
        
    def forward(self, x):
        s = x.size()
        poses, semantic = self.__feature_vector__(x)
        xyz  = self.fc_xyz(poses)
        wpqr = self.fc_wpqr(poses)
        poses = torch.cat((xyz, wpqr), 1)
        poses = poses.view(s[0], s[1], -1) #Shape(10, 3, 6)
        
        return (poses, semantic)
    
    
    

class MultiTask3D(nn.Module):
    """
    Multi-task model to learn poses and semantic segmentation

    The padding of the upconvolutions is fixed to a certain input and output size of the Images.
    """
    def __init__(self, posenet, classes=3, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False, input_size=(224,224), set_base_poses=None, freeze_feature_extraction=False):
        super(MultiTask3D, self).__init__()
        self.posenet = posenet
        self.pointnet_feat = pointnet.PointNetfeat()
        """
        TODO Experiment with model architecture
        Inspired by U-Net
        """
        self.input_size = input_size
        self.semantic_layer0 = nn.ConvTranspose2d(64, classes, kernel_size=(6,5), stride=4, padding=1, bias=False)
        self.semantic_layer1 = nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.semantic_layer2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.semantic_layer3 = nn.ConvTranspose2d(256, 128, stride=2, kernel_size=(4,3), padding=1)
        self.semantic_layer4 = nn.ConvTranspose2d(512, 256, stride=2, kernel_size=(4,3), padding=1)
        
        
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = self.posenet.feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
        #1024 of point net
        self.fc_xyz  = nn.Linear(1024+feat_dim, 3)
        self.fc_wpqr = nn.Linear(1024+feat_dim, 3)
        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        """
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        """
                    
        
        if freeze_feature_extraction:
            for m in self.feature_extractor.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    print('Freeze grad for %s'%str(m))
                    m.weight.requires_grad = False
                    
        if set_base_poses is not None:
            self.fc_xyz.weight.data = set_base_poses
            self.fc_xyz.weight.requires_grad = False
            #print('Set base poses to %s'%str(set_base_poses))
        
                    
    def __feature_vector__(self, x):
        """
        x[0] = image
        x[1] = point cloud
        """
        p = x[1].squeeze(0)
        x = x[0]
        s = x.size()
        x = x.view(-1, 3, self.input_size[0], self.input_size[1])
        x0 = x

        s1 = x.size()
        #print(s1)
        x = self.posenet.feature_extractor.conv1(x)
        x = self.posenet.feature_extractor.bn1(x)
        x = self.posenet.feature_extractor.relu(x)
        x = self.posenet.feature_extractor.maxpool(x)
        x1 = x
        s2 = x.size()
        #print(s2)
        x = self.posenet.feature_extractor.layer1(x)
        x2 = x
        s3 = x.size()
        #print(s3)
        x = self.posenet.feature_extractor.layer2(x)
        x3 = x
        s4 = x.size()
        #print(s4)
        x = self.posenet.feature_extractor.layer3(x)
        x4 = x
        s5 = x.size()
        #print(s5)
        x = self.posenet.feature_extractor.layer4(x)
        
        semantic = self.semantic_layer4(x)
        semantic = nn.functional.interpolate(semantic, size=s5[2:4])
        semantic = self.semantic_layer3(semantic + x4)
        semantic = nn.functional.interpolate(semantic, size=s4[2:4])
        #print(semantic.size())
        semantic = self.semantic_layer2(semantic + x3)
        semantic = nn.functional.interpolate(semantic, size=s3[2:4])
        #print(semantic.size())
        semantic = self.semantic_layer1(semantic + x2)
        semantic = nn.functional.interpolate(semantic, size=s2[2:4])
        #print(semantic.size())
        semantic = self.semantic_layer0(semantic + x1)
        semantic = nn.functional.interpolate(semantic, size=s1[2:4])
        semantic.view(s[0],s[1],-1)
        #print(semantic.size())
        #print('Forward pass completed')
        
        poses = self.posenet.feature_extractor.avgpool(x)
        poses = poses.view(poses.size(0), -1)
        poses = self.posenet.feature_extractor.fc(poses)
        poses = F.relu(poses)
        if self.droprate > 0:
            poses = F.dropout(poses, p=self.droprate, training=self.training)
        
        point_feat, trans, trans_feat = self.pointnet_feat(p) #(x (-1,1024), trans, trans_feat)
        #print(poses.size())
        #print(point_feat.size())
        poses = torch.cat((poses, point_feat), dim=1)
        return poses, semantic
        
        
    def forward(self, x):
        s = x[0].size()
        poses, semantic = self.__feature_vector__(x)
        xyz  = self.fc_xyz(poses)
        wpqr = self.fc_wpqr(poses)
        poses = torch.cat((xyz, wpqr), 1)
        poses = poses.view(s[0], s[1], -1) #Shape(10, 3, 6)
        #print('Output poses.shape: %s'%str(poses.shape))
        
        return (poses, semantic)
    
    
if __name__ == '__main__':
    from torchvision import models
    fe = models.resnet34(pretrained=True)
    pn = PoseNet(fe)
    MapNet(pn)
    MultiTask(pn)
    MultiTask3D(pn)
    print('No errors in constructors')