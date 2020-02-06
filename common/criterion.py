from __future__ import division
from torch import nn
import torch
from common import pose_utils
"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

"""
This module implements the various loss functions (a.k.a. criterions) used
in the paper
"""


class QuaternionLoss(nn.Module):
    """
    Implements distance between quaternions as mentioned in
    D. Huynh. Metrics for 3D rotations: Comparison and analysis
    """

    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, q1, q2):
        """
        :param q1: N x 4
        :param q2: N x 4
        :return:
        """
        loss = 1 - torch.pow(pose_utils.vdot(q1, q2), 2)
        loss = torch.mean(loss)
        return loss


class PoseNetCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
                 saq=0.0, learn_beta=False):
        """
        :param t_loss_fn: loss function to be used for translation
        :param q_loss_fn: loss function to be used for rotation
        :param sax: absolute translation loss weight
        :param saq: absolute rotation loss weight
        :param learn_beta: learn sax and saq?
        """
        super(PoseNetCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.tensor(sax, dtype=torch.float), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.tensor(saq, dtype=torch.float), requires_grad=learn_beta)

    def forward(self, pred, targ):
        """
        :param pred: N x 7
        :param targ: N x 7
        :return:
        """
        loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + \
            self.sax +\
            torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) +\
            self.saq
        return loss


class SemanticCriterion(nn.Module):
    def __init__(self, sem_loss=nn.CrossEntropyLoss()):
        """
        :param sem_loss: loss function to be used for semantics
        """
        super(SemanticCriterion, self).__init__()
        self.semantic_loss = sem_loss
    def forward(self, pred, targ):
        """
        :param pred: N x 7
        :param targ: N x 7
        :return: 
        """
        #s = pred.size()
        
        s = targ.size()
        targ = targ.view(-1,*s[2:])
        loss =  self.semantic_loss(pred, targ)
        return loss


class MapNetCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), dual_target=False,
                 sax=0.0, saq=0.0, srx=0.0, srq=0.0, sas=0.0,
                 learn_beta=False, learn_gamma=False, learn_sigma=False,
                 sem_loss=nn.CrossEntropyLoss()):
        """
        Implements L_D from eq. 2 in the paper
        :param t_loss_fn: loss function to be used for translation
        :param q_loss_fn: loss function to be used for rotation
        :param sem_loss: loss function to be used for semantics
        :param sax: absolute translation loss weight
        :param saq: absolute rotation loss weight
        :param srx: relative translation loss weight
        :param srq: relative rotation loss weight
        :param sas: absolute sematic loss weight
        :param learn_beta: learn sax and saq?
        :param learn_gamma: learn srx and srq?
        :param learn_sigma: learn sas?
        """
        super(MapNetCriterion, self).__init__()
        #print("Dual target in MapNetCriterion: %r"%self.dual_target)
        self.dual_target = dual_target
        self.sem_loss = sem_loss
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.tensor(sax, dtype=torch.float), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.tensor(saq, dtype=torch.float), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.tensor(srx, dtype=torch.float), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.tensor(srq, dtype=torch.float), requires_grad=learn_gamma)
        if self.dual_target:
            self.sas = nn.Parameter(torch.tensor(sas, dtype=torch.float), requires_grad=learn_sigma)

    def forward(self, pred, targ):
        """
        :param pred: N x T x 6
        :param targ: N x T x 6
        :return:
        """

        if self.dual_target:
            pred, sem_pred = pred[0], pred[1]
            targ, sem_targ = targ[0], targ[1]
        if len(pred.size()) < 3:
            pred = pred.unsqueeze(0)
        s = pred.size()
        
        # absolute pose loss
        # get the VOs
        pred_vos = pose_utils.calc_vos_simple(pred)
        targ_vos = pose_utils.calc_vos_simple(targ)
    
        t_loss = self.t_loss_fn(pred.view(-1, *s[2:])[:, :3],
                                targ.view(-1, *s[2:])[:, :3])
        
        q_loss = self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:],
                                targ.view(-1, *s[2:])[:, 3:])
        
        
        vo_t_loss = self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3],
                                   targ_vos.view(-1, *s[2:])[:, :3]) if not ((type(pred_vos) == list and len(pred_vos) == 0) or (type(targ_vos) == list and len(targ_vos) == 0)) else 0.0
        
        vo_q_loss = self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:],
                                   targ_vos.view(-1, *s[2:])[:, 3:]) if not ((type(pred_vos) == list and len(pred_vos) == 0) or (type(targ_vos) == list and len(targ_vos) == 0)) else 0.0
        
        
        abs_loss =(
            torch.exp(-self.sax) * t_loss + 
            self.sax + 
            torch.exp(-self.saq) * q_loss +
            self.saq
        )
        
        vo_loss = (
            torch.exp(-self.srx) * vo_t_loss + 
            self.srx + 
            torch.exp(-self.srq) * vo_q_loss + 
            self.srq
        )
        
        loss_pose = (abs_loss + vo_loss)
        loss_list = [t_loss, q_loss]
        
        if self.dual_target:
            s = sem_targ.size()
            sem_targ = sem_targ.view(-1,*s[2:])
            s_loss = self.sem_loss(sem_pred, sem_targ)
            
            loss_semantic = torch.exp(-self.sas) * s_loss + self.sas
            loss_list.append(s_loss)
        else:
            loss_semantic = 0
        
        loss = loss_pose + loss_semantic
        return loss, loss_list  


class UncertainyCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), dual_target=False,
                 sax=0.0, saq=0.0, srx=0.0, srq=0.0, sas=0.0,
                 learn_beta=False, learn_gamma=False, learn_sigma=False, learn_log=True,
                 sem_loss=nn.CrossEntropyLoss(), **kwargs):
        """
        Implements L_D from eq. 2 in the paper
        :param t_loss_fn: loss function to be used for translation
        :param q_loss_fn: loss function to be used for rotation
        :param sem_loss: loss function to be used for semantics
        :param sax: absolute translation loss weight
        :param saq: absolute rotation loss weight
        :param srx: relative translation loss weight
        :param srq: relative rotation loss weight
        :param sas: absolute sematic loss weight
        :param learn_beta: learn sax and saq?
        :param learn_gamma: learn srx and srq?
        :param learn_sigma: learn sas?
        """
        super(UncertainyCriterion, self).__init__()
        #print("Dual target in MapNetCriterion: %r"%self.dual_target)
        self.dual_target = dual_target
        self.learn_log = learn_log
        self.sem_loss = sem_loss
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.tensor(sax, dtype=torch.float), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.tensor(saq, dtype=torch.float), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.tensor(srx, dtype=torch.float), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.tensor(srq, dtype=torch.float), requires_grad=learn_gamma)
        if self.dual_target:
            self.sas = nn.Parameter(torch.tensor(sas, dtype=torch.float), requires_grad=learn_sigma)

    def forward(self, pred, targ):
        """
        :param pred: N x T x 6
        :param targ: N x T x 6
        :return:
        """

        
        if self.dual_target:
            pred, sem_pred = pred[0], pred[1]
            targ, sem_targ = targ[0], targ[1]
        s = pred.size()
        
        # absolute pose loss
        # get the VOs
        pred_vos = pose_utils.calc_vos_simple(pred)
        targ_vos = pose_utils.calc_vos_simple(targ)
        
        t_loss = self.t_loss_fn(pred.view(-1, *s[2:])[:, :3],
                                targ.view(-1, *s[2:])[:, :3])
        
        q_loss = self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:],
                                targ.view(-1, *s[2:])[:, 3:])
        
        vo_t_loss = self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3],
                                   targ_vos.view(-1, *s[2:])[:, :3])
        
        vo_q_loss = self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:],
                                   targ_vos.view(-1, *s[2:])[:, 3:])
        
        if self.learn_log:
            loss_pose = (
                t_loss / (2 * torch.exp(2 * self.sax)) +
                q_loss / (2 * torch.exp(2 * self.saq)) +
                vo_t_loss / (2 * torch.exp(2 * self.srx)) + 
                vo_q_loss / (2 * torch.exp(2 * self.srq)) +
                self.sax + self.saq + self.srx + self.srq 
            )
        else:
            loss_pose = (
                t_loss / (2 * self.sax * self.sax) +
                q_loss / (2 * self.saq * self.saq) +
                vo_t_loss / (2 * self.srx * self.srx) + 
                vo_q_loss / (2 * self.srq * self.srq) +
                torch.log(self.sax * self.saq * self.srx * self.srq)
            )
        loss_list = [t_loss, q_loss]
        
        if self.dual_target:
            s = sem_targ.size()
            sem_targ = sem_targ.view(-1,*s[2:])
            s_loss = self.sem_loss(sem_pred, sem_targ)
            if self.learn_log:
                loss_semantic = (s_loss / torch.exp(2 * self.sas) + 
                             self.sas)
            else:
                loss_semantic = (s_loss / (self.sas * self.sas) + 
                             torch.log(self.sas))
            loss_list.append(s_loss)
        else:
            loss_semantic = 0
        
        
        loss = loss_pose + loss_semantic
        
        return loss, loss_list

class MapNetOnlineCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
                 saq=0.0, srx=0, srq=0.0, learn_beta=False, learn_gamma=False,
                 gps_mode=False):
        """
        Implements L_D + L_T from eq. 4 in the paper
        :param t_loss_fn: loss function to be used for translation
        :param q_loss_fn: loss function to be used for rotation
        :param sax: absolute translation loss weight
        :param saq: absolute rotation loss weight
        :param srx: relative translation loss weight
        :param srq: relative rotation loss weight
        :param learn_beta: learn sax and saq?
        :param learn_gamma: learn srx and srq?
        :param gps_mode: If True, uses simple VO and only calculates VO error in
        position
        """
        super(MapNetOnlineCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.tensor(sax, dtype=torch.float), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.tensor(saq, dtype=torch.float), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.tensor(srx, dtype=torch.float), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.tensor(srq, dtype=torch.float), requires_grad=learn_gamma)
        self.gps_mode = gps_mode

    def forward(self, pred, targ):
        """
        targ contains N groups of pose targets, making the mini-batch.
        In each group, the first T poses are absolute poses, used for L_D while
        the next T-1 are relative poses, used for L_T
        All the 2T predictions in pred are absolute pose predictions from MapNet,
        but the last T predictions are converted to T-1 relative predictions using
        pose_utils.calc_vos()
        :param pred: N x 2T x 7
        :param targ: N x 2T-1 x 7
        :return:
        """
        s = pred.size()
        T = s[1] // 2
        pred_abs = pred[:, :T, :].contiguous()
        # these contain abs pose predictions for now
        pred_vos = pred[:, T:, :].contiguous()
        targ_abs = targ[:, :T, :].contiguous()
        # contain absolute translations if gps_mode
        targ_vos = targ[:, T:, :].contiguous()

        # absolute pose loss
        pred_abs = pred_abs.view(-1, *s[2:])
        targ_abs = targ_abs.view(-1, *s[2:])
        abs_loss =\
            torch.exp(-self.sax) * self.t_loss_fn(pred_abs[:, :3], targ_abs[:, :3]) + \
            self.sax + \
            torch.exp(-self.saq) * self.q_loss_fn(pred_abs[:, 3:], targ_abs[:, 3:]) + \
            self.saq

        # get the VOs
        if not self.gps_mode:
            pred_vos = pose_utils.calc_vos(pred_vos)

        # VO loss
        s = pred_vos.size()
        pred_vos = pred_vos.view(-1, *s[2:])
        targ_vos = targ_vos.view(-1, *s[2:])
        idx = 2 if self.gps_mode else 3
        vo_loss = \
            torch.exp(-self.srx) * self.t_loss_fn(pred_vos[:, :idx], targ_vos[:, :idx]) + \
            self.srx
        if not self.gps_mode:
            vo_loss += \
                torch.exp(-self.srq) * self.q_loss_fn(pred_vos[:, 3:], targ_vos[:, 3:]) + \
                self.srq

        # total loss
        loss = abs_loss + vo_loss
        return loss
