import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN

from .PieceWiseGammaEnhancedBN import SimpleGammaEnhancedBN,LearnableBoxCoxTransform,KLDivLossModule,BoxCoxTransform


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    def __init__(self, backbone=None, head=None, nf=16, gamma_range=[1.,4.],tmm="SimpleGammaEnhancedBN",norm_factor=1.):
        super().__init__()
        self.norm_factor = norm_factor
        # 2023-04-19 Modified by Huawei, definition of adaptive adjustment module

        if 'SimpleGammaEnhancedBN' in tmm.split(','):
            self.TMM = SimpleGammaEnhancedBN(gamma_range=gamma_range)
        else:
            raise Exception(F"unknown tmm: {tmm}")
        
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        self.backbone = backbone
        self.head = head

        self.cox_box = LearnableBoxCoxTransform()
        self.kl_loss = KLDivLossModule()

    def forward(self, x, targets=None):
        # 2023-04-19 Modified by Huawei, apply adaptive adjustment module on RAW inputs
        # self.vis_data(x)  # for debug
        bz,c,h,w = x.shape
        with torch.no_grad():
            if torch.max(x) > 1:
                x[x>1] = 114/255        # 修改padding
        x_tm = self.TMM(x)
        # modified
        x_bc = self.cox_box(x)
        kl_loss = self.kl_loss(x_bc,x_tm) * self.norm_factor

        #
        # x_tm = torch.clamp(x_tm, 0, 1) * 255.0
        x_tm = x_tm * 255.0
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x_tm)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {"total_loss": loss+kl_loss, "iou_loss": iou_loss, "l1_loss": l1_loss, "conf_loss": conf_loss, "cls_loss": cls_loss, "kl_loss":kl_loss,"num_fg": num_fg}
        else:
            outputs = self.head(fpn_outs)
        return outputs
    
class YOLOX_freeze_boxcox(YOLOX):
    def __init__(self, backbone=None, head=None, nf=16, gamma_range=[1, 4], tmm="SimpleGammaEnhancedBN", norm_factor=1, lam=0.35):
        super().__init__(backbone, head, nf, gamma_range, tmm, norm_factor)
        self.cox_box = BoxCoxTransform(lam=lam)