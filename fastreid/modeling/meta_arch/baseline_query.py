# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

@META_ARCH_REGISTRY.register()
class Baseline_query(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            _heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads
        self._heads=_heads
        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        cfg2=cfg.clone()
        if cfg2.is_frozen(): cfg2.defrost()
        cfg2.MODEL.BACKBONE.FEAT_DIM=768*2
        _heads=build_heads(cfg2)
        return {
            'backbone': backbone,
            'heads': heads,
            '_heads': _heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        if 'camids' in batched_inputs.keys():
            camids = batched_inputs['camids']
        else:
            camids = 0
        backbone_list= self.backbone(images,self.training)
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(backbone_list['cls_tokens'], targets)
            outputs_t=self.heads(backbone_list['text'],targets)
            outputs2=self._heads(backbone_list['query_tokens'], targets)

            out={}
            out['outputs']=outputs
            out['outputs2']=outputs2
            out['outputs_t'] = outputs_t
            out['oreg_loss']=backbone_list['oreg_loss']

            losses = self.losses(out, targets)
            return losses
        else:
            outputs = self.heads(backbone_list['cls_tokens'])
            outputs2=self._heads(backbone_list['query_tokens'])
            from torch.nn import functional as F
            return torch.cat((outputs, outputs2), dim=-1)


    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, out,gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = out['outputs']['pred_class_logits'].detach()
        cls_outputs       = out['outputs']['cls_outputs']
        pred_features     = out['outputs']['features']

        cls_outputs2       = out['outputs2']['cls_outputs']
        pred_features2     = out['outputs2']['features']

        cls_outputst       = out['outputs_t']['cls_outputs']
        pred_featurest     = out['outputs_t']['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']
        loss_dict['loss_oreg']=out['oreg_loss']*0.01
        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            loss_dict['loss_cls2'] = cross_entropy_loss(
                cls_outputs2,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')
            loss_dict['loss_clst'] = cross_entropy_loss(
                cls_outputst,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')*0.1
        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

            loss_dict['loss_triplet2'] = triplet_loss(
                pred_features2,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')
            loss_dict['loss_triplett'] = triplet_loss(
                pred_featurest,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')*0.1

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict