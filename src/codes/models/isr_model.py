import numpy as np
import torch
import torch.optim as optim
from collections import OrderedDict
from .base_model import BaseModel
from .networks import define_generator
from .networks.vgg_nets import VGGFeatureExtractor
from .optim import define_criterion, define_lr_schedule
from utils import net_utils, data_utils


class ISRModel(BaseModel):
    """ A simplified model wrapper for image super-resolution """

    def __init__(self, opt):
        super(ISRModel, self).__init__(opt)

        if self.verbose:
            self.logger.info('{} ISR Model Info {}'.format('=' * 20, '=' * 20))
            self.logger.info('Model: {}'.format(opt['model']['name']))

        self.set_network()

        if self.is_train:
            self.config_training()

    def set_network(self):
        self.net_G = define_generator(self.opt).to(self.device)

        if self.verbose:
            self.logger.info('Generator: {}\n'.format(
                self.opt['model']['generator']['name']) + str(self.net_G))

        load_path = self.opt['model']['generator'].get('load_path')
        if load_path:
            self.load_network(self.net_G, load_path)
            if self.verbose:
                self.logger.info('Loaded generator from: {}'.format(load_path))

    def config_training(self):
        self.set_criterion()

        self.optim_G = optim.Adam(
            self.net_G.parameters(),
            lr=self.opt['train']['generator']['lr'],
            weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
            betas=(
                self.opt['train']['generator'].get('beta1', 0.9),
                self.opt['train']['generator'].get('beta2', 0.999))
        )

        self.sched_G = define_lr_schedule(
            self.opt['train']['generator'].get('lr_schedule'), self.optim_G)

    def set_criterion(self):
        self.pix_crit = define_criterion(self.opt['train'].get('pixel_crit'))

        self.feat_crit = define_criterion(self.opt['train'].get('feature_crit'))
        if self.feat_crit is not None:
            feature_layers = self.opt['train']['feature_crit'].get(
                'feature_layers', ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'])
            self.feature_weights = self.opt['train']['feature_crit'].get(
                'feature_weights', [0.1, 0.1, 1, 1, 1])
            self.net_F = VGGFeatureExtractor(feature_layers).to(self.device)

    def train(self, data):
        """ Mini-batch training with single-frame image SR """
        lr_data, gt_data = data['lr'].squeeze(1), \
                            data['gt'].squeeze(1)

        self.net_G.train()
        self.optim_G.zero_grad()

        sr_data = self.net_G(lr_data)
        loss_G = 0
        self.log_dict = OrderedDict()

        # pixel loss
        pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
        loss_pix_G = pix_w * self.pix_crit(sr_data, gt_data)
        loss_G += loss_pix_G
        self.log_dict['l_pix_G'] = loss_pix_G.item()

        # feature (perceptual) loss
        if self.feat_crit is not None:
            hr_feat_lst = self.net_F(sr_data)
            gt_feat_lst = self.net_F(gt_data)
            loss_feat_G = 0
            for i, (hr_feat, gt_feat) in enumerate(zip(hr_feat_lst, gt_feat_lst)):
                loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach()) * self.feature_weights[i]

            feat_w = self.opt['train']['feature_crit'].get('weight', 1)
            loss_feat_G *= feat_w
            loss_G += loss_feat_G
            self.log_dict['l_feat_G'] = loss_feat_G.item()

        loss_G.backward()
        self.optim_G.step()

    def infer(self, lr_images):
        """ Inference for a single low-resolution image (3D tensor HWC or CHW) """
        self.net_G.eval()

        lr_images = data_utils.canonicalize(lr_images)  # (C,H,W) float tensor
        lr_images = lr_images.to(self.device)  # Add batch dim

        with torch.no_grad():
            sr_images = self.net_G(lr_images).cpu().numpy()

        self.net_G.train()

        return sr_images.transpose(0, 2, 3, 1)

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)