import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_nets import BaseSequenceDiscriminator
from utils.net_utils import backward_warp
from torch.nn.utils import spectral_norm


class DiscriminatorBlocks(nn.Module):
    def __init__(self, spectral_norm=False):
        super(DiscriminatorBlocks, self).__init__()

        print(f"Using spectral norm: {spectral_norm}.")
        self.spectral_norm = spectral_norm
        self.block1 = self.make_conv_block(64, 64)  # /2
        self.block2 = self.make_conv_block(64, 64)  # /4
        self.block3 = self.make_conv_block(64, 128)  # /8
        self.block4 = self.make_conv_block(128, 256)  # /16

    def make_conv_block(self, in_ch, out_ch):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2,
                         padding=1, bias=False)
        if self.spectral_norm:
            conv = spectral_norm(conv)
        layers = [conv, nn.LeakyReLU(0.2, inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        feature_list = [out1, out2, out3, out4]
        return out4, feature_list


class SpatioTemporalDiscriminator(BaseSequenceDiscriminator):
    """ Spatio-Temporal discriminator in proposed in TecoGAN
    """

    def __init__(self, in_nc=3, spatial_size=128, tempo_range=3, scale=4, spectral_norm=False):
        super(SpatioTemporalDiscriminator, self).__init__()

        # basic settings
        mult = 3  # (conditional triplet, input triplet, warped triplet)
        self.spatial_size = spatial_size
        self.tempo_range = tempo_range
        assert self.tempo_range == 3, 'currently only support 3 as tempo_range'
        self.scale = scale

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc*tempo_range*mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        # discriminator block
        self.discriminator_block = DiscriminatorBlocks(spectral_norm=spectral_norm)  # downsample 16x

        # classifier
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out, feature_list = self.discriminator_block(out)

        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out, feature_list

    def forward_sequence(self, data, args_dict):
        """
            :param data: should be either hr_data or gt_data
            :param args_dict: a dict including data/config needed here
        """

        # ------------ setup params ------------ #
        net_G = args_dict['net_G']
        lr_data = args_dict['lr_data']
        bi_data = args_dict['bi_data']
        hr_flow = args_dict['hr_flow']

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, hr_h, hr_w = data.size()

        s_size = self.spatial_size
        t = t // 3 * 3  # discard other frames
        n_clip = n * t // 3  # total number of 3-frame clips in all batches

        c_size = int(s_size * args_dict['crop_border_ratio'])
        n_pad = (s_size - c_size) // 2


        # ------------ compute forward & backward flow ------------ #
        if 'hr_flow_merge' not in args_dict:
            if args_dict['use_pp_crit']:
                hr_flow_bw = hr_flow[:, 0:t:3, ...]  # e.g., frame1 -> frame0
                hr_flow_idle = torch.zeros_like(hr_flow_bw)
                hr_flow_fw = hr_flow.flip(1)[:, 1:t:3, ...]
            else:
                lr_curr = lr_data[:, 1:t:3, ...]
                lr_curr = lr_curr.reshape(n_clip, c, lr_h, lr_w)

                lr_next = lr_data[:, 2:t:3, ...]
                lr_next = lr_next.reshape(n_clip, c, lr_h, lr_w)

                # compute forward flow
                lr_flow_fw = net_G.fnet(lr_curr, lr_next)
                hr_flow_fw = self.scale * net_G.upsample_func(lr_flow_fw)

                hr_flow_bw = hr_flow[:, 0:t:3, ...]  # e.g., frame1 -> frame0
                hr_flow_idle = torch.zeros_like(hr_flow_bw)  # frame1 -> frame1
                hr_flow_fw = hr_flow_fw.view(n, t // 3, 2, hr_h, hr_w)  # frame1 -> frame2

            # merge bw/idle/fw flows
            hr_flow_merge = torch.stack(
                [hr_flow_bw, hr_flow_idle, hr_flow_fw], dim=2)  # n,t//3,3,2,h,w

            # reshape and stop gradient propagation
            hr_flow_merge = hr_flow_merge.view(n_clip * 3, 2, hr_h, hr_w).detach()

        else:
            # reused data to reduce computations
            hr_flow_merge = args_dict['hr_flow_merge']


        # ------------ build up inputs for D (3 parts) ------------ #
        # part 1: bicubic upsampled data (conditional inputs)
        cond_data = bi_data[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        # note: permutation is not necessarily needed here, it's just to keep
        #       the same impl. as TecoGAN-Tensorflow (i.e., rrrgggbbb)
        cond_data = cond_data.permute(0, 2, 1, 3, 4)
        cond_data = cond_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # part 2: original data
        orig_data = data[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        orig_data = orig_data.permute(0, 2, 1, 3, 4)
        orig_data = orig_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # part 3: warped data
        warp_data = backward_warp(
            data[:, :t, ...].reshape(n * t, c, hr_h, hr_w), hr_flow_merge)
        warp_data = warp_data.view(n_clip, 3, c, hr_h, hr_w)
        warp_data = warp_data.permute(0, 2, 1, 3, 4)
        warp_data = warp_data.reshape(n_clip, c * 3, hr_h, hr_w)
        # remove border to increase training stability as proposed in TecoGAN
        warp_data = F.pad(
            warp_data[..., n_pad: n_pad + c_size, n_pad: n_pad + c_size],
            (n_pad,) * 4, mode='constant')

        # combine 3 parts together
        input_data = torch.cat([orig_data, warp_data, cond_data], dim=1)


        # ------------ classify ------------ #
        pred = self.forward(input_data)  # out, feature_list


        # construct output dict (return other data beside pred)
        ret_dict = {
            'hr_flow_merge': hr_flow_merge,
            'input_data': input_data
        }

        return pred, ret_dict


class SpatialDiscriminator(BaseSequenceDiscriminator):
    """ Spatial discriminator
    """

    def __init__(self, in_nc=3, spatial_size=128, use_cond=False):
        super(SpatialDiscriminator, self).__init__()

        # basic settings
        self.use_cond = use_cond  # whether to use conditional input
        mult = 2 if self.use_cond else 1
        tempo_range = 1

        # input conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc*tempo_range*mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        # discriminator block
        self.discriminator_block = DiscriminatorBlocks()  # /16

        # classifier
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out, feature_list = self.discriminator_block(out)

        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out, feature_list

    def forward_sequence(self, data, args_dict):
        # ------------ setup params ------------ #
        n, t, c, hr_h, hr_w = data.size()
        data = data.view(n * t, c, hr_h, hr_w)


        # ------------ build up inputs for D ------------ #
        if self.use_cond:
            bi_data = args_dict['bi_data'].view(n * t, c, hr_h, hr_w)
            input_data = torch.cat([bi_data, data], dim=1)
        else:
            input_data = data


        # ------------ classify ------------ #
        pred = self.forward(input_data)


        # construct output dict (nothing to return)
        ret_dict = {}

        return pred, ret_dict