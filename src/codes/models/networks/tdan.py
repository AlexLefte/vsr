import math
import torch
import torch.nn as nn

from models.networks.srres_net import ResidualBlock, SrResNet
from models.networks.modules.pcda_module import DCNv2
from time import time
from utils.net_utils import get_upsampling_func


class DCNAlignNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_feat=64,
                 num_frames=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 res_frame_idx=None,
                 upsample_func='bicubic'):
        super(DCNAlignNet, self).__init__()

        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(ResidualBlock, num_extract_block)
        self.relu = nn.ReLU(inplace=True)

        # Deformable convolutions
        self.cr = nn.Conv2d(2 * num_feat, num_feat, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(num_feat, 18 * deformable_groups, 3, padding=1, bias=True)
        self.dconv_1 = DCNv2(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.off2d_2 = nn.Conv2d(num_feat, 18 * deformable_groups, 3, padding=1, bias=True)
        self.deconv_2 = DCNv2(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.off2d_3 = nn.Conv2d(num_feat, 18 * deformable_groups, 3, padding=1, bias=True)
        self.deconv_3 = DCNv2(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.off2d = nn.Conv2d(num_feat, 18 * deformable_groups, 3, padding=1, bias=True)
        self.dconv = DCNv2(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.recon_lr = nn.Conv2d(num_feat, 3, 3, padding=1, bias=True)

        # Reconstruction module
        upsample_fn = get_upsampling_func(mode=upsample_func)
        self.reconstruction = SrResNet(in_channels=n,
                                       out_nc=out_channels,
                                       nf=num_feat,
                                       nb=num_reconstruct_block,
                                       upsample_func=upsample_fn,
                                       ref_idx=res_frame_idx)
        self.reconstruction_channels = num_feat

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, i, :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)

            # feature trans
            offset1 = self.off2d_1(fea)
            fea = self.dconv_1(fea, offset1)
            offset2 = self.off2d_2(fea)
            fea = self.deconv_2(fea, offset2)
            offset3 = self.off2d_3(fea)
            fea = self.deconv_3(supp, offset3)
            offset4 = self.off2d(fea)
            aligned_fea = self.dconv(fea, offset4)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def forward_sequence(self, x):
        """
        Args:
            x (Tensor): (B, T, C, H, W) - low-resolution input video sequence
        """
        B, T, C, H, W = x.size()
    
        # Pad the sequence by reflection on both sides
        pad = self.num_frames // 2
        
        # Create proper reflection padding
        left_pad = x[:, 1:pad+1, ...].flip(1)  # Take first 'pad' frames after first frame and flip
        right_pad = x[:, -pad-1:-1, ...].flip(1)  # Take last 'pad' frames before last frame and flip
        
        # Concatenate along time dimension
        x_padded = torch.cat([left_pad, x, right_pad], dim=1)
        
        # Ensure we have enough frames for the sliding window
        padded_T = x_padded.size(1)
        N = self.num_frames
        assert padded_T >= N, f"Padded sequence length {padded_T} must be >= window size {N}"

        outputs = []
        # Slide window across time dimension
        for i in range(T):
            window_start = i
            x_window = x_padded[:, window_start:window_start+N, :, :, :]  # (B, N, C, H, W)
            out_frame = self.forward(x_window)  # super-resolved center frame
            outputs.append(out_frame)

        # (B, T_out, C, H_up, W_up)
        hr_data = torch.stack(outputs, dim=1)
        return {
            'hr_data': hr_data,  # n,t,c,hr_h,hr_w
        }

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()  # 5 video frames

        # center frame interpolation
        center = num // 2

        # extract features
        y = x.view(-1, ch, w, h)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)

        # align supporting frames
        lrs = self.align(out, x_center)  # motion alignments
        y = lrs.view(batch_size, -1, w, h)
        
        # reconstruction
        out = self.reconstruction(y)
        return out
    