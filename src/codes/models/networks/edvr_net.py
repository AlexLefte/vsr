import torch
import torch.nn as nn
import torch.nn.functional as F
from srres_net import ResidualBlock, SrResNet
from modules.pcda_module import PCDAlignment
from modules.tsa_module import TSAFusion
from utils.net_utils import get_upsampling_func

class EDVRNet(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.

    ``Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks``

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_feat=64,
                 num_frames=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 res_frame_idx=None,
                 hr_in=False,
                 with_tsa=True,
                 upsample_func='bicubic'):
        super(EDVRNet, self).__init__()
        if res_frame_idx is None:
            self.res_frame_idx = num_frames - 1  # Pick last frame as default
        else:
            self.res_frame_idx = res_frame_idx
        self.hr_in = hr_in
        self.with_tsa = with_tsa

        # Pyramid feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        self.feature_extraction = nn.Sequential(*[ResidualBlock(num_feat) for _ in range(num_extract_block)])
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Feaure Alignment
        self.pcd_align = PCDAlignment(num_feat=num_feat, 
                                      deformable_groups=deformable_groups)
        
        # Temporal-Spatial Alignment Fusion
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frames, res_frame_idx=self.res_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * num_feat, num_feat, 1, 1)

        # Reconstruction
        upsample_fn = get_upsampling_func(upsample_func)
        self.reconstruction = SrResNet(num_feat,
                                       out_channels,
                                       num_frames, 
                                       num_reconstruct_block, 
                                       upsample_fn)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Store frame count
        self.num_frames = num_frames

    def forward_sequence(self, x):
        """
        Args:
            x (Tensor): (B, T, C, H, W) - low-resolution input video sequence
        Returns:
            Tensor: (B, T_out, C, H_up, W_up) - super-resolved output sequence
        """
        B, T, C, H, W = x.size()
        N = self.num_frames
        assert T >= N, f"Input sequence length {T} must be >= window size {N}"

        outputs = []

        # Slide window across time dimension
        for i in range(T - N + 1):
            x_window = x[:, i:i+N, :, :, :]  # (B, N, C, H, W)
            out_frame = self.forward(x_window)  # super-resolved center frame
            outputs.append(out_frame)

        # (B, T_out, C, H_up, W_up)
        return torch.stack(outputs, dim=1)
    
    def forward(self, x):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        # Extract pyramidal features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.res_frame_idx, :, :, :].clone(), 
            feat_l2[:, self.res_frame_idx, :, :, :].clone(),
            feat_l3[:, self.res_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        # Fusion features with TSA if needed
        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        # Reconstruction with SRNet
        out = self.reconstruction(feat)
        return out