import math
import torch
import torch.nn as nn

from models.networks.srres_net import ResidualBlock, SrResNet
from models.networks.modules.pcda_module import DCNv2
from utils.net_utils import get_upsampling_func


class DCNAlignNet(nn.Module):
    def __init__(self,
                 num_feat=64,
                 deformable_groups=8,
                 num_extract_block=5,
                 activation=nn.ReLU):
        super(DCNAlignNet, self).__init__()
        # Feature extraction
        self.conv_first = nn.Conv2d(3, num_feat, 3, padding=1, bias=True)
        
        # Feature extraction residual layers
        self.residual_layer = nn.Sequential(*[ResidualBlock(num_feat, activation) 
                                              for _ in range(num_extract_block)])

        # Activation fn
        self.activation = activation(inplace=True)

        # Deformable convolutions for feature alignment
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

        # Init weights
        self.init_weights()

    def init_weights(self):
        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def forward(self, precomputed_features, x_center):
        """
        Align frames using precomputed features
        
        Args:
            precomputed_features (Tensor): (B, N, C, H, W) - precomputed features for all frames
            x_center (Tensor): (B, C, H, W) - center frame
        
        Returns:
            Tensor: Aligned frames
        """
        y = []
        batch_size, num, ch, w, h = precomputed_features.size()
        center = num // 2
        ref = precomputed_features[:, center, :, :, :].clone()
        
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = precomputed_features[:, i, :, :, :].contiguous()
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
    

class TDAN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_feat=64,
                 num_frames=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 res_frame_idx=None,
                 upsampling_fn='bicubic',
                 activation=nn.ReLU,
                 with_tsa=False):
        super(TDAN, self).__init__()
        # Alignment module
        self.align_net = DCNAlignNet(num_feat=num_feat,
                                     deformable_groups=deformable_groups,
                                     num_extract_block=num_extract_block,
                                     activation=activation)

        # Reference index
        if res_frame_idx is None:
            res_frame_idx = num_frames // 2

        # Reconstruction module
        upsample_fn = get_upsampling_func(mode=upsampling_fn)
        self.srnet = SrResNet(in_nc=in_channels*num_frames,
                              out_nc=out_channels,
                              nf=num_feat,
                              nb=num_reconstruct_block,
                              upsample_func=upsample_fn,
                              ref_idx=res_frame_idx,
                              with_tsa=with_tsa)
        self.reconstruction_channels = num_feat

        # Others
        self.num_frames = num_frames  # Size of the sliding window
        
        # Save the activation and conv_first from align_net for feature extraction
        self.activation = self.align_net.activation
        self.conv_first = self.align_net.conv_first
        self.residual_layer = self.align_net.residual_layer

    def forward(self, x):
        # Extract and align features
        out = self.align_net(x)

        # Fuse features and reconstruct
        out = self.srnet(out)

        return out

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
        padded_T = x_padded.size(1)
        
        # Extract features for all frames once - this avoids redundant computation
        # Flattened view for feature extraction
        x_flat = x_padded.view(-1, C, H, W)  # (B*padded_T, C, H, W)
        
        # Extract features
        features = self.activation(self.conv_first(x_flat))
        features = self.residual_layer(features)
        
        # Reshape back to sequence form
        features = features.view(B, padded_T, -1, H, W)  # (B, padded_T, feat_channels, H, W)
        
        outputs = []
        # Slide window across time dimension
        for i in range(T):
            window_start = i
            x_window = x_padded[:, window_start:window_start+self.num_frames, :, :, :]  # (B, N, C, H, W)
            features_window = features[:, window_start:window_start+self.num_frames, :, :, :]  # (B, N, feat_channels, H, W)
            
            # Use pre-computed features and center frame for alignment
            center_idx = self.num_frames // 2
            x_center = x_window[:, center_idx, :, :, :]
            
            # Perform alignment using cached features
            aligned_frames = self.align_net(features_window, x_center)

            # Reconstruct
            out_frame = self.srnet(aligned_frames.view(B, -1, H, W), x_center)
            outputs.append(out_frame)

        # (B, T_out, C, H_up, W_up)
        hr_data = torch.stack(outputs, dim=1)
        return {
            'hr_data': hr_data,  # n,t,c,hr_h,hr_w
        }
    
def infer_sequence(self, x, device=None):
    """
    Efficient inference on a full video sequence using feature caching
    
    Args:
        x (Tensor): (T, C, H, W) - low-resolution input video sequence
        device: Optional device to move tensors to
        
    Returns:
        Tensor: (T, C, H_out, W_out) - super-resolved video sequence
    """
    # Move to specified device if provided
    if device is not None:
        x = x.to(device)
    
    # Add batch dimension if not present
    if x.dim() == 4:  # (T, C, H, W)
        x = x.unsqueeze(0)  # (1, T, C, H, W)
    
    B, T, C, H, W = x.size()
    
    # Pad the sequence by reflection on both sides
    pad = self.num_frames // 2
    
    # Create proper reflection padding
    left_pad = x[:, 1:pad+1, ...].flip(1)  # Take first 'pad' frames after first frame and flip
    right_pad = x[:, -pad-1:-1, ...].flip(1)  # Take last 'pad' frames before last frame and flip
    
    # Concatenate along time dimension
    x_padded = torch.cat([left_pad, x, right_pad], dim=1)
    padded_T = x_padded.size(1)
    
    with torch.no_grad():  # No gradients needed for inference
        # Extract features for all frames once - this avoids redundant computation
        # Flattened view for feature extraction
        x_flat = x_padded.view(-1, C, H, W)  # (B*padded_T, C, H, W)
        
        # Extract features
        features = self.activation(self.conv_first(x_flat))
        features = self.residual_layer(features)
        
        # Reshape back to sequence form
        features = features.view(B, padded_T, -1, H, W)  # (B, padded_T, feat_channels, H, W)
        
        outputs = []
        # Slide window across time dimension
        for i in range(T):
            window_start = i
            x_window = x_padded[:, window_start:window_start+self.num_frames, :, :, :]  # (B, N, C, H, W)
            features_window = features[:, window_start:window_start+self.num_frames, :, :, :]  # (B, N, feat_channels, H, W)
            
            # Use pre-computed features and center frame for alignment
            center_idx = self.num_frames // 2
            x_center = x_window[:, center_idx, :, :, :]
            
            # Perform alignment using cached features
            aligned_frames = self.align_net(features_window, x_center)
            
            # Reconstruct
            out_frame = self.reconstruction(aligned_frames.view(B, -1, H, W))
            outputs.append(out_frame)

        # (B, T, C_out, H_out, W_out)
        hr_data = torch.stack(outputs, dim=1)
        
        # Remove batch dimension if input didn't have it
        if hr_data.size(0) == 1:
            hr_data = hr_data.squeeze(0)  # (T, C_out, H_out, W_out)
            
    return hr_data