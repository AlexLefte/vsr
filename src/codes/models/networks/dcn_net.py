import math
import numpy as np
import torch
import torch.nn as nn

from models.networks.srnet import ResidualBlock, SRNet
from models.networks.modules.pcda_module import DCNv2
from utils.net_utils import get_upsampling_func


class DCNAlignNet(nn.Module):
    def __init__(self,
                 num_feat=64,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_deform_blocks=1,
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
        self.offset_proj = nn.ModuleList([
            nn.Conv2d(2 * num_feat, num_feat, 3, padding=1)
            for _ in range(num_deform_blocks)
        ])
        self.deformable_blocks = nn.ModuleList([
            DCNv2(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, deformable_groups=deformable_groups)
            for _ in range(num_deform_blocks)
        ])
        
        # Store the number of deformable blocks
        self.num_deform_blocks = num_deform_blocks

        # Reconstruction layer
        self.recon_lr = nn.Conv2d(num_feat, 3, 3, padding=1, bias=True)

        # Init weights
        self.init_weights()

    def init_weights(self):
        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def forward(self, features, x_center):
        """
        Align frames using precomputed features
        
        Args:
            features (Tensor): (B, N, C, H, W) - precomputed features for all frames
            x_center (Tensor): (B, C, H, W) - center frame
        
        Returns:
            Tensor: Aligned frames
        """
        B, N, C, H, W = features.size()
        center = N // 2
        ref = features[:, center, :, :, :]
        aligned_list = []

        for i in range(N):
            if i == center:
                aligned_list.append(x_center.unsqueeze(1))  # passthrough the original LR center frame
            else:
                # Perform alignment
                supp = features[:, i, :, :, :]
                for idx in range(self.num_deform_blocks):
                    concat_feat = torch.cat([ref, supp], dim=1)
                    offset_feat = self.offset_proj[idx](concat_feat)
                    supp = self.deformable_blocks[idx](supp, offset_feat)

                # Reconstruct
                recon = self.recon_lr(supp).unsqueeze(1)
                aligned_list.append(recon)

        return torch.cat(aligned_list, dim=1)  # (B, N, C, H, W)
    

class DcnVSR(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_feat=64,
                 num_frames=3,
                 deformable_groups=8,
                 num_extract_block=3,
                 num_reconstruct_block=10,
                 num_deform_blocks=1,
                 res_frame_idx=None,
                 upsampling_fn='bicubic',
                 activation=nn.ReLU,
                 with_tsa=False,
                 shallow_feat_res=False):
        super(DcnVSR, self).__init__()
        # Alignment module
        self.align_net = DCNAlignNet(num_feat=num_feat,
                                     deformable_groups=deformable_groups,
                                     num_extract_block=num_extract_block,
                                     num_deform_blocks=num_deform_blocks,
                                     activation=activation)

        # Reference index
        if res_frame_idx is None:
            res_frame_idx = num_frames // 2

        # Reconstruction module
        upsample_fn = get_upsampling_func(mode=upsampling_fn)
        self.srnet = SRNet(in_nc=in_channels*num_frames,
                        out_nc=out_channels,
                        nf=num_feat,
                        nb=num_reconstruct_block,
                        upsample_func=upsample_fn,
                        ref_idx=res_frame_idx,
                        with_tsa=with_tsa,
                        shallow_feat_res=shallow_feat_res)
        self.reconstruction_channels = num_feat

        # Others
        self.num_frames = num_frames  # Size of the sliding window
        
        # Save the activation and conv_first from align_net for feature extraction
        self.activation = self.align_net.activation
        self.conv_first = self.align_net.conv_first
        self.residual_layer = self.align_net.residual_layer
        self.cached_feats = None

    def forward(self, x):
        B, N, C, H, W = x.size()
        center_idx = N // 2

        if hasattr(self, 'cached_feats') and self.cached_feats is not None:
            # Extrage caracteristici doar pentru ultimul frame (cel nou venit)
            new_frame = x[:, -1, :, :, :]  # (B, C, H, W)
            new_feat = self.activation(self.conv_first(new_frame))
            new_feat = self.residual_layer(new_feat)
            new_feat = new_feat.unsqueeze(1)  # (B, 1, feat_channels, H, W)

            # Concatenează feat-urile cache-uite cu cel nou
            feats = torch.cat([self.cached_feats, new_feat], dim=1)  # (B, N, feat_channels, H, W)
        else:
            # Pentru prima fereastră, calculează feat-uri pentru toate cadrele
            x_flat = x.view(-1, C, H, W)  # (B*N, C, H, W)
            feats = self.activation(self.conv_first(x_flat))
            feats = self.residual_layer(feats)
            feats = feats.view(B, N, -1, H, W)  # (B, N, feat_channels, H, W)

        # Actualizează cache-ul cu ultimele N-1 feat-uri (detach pentru a nu propaga gradientul înapoi)
        self.cached_feats = feats[:, 1:, :, :, :].detach()

        # Frame-ul central LR din fereastră
        x_center = x[:, center_idx, :, :, :]

        # Aliniază folosind feat-urile și frame-ul central
        aligned = self.align_net(feats, x_center)  # (B, N, C, H, W)

        # Super-rezoluează frame-ul central folosind cadrele aliniate
        out = self.srnet(aligned.view(B, -1, H, W), x_center)

        return out

    def forward_sequence(self, x):
        """
        Args:
            x (Tensor): (B, T, C, H, W) - low-resolution input video sequence
        """
        B, T, C, H, W = x.size()
    
        # Pad the sequence by reflection on both sides
        center_idx = self.num_frames // 2
        
        # Create proper reflection padding
        left_pad = x[:, 1:center_idx+1, ...].flip(1)  # Take first 'pad' frames after first frame and flip
        right_pad = x[:, -center_idx-1:-1, ...].flip(1)  # Take last 'pad' frames before last frame and flip
        
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
        # print(f'Extracted features shape: {features.shape}')

        outputs = []
        lrs_aligned = []
        # Slide window across time dimension
        for i in range(T):
            window_start = i
            x_window = x_padded[:, window_start:window_start+self.num_frames, :, :, :]  # (B, N, C, H, W)
            features_window = features[:, window_start:window_start+self.num_frames, :, :, :]  # (B, N, feat_channels, H, W)
            
            # Use pre-computed features and center frame for alignment
            x_center = x_window[:, center_idx, :, :, :]
            
            # Perform alignment using cached features
            lr_aligned = self.align_net(features_window, x_center)  # B Num_Frames C H W
            # print(f"Aligned lr shape: {lr_aligned.shape}")

            # Reconstruct
            # out_frame = self.srnet(lr_aligned.view(B, -1, H, W), x_center)  # DELETE: edited here
            out_frame = self.srnet(lr_aligned, x_center)
            outputs.append(out_frame)

            # Append aligned frames (without the central frame)
            lr_aligned_left = lr_aligned[:, :center_idx]
            lr_aligned_right = lr_aligned[:, center_idx + 1:]
            lrs_aligned.append(torch.cat([lr_aligned_left, lr_aligned_right], dim=1))

        # (B, T_out, C, H_up, W_up)
        hr_data = torch.stack(outputs, dim=1)
        lrs_aligned = torch.stack(lrs_aligned, dim=1)  # B x T x Win_Size x C x H x W
        return {
            'hr_data': hr_data,  # n,t,c,hr_h,hr_w
            'lr_aligned_feat': lrs_aligned  # B x T x Win_Size - 1 x C x H x W
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
        x_padded = torch.cat([left_pad, x, right_pad], dim=1).to(device)
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
                x_center = x_window[:, center_idx, :, :, :].to(device)
                
                # Perform alignment using cached features
                aligned_frames = self.align_net(features_window, x_center)
                
                # Reconstruct
                out_frame = self.srnet(aligned_frames.view(B, -1, H, W), x_center)
                outputs.append(out_frame.detach().squeeze(0).cpu().numpy())

            # (B, T, C_out, H_out, W_out)
            hr_data = np.stack(outputs)  
        
        # Cleanup
        # del x, x_padded, x_flat, features, x_window, features_window, x_center, aligned_frames, out_frame
        return hr_data.transpose(0, 2, 3, 1)
    
    def generate_dummy_input(self, lr_size):
        # Input video dummy: (B=1, T=3 frames)
        c, lr_h, lr_w = lr_size
        dummy_input = torch.randn(3, c, lr_w, lr_h, dtype=torch.float32)
        dummy_input = dummy_input.unsqueeze(dim=0)

        return {
            'x': dummy_input
        }