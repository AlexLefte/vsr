import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from models.networks.srres_net import ResidualBlock, SrResNet
from models.networks.modules.pcda_module import PCDAlignment
from models.networks.modules.tsa_module import TSAFusion
from time import time
from utils.net_utils import get_upsampling_func


# class SRNet(nn.Module):
#     """ Reconstruction & Upsampling network
#     """

#     def __init__(self, out_nc=3, nf=64, nb=16, upsample_func=None,
#                  scale=4):
#         super(SRNet, self).__init__()

#         # residual blocks
#         self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

#         # upsampling
#         self.conv_up = nn.Sequential(
#             nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
#             nn.ReLU(inplace=True))

#         self.conv_up_cheap = nn.Sequential(
#             nn.PixelShuffle(4),
#             nn.ReLU(inplace=True))

#         # output conv.
#         self.conv_out = nn.Conv2d(4, out_nc, 3, 1, 1, bias=True)

#         # upsampling function
#         self.upsample_func = upsample_func

#     def forward(self, x):
#         out = self.resblocks(x)
#         out = self.conv_up_cheap(out)
#         out = self.conv_out(out)
#         # out += self.upsample_func(lr_curr)

#         return out
    

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
            self.res_frame_idx = num_frames // 2  # Pick middle frame as default
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
        self.reconstruction = SrResNet(in_channels=num_feat,
                                       out_nc=out_channels,
                                       nf=num_feat,
                                       nb=num_reconstruct_block,
                                       upsample_func=upsample_fn)
        self.reconstruction_channels = num_feat

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Store frame count
        self.num_frames = num_frames

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
        # print(f"Padded shape: {x_padded.shape}")  # Should be (B, T+2*pad, C, H, W)
        
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
        # print(f"HR data shape: {hr_data.shape}")

        # # Save the input patches and the hr data for a sanity check
        # import cv2
        # import numpy as np
        # for i in range(B):
        #     for j, img in enumerate(x[i]):
        #         print(img.shape)
        #         cv2.imwrite(f"test_images/img_{i}_{j}.png", (img.cpu().numpy().transpose((1, 2, 0))* 255).astype(np.uint8))
        #     for j, img in enumerate(x_padded[i]):
        #         print(img.shape)
        #         cv2.imwrite(f"test_images/img_padded_{i}_{j}.png", (img.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))

        return {
            'hr_data': hr_data,  # n,t,c,hr_h,hr_w
        }
    
    def forward(self, x):
        b, t, c, h, w = x.size()
        # print(f'Input size: {x.shape}')
        # if self.hr_in:
        #     assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        # else:
        #     assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        # Extract pyramidal features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.reshape(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.reshape(b, t, -1, h, w)
        feat_l2 = feat_l2.reshape(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.reshape(b, t, -1, h // 4, w // 4)

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
        # print(f"Feature shape: {feat.shape}")
        out = self.reconstruction(feat)
        return out
    

    def infer_sequence(self, lr_data, device):
        # setup params
        tot_frm, c, h, w  = lr_data.size()
        
        # Calculate spatial padding if needed (for multiples of 4)
        h_pad = (4 - h % 4) % 4  # Height padding
        w_pad = (4 - w % 4) % 4  # Width padding
        
        # Apply spatial padding if necessary
        if h_pad > 0 or w_pad > 0:
            lr_data_resized = F.pad(lr_data, (0, w_pad, 0, h_pad), mode='reflect')
            print(f"Adjusted dimensions: {h}x{w} -> {h+h_pad}x{w+w_pad}")
        else:
            lr_data_resized = lr_data
        
        # Pad the sequence by reflection on both sides
        pad = self.num_frames // 2
        
        # Create proper reflection padding
        left_pad = lr_data_resized[1:pad+1, ...].flip(0)  # Take first 'pad' frames after first frame and flip
        right_pad = lr_data_resized[-pad-1:-1, ...].flip(0)  # Take last 'pad' frames before last frame and flip
        
        # Concatenate along time dimension
        lr_data_padded = torch.cat([left_pad, lr_data_resized, right_pad], dim=0)

        # Forward pass
        times = []
        hr_seq = []
        
        for i in tqdm.tqdm(range(tot_frm)):
            with torch.no_grad():
                self.eval()
                
                # Get window for current frame
                window_start = i
                window_end = window_start + self.num_frames
                
                # Extract frames for the window
                lr_window = lr_data_padded[window_start:window_end].to(device)
                
                # Start timing
                start = time()
                
                # Reshape to match expected input format for forward: [B, T, C, H, W]
                lr_window = lr_window.unsqueeze(0)  # Add batch dimension: [1, num_frames, C, H, W]
                
                # Forward pass
                print(lr_window.shape)
                output = self.forward(lr_window)
                
                # Extract only the area corresponding to the original frame (without spatial padding)
                if h_pad > 0 or w_pad > 0:
                    output = output[..., :h*4, :w*4]

                # Get output frame
                hr_frm = output.squeeze(0).cpu()
                
                end = time()
                times.append(end-start)
                
                hr_frm = hr_frm.numpy()
                hr_seq.append(hr_frm)
        
        avg_time = sum(times) / len(times)
        print(f"Average inference time per frame: {avg_time:.4f}s")
        
        return np.stack(hr_seq).transpose(0, 2, 3, 1)


def test_edvr_inference_speed(with_tsa=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDVRNet(with_tsa=with_tsa).to(device)
    model.eval()

    # Input video dummy: (B=1, T=5 frames, C=3, H=64, W=64)
    dummy_input = torch.randn(1, 5, 3, 64, 64).to(device)

    # Warm-up
    for _ in range(5):
        _ = model(dummy_input)

    # Măsurare timp inferență
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):  # rulăm de mai multe ori pentru medie
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    print(f"Average inference time per forward pass: {avg_time:.4f} seconds")



if __name__ == "__main__":
    # Run an inference test speed
    test_edvr_inference_speed()