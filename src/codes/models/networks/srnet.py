import torch
import torch.nn as nn
from models.networks.modules.tsa_module import TSAFusion


class ResidualBlock(nn.Module):
    """ Residual block without batch normalization
    """
    def __init__(self, nf=64, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            activation(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(nn.Module):
    """ Reconstruction & Upsampling network
    """
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upsample_func=None, 
                 scale=4, transp_conv=False, ref_idx=None, with_tsa=False,
                 shallow_feat_res=False, activation=nn.ReLU):
        super(SRNet, self).__init__()

        # input conv. - low frequency information extraction layer
        if with_tsa:
            print("With TSA")
            self.conv_in = TSAFusion(num_frame=in_nc,
                                     num_feat=nf,
                                     res_frame_idx=-1 if ref_idx is None else ref_idx)
        else:
            self.conv_in = nn.Sequential(
                nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True),
                activation(inplace=True))

        # residual blocks - high frequency information extraction
        self.resblocks = nn.Sequential(*[ResidualBlock(nf, activation) for _ in range(nb)])

        # upsampling
        if transp_conv:
            self.conv_up = nn.Sequential(
                nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
                activation(inplace=True),
                nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
                activation(inplace=True))
            conv_out_ch =  nf
        else:
            self.conv_up = nn.Sequential(
                nn.PixelShuffle(scale),
                activation(inplace=True))
            conv_out_ch =  scale

        # output conv.
        self.conv_out = nn.Conv2d(conv_out_ch, out_nc, 3, 1, 1, bias=True)

        # upsampling function
        self.upsample_func = upsample_func
        self.ref_idx = ref_idx
        self.shallow_feat_res = shallow_feat_res

    def forward(self, x, lr=None):
            """ x: input data
            """
            # Shallow feature extraction
            shallow_feat = self.conv_in(x)

            # Deep feature extraction
            deep_feat = self.resblocks(shallow_feat)
            if self.shallow_feat_res:
                deep_feat = deep_feat + shallow_feat

            # Upsampling
            upsampled = self.conv_up(deep_feat)

            # Refinement
            out = self.conv_out(upsampled)

            # Upsample LR and add to the final output
            if self.upsample_func is not None:
                if lr is not None:
                    out += self.upsample_func(lr)
                else:
                    out += self.upsample_func(x)

            return out
    
    def generate_dummy_input(self, lr_size):
        c, lr_h, lr_w = lr_size
        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)

        data_dict = {
            'x': lr_curr,
            'lr': lr_curr,
        }

        return data_dict