import torch.nn as nn


class ResidualBlock(nn.Module):
    """ Residual block without batch normalization
    """
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SrResNet(nn.Module):
    """ Reconstruction & Upsampling network
    """
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upsample_func=None, 
                 scale=4, transp_conv=False, ref_idx=None):
        super(SrResNet, self).__init__()

        # input conv. - low frequency information extraction layer
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks - high frequency information extraction
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        if transp_conv:
            self.conv_up = nn.Sequential(
                nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
                nn.ReLU(inplace=True))
            conv_out_ch =  nf
        else:
            self.conv_up = nn.Sequential(
                nn.PixelShuffle(scale),
                nn.ReLU(inplace=True))
            conv_out_ch =  scale

        # output conv.
        self.conv_out = nn.Conv2d(conv_out_ch, out_nc, 3, 1, 1, bias=True)

        # upsampling function
        self.upsample_func = upsample_func
        self.ref_idx = ref_idx

    def forward(self, x, lr=None):
        """ x: input data
        """
        # Shallow feature extraction
        out = self.conv_in(x)

        # Deep feature extraction
        out = self.resblocks(out)

        # Upsampling
        out = self.conv_up(out)
        out = self.conv_out(out)

        # Upsample LR and add to the final output
        if self.upsample_func is not None and lr is not None:
            if len(x.shape) == 5:
                out += self.upsample_func(lr[:, self.ref_idx, :, :, :].squeeze(1))
            elif lr is not None:
                out += self.upsample_func(lr)
            else:
                out += self.upsample_func(x)

        return out