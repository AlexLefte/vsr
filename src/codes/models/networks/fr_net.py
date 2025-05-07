import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_net import FNet
from srres_net import SrResNet
from time import time
from models.networks.base_nets import BaseSequenceGenerator
from utils.net_utils import space_to_depth, backward_warp, get_upsampling_func
import tqdm


# -------------------- generator modules -------------------- #
class FRNet(BaseSequenceGenerator):
    """ Frame-recurrent network proposed in https://arxiv.org/abs/1801.04590
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, mode='bilinear',
                 scale=4, transp_conv=False):
        super(FRNet, self).__init__()

        self.scale = scale
        self.reconstruction_channels = (scale**2 + 1) * in_nc  # SR input channels

        # get upsampling function according to the degradation mode
        self.upsample_func = get_upsampling_func(self.scale, mode)

        # define fnet & srnet
        self.fnet = FNet(in_nc)
        self.srnet = SrResNet(self.reconstruction_channels, out_nc, nf, nb,
                              self.upsample_func, transp_conv=transp_conv, ref_idx=0)

    def generate_dummy_input(self, lr_size):
        c, lr_h, lr_w = lr_size
        s = self.scale

        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        lr_prev = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        hr_prev = torch.rand(1, c, s * lr_h, s * lr_w, dtype=torch.float32)

        data_dict = {
            'lr_curr': lr_curr,
            'lr_prev': lr_prev,
            'hr_prev': hr_prev
        }

        return data_dict

    def forward(self, lr_curr, lr_prev, hr_prev):
        """
            Parameters:
                :param lr_curr: the current lr data in shape nchw
                :param lr_prev: the previous lr data in shape nchw
                :param hr_prev: the previous hr data in shape nc(4h)(4w)
        """
        # start = time()
        # estimate lr flow (lr_curr -> lr_prev)
        lr_flow = self.fnet(lr_curr, lr_prev)

        # pad if size is not a multiple of 8
        pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), 'reflect')

        # upsample lr flow
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)

        # warp hr_prev
        hr_prev_warp = backward_warp(hr_prev, hr_flow)
        hr_prev_aligned = space_to_depth(hr_prev_warp, self.scale)
        # torch.cuda.synchronize()
        # end = time()

        # compute hr_curr
        #  start1 = time()

        hr_curr = self.srnet(torch.cat([lr_curr, hr_prev_aligned], dim=1))
        # torch.cuda.synchronize()
        # end1 = time()

        # return hr_curr, end - start, end1 - start1
        return hr_curr

    def forward_sequence(self, lr_data):
        """
            Parameters:
                :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_flow = self.fnet(torch.cat((lr_prev, lr_curr), dim=1))

        # upsample lr flows
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w)

        # compute the first hr data
        hr_data = []
        sr_input = torch.cat([lr_data[:, 0, ...],
                    torch.zeros(n, (self.scale**2)*c, lr_h, lr_w, dtype=torch.float32,
                        device=lr_data.device)], dim=1)
        hr_prev = self.srnet(sr_input)
        hr_data.append(hr_prev)

        # compute the remaining hr data
        for i in range(1, t):
            # warp hr_prev
            hr_prev_warp = backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # compute hr_curr
            sr_input = torch.cat([lr_data[:, i, ...],
                                  space_to_depth(hr_prev_warp, self.scale)], dim=1)
            hr_curr = self.srnet(sr_input)

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w

        # construct output dict
        ret_dict = {
            'hr_data': hr_data,  # n,t,c,hr_h,hr_w
            'hr_flow': hr_flow,  # n,t,2,hr_h,hr_w
            'lr_prev': lr_prev,  # n(t-1),c,lr_h,lr_w
            'lr_curr': lr_curr,  # n(t-1),c,lr_h,lr_w
            'lr_flow': lr_flow,  # n(t-1),2,lr_h,lr_w
        }

        return ret_dict

    def infer_sequence(self, lr_data, device):
        """
            Parameters:
                :param lr_data: torch.FloatTensor in shape tchw
                :param device: torch.device

                :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # setup params
        tot_frm, c, h, w = lr_data.size()
        s = self.scale

        # forward
        times = []
        hr_seq = []
        lr_prev = torch.zeros(1, c, h, w, dtype=torch.float32).to(device)
        hr_prev = torch.zeros(
            1, c, s * h, s * w, dtype=torch.float32).to(device)

        for i in tqdm.tqdm(range(tot_frm)):
            with torch.no_grad():
                self.eval()
                lr_curr = lr_data[i: i + 1, ...]
                start = time()
                lr_curr = lr_curr.cuda()
                # end1 = time()
                # print(f'Load time: {end1-start}')
                hr_curr = self.forward(lr_curr, lr_prev, hr_prev)
                torch.cuda.synchronize()
                # end2 = time()
                # print(f"Inf time: {end2-end1}")
                hr_frm = hr_curr.squeeze(0).cpu()
                torch.cuda.synchronize()
                end = time()        
                # print(f"\nGPU -> CPU time: {end-end2}")
                times.append(end-start)
                lr_prev, hr_prev = lr_curr, hr_curr

                # hr_frm = hr_curr.squeeze(0).cpu().numpy()  # chw|rgb|uint8
                hr_frm = hr_frm.numpy()  # chw|rgb|uint8
            # hr_seq.append(float32_to_uint8(hr_frm))
            hr_seq.append(hr_frm)
        return np.stack(hr_seq).transpose(0, 2, 3, 1)  # thwc


# if __name__ == "__main__":
#     from torchsummary import summary

#     img_size = (960, 640)
#     cpu_cuda = 'cuda'

#     device = torch.device(cpu_cuda)

#     model = ResidualBlock()
#     model.eval()
#     model.to(device)

#     print(model)
#     summary(model, (64, *img_size), batch_size=1, device=cpu_cuda)