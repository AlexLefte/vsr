import os 
import cv2
import numpy as np
from tqdm import tqdm

import torch

from models.networks.egvsr_nets import FRNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-input', type = str , required = True, help = 'path to DVD images')
parser.add_argument('-output', type = str , required = True, help = 'path to the resulted UHD images')
parser.add_argument('-model', type = str , required = True, help = 'path to the trained model')
parser.add_argument('-scale', type = int , default=4, help = 'scale factor')
args = parser.parse_args()

def post_process(image, bit_depth=16):
    image = np.clip(image, 0, 1) * (2 ** bit_depth - 1)

    return image

def main (args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FRNet(in_nc=3, out_nc=3, nf=64, nb=8, degradation='BI', scale=args.scale)
    model.load_state_dict(torch.load(args.model))

    model.eval()
    model.to(device)

    # init_img = cv2.imread(os.path.join(args.input, os.listdir(args.input)[0]))
    first_method_path = os.path.join(args.input, os.listdir(args.input)[0])
    first_clip_path = os.path.join(first_method_path, os.listdir(first_method_path)[0])
    init_img = cv2.imread(os.path.join(first_clip_path, os.listdir(first_clip_path)[0]))
    h, w, c = init_img.shape
    s = args.scale

    #define initial prev frames
    lr_prev = torch.zeros(1, c, h, w, dtype=torch.float32).to(device)
    hr_prev = torch.zeros(1, c, s * h, s * w, dtype=torch.float32).to(device)

    os.makedirs(args.output, exist_ok=True)
    
    gt_dtype = np.uint8
    methods = os.listdir(args.input)
    for method in methods:
        if method != 'Lanczos_Custom_3':
            continue
        method_path = os.path.join(args.input, method)
        clips = os.listdir(method_path)
        for clip in clips:
            output_path = os.path.join(args.output, method, clip)
            os.makedirs(output_path, exist_ok=True)
            clip_path = os.path.join(args.input, method, clip)
            names = os.listdir(clip_path)
            for name in tqdm(names):
                with torch.no_grad():
                    lr_curr_img = cv2.imread(os.path.join(clip_path, name))
                    lr_curr_img = cv2.cvtColor(lr_curr_img, cv2.COLOR_BGR2RGB) #bgr2rgb

                    lr_curr_img = lr_curr_img.astype(np.float32) / 255.0
                    lr_curr_img = torch.from_numpy(np.ascontiguousarray(lr_curr_img))
                    lr_curr_img.unsqueeze_(0)

                    lr_curr_img = lr_curr_img.permute(0, 3, 1, 2)

                    lr_curr_img = lr_curr_img.to(device)
                    torch.cuda.synchronize()
                    hr_curr =  model(lr_curr_img, lr_prev, hr_prev)

                    lr_prev, hr_prev = lr_curr_img, hr_curr
                    hr_frm = hr_curr.squeeze(0).cpu().detach().numpy()

                    torch.cuda.synchronize()
                    hr_frm = hr_frm.transpose(1, 2, 0)

                    hr_frm = post_process(hr_frm, bit_depth=8)

                    hr_frm = hr_frm[..., ::-1] 

                    cv2.imwrite(os.path.join(output_path, name), hr_frm.astype(gt_dtype))


if __name__== "__main__":
    main(args)
