import os
import cv2
import lpips
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import normalize
import yaml
from tqdm import tqdm
import torch


def rgb_to_ycbcr(img):
    """ Coefficients are taken from the  official codes of DUF-VSR
        This conversion is also the same as that in BasicSR

        Parameters:
            :param  img: rgb image in type np.uint8
            :return: ycbcr image in type np.uint8
    """

    T = np.array([
        [0.256788235294118, -0.148223529411765,  0.439215686274510],
        [0.504129411764706, -0.290992156862745, -0.367788235294118],
        [0.097905882352941,  0.439215686274510, -0.071427450980392],
    ], dtype=np.float64)

    O = np.array([16, 128, 128], dtype=np.float64)

    img = img.astype(np.float64)
    res = np.matmul(img, T) + O
    res = res.clip(0, 255).round().astype(np.uint8)

    return res

def compute_psnr(true_img, pred_img, gt_bit_depth, colorspace='y'):
    if colorspace != 'rgb':
        true_img = rgb_to_ycbcr(true_img)[..., 0]
        pred_img = rgb_to_ycbcr(pred_img)[..., 0]
    
    diff = true_img.astype(np.float64) - pred_img.astype(np.float64)
    rmse = np.sqrt(np.mean(diff ** 2))
    if rmse == 0:
        return float('inf')
    return 20 * np.log10((2 ** gt_bit_depth - 1) / rmse)

def compute_ssim(true_img, pred_img, gt_bit_depth):
    true_y = rgb_to_ycbcr(true_img)[..., 0]
    pred_y = rgb_to_ycbcr(pred_img)[..., 0]
    return ssim(true_y, pred_y, data_range=(2 ** gt_bit_depth - 1), channel_axis=None)

def compute_lpips(true_img, pred_img, lpips_model):
    true_img = np.ascontiguousarray(true_img)
    pred_img = np.ascontiguousarray(pred_img)

    # convert to tensor, normalize to [-1, 1]
    true_tensor = torch.FloatTensor(true_img).permute(2, 0, 1).unsqueeze(0)
    pred_tensor = torch.FloatTensor(pred_img).permute(2, 0, 1).unsqueeze(0)
    normalize(true_tensor, [0.5]*3, [0.5]*3)
    normalize(pred_tensor, [0.5]*3, [0.5]*3)

    with torch.no_grad():
        dist = lpips_model(true_tensor, pred_tensor)
    return dist.item()

def upscale_lanczos(opt_path, dest_dir, save_images=True, compute_metrics=True):
    with open(opt_path, 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)

    gt_bit_depth = opt.get('gt_bit_depth', 8)
    gt_dtype = np.uint8 if gt_bit_depth == 8 else np.uint16
    scale_factor = opt.get('scale', 4)
    lpips_model = lpips.LPIPS(net='alex') if compute_metrics else None

    for dataset_idx in sorted(opt['dataset'].keys()):
        if not dataset_idx.startswith('test'):
            continue

        ds_cfg = opt['dataset'][dataset_idx]
        ds_name = ds_cfg['name']
        lr_dir = Path(ds_cfg['lr_seq_dir'])
        gt_dir = Path(ds_cfg['gt_seq_dir']) if 'gt_seq_dir' in ds_cfg else None
        dest_path = Path(dest_dir) / ds_name
        dest_path.mkdir(parents=True, exist_ok=True)

        if compute_metrics and gt_dir is None:
            print("[ERROR] No GT path provided in opt file.")
            compute_metrics = False
            continue

        print(f"[INFO] Processing dataset: {ds_name}")
        psnr_total, ssim_total, lpips_total = [], [], []

        for root, _, files in tqdm(os.walk(lr_dir)):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    continue

                src_path = Path(root) / file
                relative_path = src_path.relative_to(lr_dir)
                out_path = dest_path / relative_path
                out_path.parent.mkdir(parents=True, exist_ok=True)

                img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[ERROR] Can't read: {src_path}")
                    continue

                height, width = img.shape[:2]
                new_size = (int(width * scale_factor), int(height * scale_factor))
                upscaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

                if save_images:
                    cv2.imwrite(str(out_path), upscaled_img)

                if compute_metrics:
                    gt_path = gt_dir / relative_path
                    if not gt_path.exists():
                        print(f"[WARNING] No ground truth for: {gt_path}")
                        continue

                    gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
                    if gt_img is None:
                        print(f"[ERROR] Can't read: {gt_path}")
                        continue

                    gt_img = cv2.resize(gt_img, new_size, interpolation=cv2.INTER_CUBIC)

                    try:
                        psnr = compute_psnr(gt_img, upscaled_img, gt_bit_depth)
                        ssim_val = compute_ssim(gt_img, upscaled_img, gt_bit_depth)
                        lpips_val = compute_lpips(gt_img, upscaled_img, lpips_model)
                        ssim_val = 0
                        lpips_val = 0

                        if not np.isinf(psnr):
                            psnr_total.append(psnr)
                            ssim_total.append(ssim_val)
                            lpips_total.append(lpips_val)

                        # print(f"[METRIC] {relative_path}: PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
                    except Exception as e:
                        print(f"[ERROR] Metric error for {relative_path}: {e}")

        if compute_metrics and psnr_total:
            print("\n===== Average Metrics =====")
            print(f"PSNR:  {np.mean(psnr_total):.2f}")
            print(f"SSIM:  {np.mean(ssim_total):.4f}")
            print(f"LPIPS: {np.mean(lpips_total):.4f}")


# Exemplu de rulare
opt_path = "src/config/test_frvsr.yml"
destination_folder = 'D:/Repos/sample_vsr/results/'

upscale_lanczos(opt_path=opt_path, dest_dir=destination_folder, save_images=False)
