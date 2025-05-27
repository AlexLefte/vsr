import os
import argparse
import numpy as np
import cv2
import torch
import lpips
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import normalize
from tqdm import tqdm


def rgb_to_ycbcr(img):
    T = np.array([
        [0.256788235294118, -0.148223529411765,  0.439215686274510],
        [0.504129411764706, -0.290992156862745, -0.367788235294118],
        [0.097905882352941,  0.439215686274510, -0.071427450980392],
    ])
    O = np.array([16, 128, 128])
    img = img.astype(np.float64)
    res = np.matmul(img, T) + O
    return res.clip(0, 255).round().astype(np.uint8)


def compute_psnr(gt, pred, bit_depth=8, colorspace='y'):
    if colorspace != 'rgb':
        gt = rgb_to_ycbcr(gt)[..., 0]
        pred = rgb_to_ycbcr(pred)[..., 0]
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_val = (2 ** bit_depth) - 1
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(gt, pred, bit_depth=8):
    gt_y = rgb_to_ycbcr(gt)[..., 0]
    pred_y = rgb_to_ycbcr(pred)[..., 0]
    return ssim(gt_y, pred_y, data_range=(2**bit_depth - 1), channel_axis=None)


def compute_lpips(gt, pred, lpips_model):
    gt_tensor = torch.FloatTensor(gt).permute(2, 0, 1).unsqueeze(0)
    pred_tensor = torch.FloatTensor(pred).permute(2, 0, 1).unsqueeze(0)
    normalize(gt_tensor, [0.5]*3, [0.5]*3)
    normalize(pred_tensor, [0.5]*3, [0.5]*3)
    with torch.no_grad():
        dist = lpips_model(gt_tensor, pred_tensor)
    return dist.item()


def compute_tOF(gt, pred, gt_prev, pred_prev):
    gt_cur = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    pred_cur = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    gt_prev = cv2.cvtColor(gt_prev, cv2.COLOR_RGB2GRAY)
    pred_prev = cv2.cvtColor(pred_prev, cv2.COLOR_RGB2GRAY)

    flow_gt = cv2.calcOpticalFlowFarneback(gt_prev, gt_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_pred = cv2.calcOpticalFlowFarneback(pred_prev, pred_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    epe = np.sqrt(np.sum((flow_gt - flow_pred) ** 2, axis=-1)).mean()
    return epe


def evaluate_metrics(gt_dir, pred_dir, bit_depth=8):
    lpips_model = lpips.LPIPS(net='alex')

    psnr_scores, ssim_scores, lpips_scores, tof_scores = [], [], [], []

    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    all_files = sorted([f for f in gt_dir.rglob("*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']])
    
    gt_prev, pred_prev = None, None

    for gt_path in tqdm(all_files, desc="Evaluating"):
        rel_path = gt_path.relative_to(gt_dir)
        pred_path = pred_dir / rel_path
        if not pred_path.exists():
            print(f"[WARNING] Prediction missing: {pred_path}")
            continue

        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if gt_img is None or pred_img is None:
            print(f"[ERROR] Cannot read one of the images: {gt_path} or {pred_path}")
            continue

        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

        try:
            psnr = compute_psnr(gt_img, pred_img, bit_depth)
            ssim_val = compute_ssim(gt_img, pred_img, bit_depth)
            lpips_val = compute_lpips(gt_img, pred_img, lpips_model)

            psnr_scores.append(psnr)
            ssim_scores.append(ssim_val)
            lpips_scores.append(lpips_val)

            if gt_prev is not None and pred_prev is not None:
                tof_val = compute_tOF(gt_img, pred_img, gt_prev, pred_prev)
                tof_scores.append(tof_val)

            gt_prev = gt_img
            pred_prev = pred_img

        except Exception as e:
            print(f"[ERROR] Metric error for {rel_path}: {e}")

    print("\n==== Average Metrics ====")
    print(f"PSNR:  {np.mean(psnr_scores):.2f}")
    print(f"SSIM:  {np.mean(ssim_scores):.4f}")
    print(f"LPIPS: {np.mean(lpips_scores):.4f}")
    if tof_scores:
        print(f"tOF:   {np.mean(tof_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PSNR, SSIM, LPIPS, tOF between GT and predicted images.")
    parser.add_argument('--gt', type=str, required=True, help="Path to ground-truth images folder.")
    parser.add_argument('--pred', type=str, required=True, help="Path to predicted images folder.")
    parser.add_argument('--bit_depth', type=int, default=8, help="Bit depth of ground-truth images.")
    args = parser.parse_args()

    evaluate_metrics(args.gt, args.pred, bit_depth=args.bit_depth)
