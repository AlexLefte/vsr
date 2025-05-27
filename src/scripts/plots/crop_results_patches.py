import os
from PIL import Image, ImageDraw


def save_crops(gt_path, pred_paths, crop_coords, descriptions, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Accept either a single bounding box or a list of them
    if isinstance(crop_coords[0], (int, float)):
        crop_coords = [crop_coords]

    # --- Load Ground Truth image ---
    gt_img = Image.open(gt_path)
    gt_w, gt_h = gt_img.size
    gt_draw_img = gt_img.copy()  # Copy for drawing boxes on the full image

    for i, (x, y, w, h) in enumerate(crop_coords):
        # Absolute coordinates for the crop
        left = max(x, 0)
        upper = max(y, 0)
        right = min(x + w, gt_w)
        lower = min(y + h, gt_h)

        gt_crop = gt_img.crop((left, upper, right, lower))

        # Draw bounding box on the crop itself
        draw = ImageDraw.Draw(gt_crop)
        draw.rectangle([0, 0, right - left, lower - upper], outline="red", width=3)

        # Save GT crop
        gt_crop_path = os.path.join(output_dir, f"GT_cropped_{i}.png")
        gt_crop.save(gt_crop_path)
        print(f"Saved GT crop {i}: {gt_crop_path}")

        # Draw the box also on the full-size GT image for reference
        draw_full = ImageDraw.Draw(gt_draw_img)
        draw_full.rectangle([x, y, x + w, y + h], outline="red", width=3)

        # --- Process and save each predicted image crop ---
        for pred_path, desc in zip(pred_paths, descriptions):
            pred_img = Image.open(pred_path)
            pred_w, pred_h = pred_img.size

            # Ensure crop coordinates are within prediction image bounds
            crop_left = max(x, 0)
            crop_upper = max(y, 0)
            crop_right = min(x + w, pred_w)
            crop_lower = min(y + h, pred_h)

            cropped = pred_img.crop((crop_left, crop_upper, crop_right, crop_lower))

            safe_desc = desc.replace(" ", "_").replace("/", "_")
            save_path = os.path.join(output_dir, f"{safe_desc}_crop_{i}.png")
            cropped.save(save_path)
            print(f"Saved crop: {save_path}")

    # Save the full GT image with all bounding boxes drawn
    draw_full_path = os.path.join(output_dir, "GT_full_with_boxes.png")
    gt_draw_img.save(draw_full_path)
    print(f"Saved full GT image with boxes: {draw_full_path}")


# Exemplu de apel
sequence = 'clip_000'
img = '0094'
gt_img = 'images_0095.png'
pred_root = 'D:/Repos/sample_vsr/results/ICMP_27'
gt_path = f"D:/Repos/sample_vsr/datasets/MicrosoftVSR/frames/val_gt/{sequence}/{gt_img}"

pred_paths = [
    os.path.join(pred_root, f"frvsr_12_bicubic/{sequence}/{img}.png"),
    os.path.join(pred_root, f"frvsr_12_bicubic_icmp/{sequence}/{img}.png"),
    os.path.join(pred_root, f"frvsr_12_gan/{sequence}/{img}.png"),
    os.path.join(pred_root, f"edvr_10/{sequence}/{img}.png"),
    os.path.join(pred_root, f"edvr_10_icme/{sequence}/{img}.png"),
    os.path.join(pred_root, f"basicvsr/{sequence}/{gt_img}")
]
# descriptions = ["Lanczos", "SRNet", "FRVSR+", "FRVSR+/TC", "DcnVSR", "EDVR"]
descriptions = ["FRVSR+", "FRVSR+_f", "FRVSR+_adv.", "edvr10", "edvr10_f", "basicvsr"]
output_dir = f"D:/Updates/27.05.2025 - diagrams/diagrams/icme/27/{sequence}"

crop_coords = [
    (700, 420, 200, 100),  # box 1
    (795, 0, 200, 100),   # box 2
    (820, 245, 200, 100)   # box 3
]
save_crops(gt_path, pred_paths, crop_coords, descriptions, output_dir)
