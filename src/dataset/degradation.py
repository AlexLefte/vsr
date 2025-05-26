import os
import cv2
from pathlib import Path
import os

def resize_and_save_images(src_dir, dst_dir, scale=0.25, 
                           interpolation='bicubic', image_extensions=('.png', '.jpg', '.jpeg')):
    # Select interpolation method
    interpolation_methods = {
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC
    }

    if interpolation not in interpolation_methods:
        raise ValueError("Interpolation must be 'bilinear' or 'bicubic'.")

    interp_method = interpolation_methods[interpolation]

    clips = sorted(os.listdir(src_dir))
    
    # # Iterăm toate subfolderele
    # clips = []
    # from collections import defaultdict
    # clip_dict = defaultdict(list)
    # max_subclips=20
    # for folder_name in sorted(os.listdir(src_dir)):
    #     if os.path.isdir(os.path.join(src_dir, folder_name)) and '_' in folder_name:
    #         clip_id, subclip_id = folder_name.split('_', 1)

    #         if len(clip_dict[clip_id]) < max_subclips:
    #             clip_dict[clip_id].append(folder_name)
    #             clips.append(folder_name)


    for clip in clips:
        os.makedirs(os.path.join(dst_dir, clip), exist_ok=True)
        files = os.listdir(os.path.join(src_dir, clip))
        for file in files:
            if file.lower().endswith(image_extensions):
                src_img_path = os.path.join(src_dir, clip, file)
                dst_img_path = os.path.join(dst_dir, clip, file)

                # Read and resize image
                img = cv2.imread(str(src_img_path))
                if img is None:
                    print(f"[WARNING] Couldn't read {src_img_path}, skipping.")
                    continue

                new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                resized_img = cv2.resize(img, new_size, interpolation=interp_method)

                # Save resized image
                cv2.imwrite(str(dst_img_path), resized_img)
                print(f"Saved: {dst_img_path}")


if __name__ == "__main__":
    resize_and_save_images(
        src_dir=r'datasets/Vid4/GT',
        dst_dir=r'datasets/Vid4/LR_BI_x4',
        scale=0.25,          
        interpolation='bicubic'          
    )