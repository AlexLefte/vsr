import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def plot_frvsr_comparison(base_path, image_name, bbox_x, bbox_y, bbox_width, bbox_height):
    models = ['frvsr_12_bicubic', 'frvsr_12_bicubic_icme']
    qp_values = [17, 27, 37]
    
    for model in models:
        # Create figure: 3 rows, 3 columns
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'{model}', fontsize=16, fontweight='bold')
        
        # Load GT image
        gt_path = os.path.join(base_path, model, 'gt', image_name)
        gt_img = Image.open(gt_path)
        
        # Row 0: GT image in all 3 columns
        for col in range(3):
            axes[0, col].imshow(gt_img)
            axes[0, col].set_title('Ground Truth')
            
            # Add bounding box
            rect = patches.Rectangle((bbox_x, bbox_y), bbox_width, bbox_height, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            axes[0, col].add_patch(rect)
            axes[0, col].axis('off')
        
        # Rows 1 and 2: LR and Upscaled for each QP
        for col, qp in enumerate(qp_values):
            # Row 1: LR images
            lr_path = os.path.join(base_path, model, 'lr', str(qp), image_name)
            lr_img = Image.open(lr_path)
            lr_crop = lr_img.crop((bbox_x/4, bbox_y/4, bbox_x/4 + bbox_width/4, bbox_y/4 + bbox_height/4))
            
            axes[1, col].imshow(lr_crop)
            axes[1, col].set_title(f'LR QP{qp}')
            axes[1, col].axis('off')
            
            # Row 2: Upscaled images
            up_path = os.path.join(base_path, model, str(qp), image_name)
            up_img = Image.open(up_path)
            up_crop = up_img.crop((bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height))
            
            axes[2, col].imshow(up_crop)
            axes[2, col].set_title(f'Upscaled QP{qp}')
            axes[2, col].axis('off')
        
        plt.tight_layout()
        plt.show()

# Usage
base_path = "model/"
image_name = "0007.png"
bbox_x = 100
bbox_y = 100
bbox_width = 200
bbox_height = 200

plot_frvsr_comparison(base_path, image_name, bbox_x, bbox_y, bbox_width, bbox_height)