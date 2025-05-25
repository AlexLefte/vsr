# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# import numpy as np
# import os
# import glob

# class VSRMultiModelComparison:
#     def __init__(self, base_dir, image_filename=None, gt_dir=None, gt_filename=None):
#         """
#         Initialize VSR comparison tool for multiple model directories
        
#         Args:
#             base_dir (str): Base directory containing model subdirectories
#             image_filename (str): Specific image filename to load from each model directory
#             gt_dir (str): Directory containing ground truth images (optional)
#             gt_filename (str): Specific ground truth filename (optional)
#         """
#         self.base_dir = base_dir
#         self.image_filename = image_filename
#         self.gt_dir = gt_dir
#         self.gt_filename = gt_filename
#         self.images = {}
#         self.model_order = []
#         self.gt_image = None
        
#         # Define the expected model folders and their display names
#         self.model_mapping = {
#             'GT': 'Ground Truth',
#             'Lanczos': 'Lanczos (Baseline)',
#             'dcn_12': 'DCN',
#             'edvr_10': 'EDVR',
#             'frvsr_12_bicubic': 'FRVSR Bicubic',
#             'frvsr_12_bicubic_icme': 'FRVSR Bicubic ICME',
#             'frvsr_12_gan_vanilla': 'FRVSR GAN',
#             'frvsr_12_transp_conv': 'FRVSR TranspConv',
#             'srnet_12_bicubic': 'SRNet Bicubic'
#         }
        
#         self.load_ground_truth()
#         self.load_model_images()
    
#     def load_ground_truth(self):
#         """Load ground truth image if specified"""
#         if self.gt_dir and os.path.exists(self.gt_dir):
#             print(f"Loading ground truth from: {self.gt_dir}")
            
#             # Get all image files from GT directory
#             extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
#             gt_files = []
            
#             for ext in extensions:
#                 gt_files.extend(glob.glob(os.path.join(self.gt_dir, ext)))
#                 gt_files.extend(glob.glob(os.path.join(self.gt_dir, ext.upper())))
            
#             if gt_files:
#                 # If specific GT filename is provided, look for it
#                 if self.gt_filename:
#                     target_path = os.path.join(self.gt_dir, self.gt_filename)
#                     if os.path.exists(target_path):
#                         self.gt_image = np.array(Image.open(target_path).convert('RGB'))
#                         print(f"  ✓ Loaded GT: {self.gt_filename} - {self.gt_image.shape}")
#                     else:
#                         # Look for the filename in the found images
#                         for gt_path in gt_files:
#                             if os.path.basename(gt_path) == self.gt_filename:
#                                 self.gt_image = np.array(Image.open(gt_path).convert('RGB'))
#                                 print(f"  ✓ Loaded GT: {self.gt_filename} - {self.gt_image.shape}")
#                                 break
#                         else:
#                             print(f"  Warning: {self.gt_filename} not found in GT directory, using first available")
#                             gt_files.sort()
#                             self.gt_image = np.array(Image.open(gt_files[0]).convert('RGB'))
#                             print(f"  ✓ Loaded GT: {os.path.basename(gt_files[0])} - {self.gt_image.shape}")
#                 else:
#                     # Use first available GT image
#                     gt_files.sort()
#                     self.gt_image = np.array(Image.open(gt_files[0]).convert('RGB'))
#                     print(f"  ✓ Loaded GT: {os.path.basename(gt_files[0])} - {self.gt_image.shape}")
#             else:
#                 print(f"  Warning: No GT images found in {self.gt_dir}")
#         else:
#             print("No ground truth directory specified or directory doesn't exist")
    
#     def find_model_directories(self):
#         """Find all model directories in base directory"""
#         model_dirs = []
        
#         if not os.path.exists(self.base_dir):
#             raise ValueError(f"Base directory {self.base_dir} does not exist")
        
#         # Get all subdirectories
#         for item in os.listdir(self.base_dir):
#             item_path = os.path.join(self.base_dir, item)
#             if os.path.isdir(item_path):
#                 model_dirs.append(item)
        
#         # Sort with Lanczos first (baseline), then alphabetically
#         def sort_key(model_name):
#             if model_name == 'Lanczos':
#                 return '0_Lanczos'  # Ensure Lanczos comes first
#             return model_name
        
#         model_dirs.sort(key=sort_key)
#         return model_dirs
    
#     def load_image_from_directory(self, model_dir):
#         """Load image from a specific model directory"""
#         dir_path = os.path.join(self.base_dir, model_dir)
        
#         # Get all image files
#         extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
#         image_files = []
        
#         for ext in extensions:
#             image_files.extend(glob.glob(os.path.join(dir_path, ext)))
#             image_files.extend(glob.glob(os.path.join(dir_path, ext.upper())))
        
#         if not image_files:
#             print(f"Warning: No images found in {dir_path}")
#             return None
        
#         # If specific filename is provided, look for it
#         if self.image_filename:
#             target_path = os.path.join(dir_path, self.image_filename)
#             if os.path.exists(target_path):
#                 return Image.open(target_path).convert('RGB')
#             else:
#                 # Look for the filename in the found images
#                 for img_path in image_files:
#                     if os.path.basename(img_path) == self.image_filename:
#                         return Image.open(img_path).convert('RGB')
#                 print(f"Warning: {self.image_filename} not found in {dir_path}, using first available image")
        
#         # Sort files for consistent ordering and use the first one
#         image_files.sort()
#         return Image.open(image_files[0]).convert('RGB')
    
#     def load_model_images(self):
#         """Load images from all model directories"""
#         model_dirs = self.find_model_directories()
        
#         print(f"Found {len(model_dirs)} model directories:")
        
#         for model_dir in model_dirs:
#             print(f"  Loading from: {model_dir}")
#             img = self.load_image_from_directory(model_dir)
            
#             if img is not None:
#                 self.images[model_dir] = np.array(img)
#                 self.model_order.append(model_dir)
#                 print(f"    ✓ Loaded image: {self.images[model_dir].shape}")
#             else:
#                 print(f"    ✗ Failed to load image from {model_dir}")
        
#         if not self.images:
#             raise ValueError("No images were successfully loaded")
        
#         print(f"\nSuccessfully loaded {len(self.images)} images")
    
#     def extract_bbox_region(self, image, x, y, width, height):
#         """Extract bounding box region from image"""
#         # Ensure coordinates are within image bounds
#         h, w = image.shape[:2]
#         x = max(0, min(x, w - width))
#         y = max(0, min(y, h - height))
#         width = min(width, w - x)
#         height = min(height, h - y)
        
#         return image[y:y+height, x:x+width]
    
#     def visualize_bbox_selection(self, x, y, width, height):
#         """Show the GT or baseline image with bounding box overlay for verification"""
#         # Use GT if available, otherwise use Lanczos (baseline), or first available image
#         if self.gt_image is not None:
#             reference_img = self.gt_image
#             reference_name = self.model_mapping['GT']
#             title_suffix = " (Ground Truth)"
#         elif 'Lanczos' in self.images:
#             reference_img = self.images['Lanczos']
#             reference_name = self.model_mapping['Lanczos']
#             title_suffix = " (Baseline)"
#         else:
#             baseline_key = self.model_order[0]
#             reference_img = self.images[baseline_key]
#             reference_name = self.model_mapping.get(baseline_key, baseline_key)
#             title_suffix = ""
        
#         fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
#         ax.imshow(reference_img)
        
#         # Draw bounding box
#         rect = patches.Rectangle((x, y), width, height, 
#                                linewidth=3, edgecolor='red', facecolor='none')
#         ax.add_patch(rect)
        
#         ax.set_title(f'{reference_name} with Selected Region{title_suffix}', fontsize=14, fontweight='bold')
#         ax.set_xlabel(f'Bounding Box: ({x}, {y}) - {width}×{height}')
#         ax.axis('on')
        
#         plt.tight_layout()
#         plt.show()
    
#     def create_comparison_plot(self, x, y, width, height, save_path=None):
#         """
#         Create comparison plot showing GT (if available) and all models with their cropped regions
        
#         Args:
#             x, y (int): Top-left corner coordinates of bounding box
#             width, height (int): Dimensions of bounding box
#             save_path (str): Path to save the figure (optional)
#         """
#         # Extract regions from all images
#         regions = {}
#         full_images = {}
        
#         # Add GT if available
#         if self.gt_image is not None:
#             full_images['GT'] = self.gt_image
#             regions['GT'] = self.extract_bbox_region(self.gt_image, x, y, width, height)
        
#         # Add model images
#         for model_key in self.model_order:
#             img = self.images[model_key]
#             full_images[model_key] = img
#             regions[model_key] = self.extract_bbox_region(img, x, y, width, height)
        
#         # Determine layout
#         all_keys = (['GT'] if self.gt_image is not None else []) + self.model_order
#         n_models = len(all_keys)
        
#         # Create figure - each model gets full image + crop
#         n_cols = 2
#         n_rows = (n_models + 1) // 2
#         fig = plt.figure(figsize=(20, 4 * n_rows))
        
#         from matplotlib import gridspec
#         gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
        
#         for idx, key in enumerate(all_keys):
#             row = idx // n_cols
#             col = idx % n_cols
            
#             # Create subplot with nested grid for full image + crop
#             gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, gs[row, col], 
#                                                      width_ratios=[2, 1], wspace=0.1)
            
#             # Full image with bounding box
#             ax_full = fig.add_subplot(gs_sub[0, 0])
#             ax_full.imshow(full_images[key])
#             rect = patches.Rectangle((x, y), width, height, 
#                                    linewidth=2, edgecolor='red', facecolor='none')
#             ax_full.add_patch(rect)
            
#             model_name = self.model_mapping.get(key, key)
#             # Add special styling for GT
#             if key == 'GT':
#                 ax_full.set_title(f'{model_name}', fontsize=12, fontweight='bold', 
#                                 color='darkgreen', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
#             else:
#                 ax_full.set_title(f'{model_name}', fontsize=12, fontweight='bold')
#             ax_full.axis('off')
            
#             # Cropped region
#             ax_crop = fig.add_subplot(gs_sub[0, 1])
#             ax_crop.imshow(regions[key])
#             ax_crop.set_title('Detail', fontsize=10)
#             ax_crop.axis('off')
            
#             # Add border around crop for clarity
#             border_color = 'darkgreen' if key == 'GT' else 'black'
#             border_width = 2 if key == 'GT' else 1
#             for spine in ax_crop.spines.values():
#                 spine.set_visible(True)
#                 spine.set_color(border_color)
#                 spine.set_linewidth(border_width)
        
#         # Add main title
#         title_prefix = "VSR Model Comparison with Ground Truth" if self.gt_image is not None else "VSR Model Comparison"
#         fig.suptitle(f'{title_prefix} - Region: ({x}, {y}) - {width}×{height}', 
#                     fontsize=16, fontweight='bold', y=0.98)
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"Figure saved to: {save_path}")
        
#         plt.show()
    
#     def create_crops_only_comparison(self, x, y, width, height, save_path=None):
#         """
#         Create a compact comparison showing only the cropped regions including GT
        
#         Args:
#             x, y (int): Top-left corner coordinates of bounding box
#             width, height (int): Dimensions of bounding box
#             save_path (str): Path to save the figure (optional)
#         """
#         # Extract regions from all images
#         regions = {}
        
#         # Add GT if available
#         if self.gt_image is not None:
#             regions['GT'] = self.extract_bbox_region(self.gt_image, x, y, width, height)
        
#         # Add model regions
#         for model_key in self.model_order:
#             img = self.images[model_key]
#             regions[model_key] = self.extract_bbox_region(img, x, y, width, height)
        
#         # Determine layout
#         all_keys = (['GT'] if self.gt_image is not None else []) + self.model_order
#         n_models = len(all_keys)
        
#         # Create figure for crops only
#         fig = plt.figure(figsize=(16, 8))
        
#         # Calculate grid layout
#         n_cols = 4
#         n_rows = (n_models + n_cols - 1) // n_cols
        
#         for idx, key in enumerate(all_keys):
#             ax = plt.subplot(n_rows, n_cols, idx + 1)
#             ax.imshow(regions[key])
            
#             model_name = self.model_mapping.get(key, key)
            
#             # Special styling for GT
#             if key == 'GT':
#                 ax.set_title(model_name, fontsize=11, fontweight='bold', 
#                            color='darkgreen', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
#                 border_color = 'darkgreen'
#                 border_width = 2
#             else:
#                 ax.set_title(model_name, fontsize=10, fontweight='bold')
#                 border_color = 'black'
#                 border_width = 1
            
#             ax.axis('off')
            
#             # Add border for clarity
#             for spine in ax.spines.values():
#                 spine.set_visible(True)
#                 spine.set_color(border_color)
#                 spine.set_linewidth(border_width)
        
#         # Add main title
#         title_prefix = "VSR Comparison with GT (Crops Only)" if self.gt_image is not None else "VSR Model Comparison (Crops Only)"
#         fig.suptitle(f'{title_prefix} - Region: ({x}, {y}) - {width}×{height}', 
#                     fontsize=14, fontweight='bold', y=0.95)
        
#         plt.tight_layout()
        
#         if save_path:
#             crops_save_path = save_path.replace('.png', '_crops_only.png')
#             plt.savefig(crops_save_path, dpi=300, bbox_inches='tight')
#             print(f"Crops-only figure saved to: {crops_save_path}")
        
#         plt.show()
    
#     def update_model_names(self, model_dict):
#         """Update model names for captions"""
#         self.model_mapping.update(model_dict)

# # Convenience function
# def run_multi_model_vsr_comparison(base_dir, bbox_coords, image_filename=None, 
#                                  gt_dir=None, gt_filename=None, custom_model_names=None, 
#                                  save_path=None, show_crops_only=True):
#     """
#     Convenience function to run multi-model VSR comparison with Ground Truth
    
#     Args:
#         base_dir (str): Base directory containing model subdirectories
#         bbox_coords (tuple): (x, y, width, height) for bounding box
#         image_filename (str): Specific image filename to load (optional)
#         gt_dir (str): Directory containing ground truth images (optional)
#         gt_filename (str): Specific ground truth filename (optional)
#         custom_model_names (dict): Custom model names (optional)
#         save_path (str): Path to save figure (optional)
#         show_crops_only (bool): Whether to show crops-only comparison
#     """
#     # Initialize comparison
#     vsr = VSRMultiModelComparison(base_dir, image_filename, gt_dir, gt_filename)
    
#     # Update model names if provided
#     if custom_model_names:
#         vsr.update_model_names(custom_model_names)
    
#     # Extract coordinates
#     x, y, width, height = bbox_coords
    
#     # Show bbox selection
#     print("Showing bounding box selection...")
#     vsr.visualize_bbox_selection(x, y, width, height)
    
#     # Create comparison plot
#     print("Creating full comparison plot...")
#     vsr.create_comparison_plot(x, y, width, height, save_path)
    
#     # Create crops-only comparison if requested
#     if show_crops_only:
#         print("Creating crops-only comparison...")
#         vsr.create_crops_only_comparison(x, y, width, height, save_path)
    
#     return vsr

# # Example usage
# if __name__ == "__main__":
#     # Configuration
#     BASE_DIR = "./00002_0112"  # Change to your base directory containing model folders
#     BBOX_COORDS = (250, 175, 100, 100)  # (x, y, width, height) - adjust as needed
#     IMAGE_FILENAME = None  # Set to specific filename if all folders have the same image name, e.g., "frame_001.png"
    
#     # Ground Truth configuration
#     GT_DIR = "./ground_truth"  # Directory containing ground truth images
#     GT_FILENAME = None  # Set to specific GT filename if needed, e.g., "gt_frame_001.png"
    
#     # Custom model names (optional) - only override if you want different names
#     CUSTOM_MODEL_NAMES = {
#         'dcn_12': 'DCN-12',
#         'edvr_10': 'EDVR-10',
#         'frvsr_12_bicubic': 'FRVSR-12 (Bicubic)',
#         'frvsr_12_bicubic_icme': 'FRVSR-12 (ICME)',
#         'frvsr_12_gan_vanilla': 'FRVSR-12 (GAN)',
#         'frvsr_12_transp_conv': 'FRVSR-12 (TranspConv)',
#         'srnet_12_bicubic': 'SRNet-12 (Bicubic)'
#     }
    
#     # Output path (optional)
#     SAVE_PATH = "vsr_multi_model_comparison.png"
    
#     try:
#         # Run comparison
#         vsr_tool = run_multi_model_vsr_comparison(
#             base_dir=BASE_DIR,
#             bbox_coords=BBOX_COORDS,
#             image_filename=IMAGE_FILENAME,
#             gt_dir=GT_DIR,
#             gt_filename=GT_FILENAME,
#             custom_model_names=CUSTOM_MODEL_NAMES,
#             save_path=SAVE_PATH,
#             show_crops_only=True  # Set to False if you only want the full comparison
#         )
        
#         print("Multi-model VSR comparison completed successfully!")
        
#     except Exception as e:
#         print(f"Error: {e}")
#         print("\nMake sure to:")
#         print("1. Set the correct BASE_DIR path containing your model folders")
#         print("2. Set the correct GT_DIR path containing ground truth images (optional)")
#         print("3. Adjust BBOX_COORDS (x, y, width, height)")
#         print("4. Set IMAGE_FILENAME if all folders contain the same image file")
#         print("5. Set GT_FILENAME if you have a specific ground truth file")
#         print("6. Update CUSTOM_MODEL_NAMES if needed")


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import glob

class VSRMultiModelComparison:
    def __init__(self, base_dir, image_filename=None, gt_dir=None, gt_filename=None):
        """
        Initialize VSR comparison tool for multiple model directories
        
        Args:
            base_dir (str): Base directory containing model subdirectories
            image_filename (str): Specific image filename to load from each model directory
            gt_dir (str): Directory containing ground truth images (optional)
            gt_filename (str): Specific ground truth filename (optional)
        """
        self.base_dir = base_dir
        self.image_filename = image_filename
        self.gt_dir = gt_dir
        self.gt_filename = gt_filename
        self.images = {}
        self.model_order = []
        self.gt_image = None
        
        # Define the expected model folders and their display names
        self.model_mapping = {
            'GT': 'Ground Truth',
            'Lanczos': 'Lanczos (Baseline)',
            'dcn_12': 'DCN',
            'edvr_10': 'EDVR',
            'frvsr_12_bicubic': 'FRVSR Bicubic',
            'frvsr_12_bicubic_icme': 'FRVSR Bicubic ICME',
            'frvsr_12_gan_vanilla': 'FRVSR GAN',
            'frvsr_12_transp_conv': 'FRVSR TranspConv',
            'srnet_12_bicubic': 'SRNet Bicubic'
        }
        
        self.load_ground_truth()
        self.load_model_images()
    
    def load_ground_truth(self):
        """Load ground truth image if specified"""
        if self.gt_dir and os.path.exists(self.gt_dir):
            print(f"Loading ground truth from: {self.gt_dir}")
            
            # Get all image files from GT directory
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
            gt_files = []
            
            for ext in extensions:
                gt_files.extend(glob.glob(os.path.join(self.gt_dir, ext)))
                gt_files.extend(glob.glob(os.path.join(self.gt_dir, ext.upper())))
            
            if gt_files:
                # If specific GT filename is provided, look for it
                if self.gt_filename:
                    target_path = os.path.join(self.gt_dir, self.gt_filename)
                    if os.path.exists(target_path):
                        self.gt_image = np.array(Image.open(target_path).convert('RGB'))
                        print(f"  ✓ Loaded GT: {self.gt_filename} - {self.gt_image.shape}")
                    else:
                        # Look for the filename in the found images
                        for gt_path in gt_files:
                            if os.path.basename(gt_path) == self.gt_filename:
                                self.gt_image = np.array(Image.open(gt_path).convert('RGB'))
                                print(f"  ✓ Loaded GT: {self.gt_filename} - {self.gt_image.shape}")
                                break
                        else:
                            print(f"  Warning: {self.gt_filename} not found in GT directory, using first available")
                            gt_files.sort()
                            self.gt_image = np.array(Image.open(gt_files[0]).convert('RGB'))
                            print(f"  ✓ Loaded GT: {os.path.basename(gt_files[0])} - {self.gt_image.shape}")
                else:
                    # Use first available GT image
                    gt_files.sort()
                    self.gt_image = np.array(Image.open(gt_files[0]).convert('RGB'))
                    print(f"  ✓ Loaded GT: {os.path.basename(gt_files[0])} - {self.gt_image.shape}")
            else:
                print(f"  Warning: No GT images found in {self.gt_dir}")
        else:
            print("No ground truth directory specified or directory doesn't exist")
    
    def find_model_directories(self):
        """Find all model directories in base directory"""
        model_dirs = []
        
        if not os.path.exists(self.base_dir):
            raise ValueError(f"Base directory {self.base_dir} does not exist")
        
        # Get all subdirectories
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                model_dirs.append(item)
        
        # Sort with Lanczos first (baseline), then alphabetically
        def sort_key(model_name):
            if model_name == 'Lanczos':
                return '0_Lanczos'  # Ensure Lanczos comes first
            return model_name
        
        model_dirs.sort(key=sort_key)
        return model_dirs
    
    def load_image_from_directory(self, model_dir):
        """Load image from a specific model directory"""
        dir_path = os.path.join(self.base_dir, model_dir)
        
        # Get all image files
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(dir_path, ext)))
            image_files.extend(glob.glob(os.path.join(dir_path, ext.upper())))
        
        if not image_files:
            print(f"Warning: No images found in {dir_path}")
            return None
        
        # If specific filename is provided, look for it
        if self.image_filename:
            target_path = os.path.join(dir_path, self.image_filename)
            if os.path.exists(target_path):
                return Image.open(target_path).convert('RGB')
            else:
                # Look for the filename in the found images
                for img_path in image_files:
                    if os.path.basename(img_path) == self.image_filename:
                        return Image.open(img_path).convert('RGB')
                print(f"Warning: {self.image_filename} not found in {dir_path}, using first available image")
        
        # Sort files for consistent ordering and use the first one
        image_files.sort()
        return Image.open(image_files[0]).convert('RGB')
    
    def load_model_images(self):
        """Load images from all model directories"""
        model_dirs = self.find_model_directories()
        
        print(f"Found {len(model_dirs)} model directories:")
        
        for model_dir in model_dirs:
            print(f"  Loading from: {model_dir}")
            img = self.load_image_from_directory(model_dir)
            
            if img is not None:
                self.images[model_dir] = np.array(img)
                self.model_order.append(model_dir)
                print(f"    ✓ Loaded image: {self.images[model_dir].shape}")
            else:
                print(f"    ✗ Failed to load image from {model_dir}")
        
        if not self.images:
            raise ValueError("No images were successfully loaded")
        
        print(f"\nSuccessfully loaded {len(self.images)} images")
    
    def extract_bbox_region(self, image, x, y, width, height):
        """Extract bounding box region from image"""
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x = max(0, min(x, w - width))
        y = max(0, min(y, h - height))
        width = min(width, w - x)
        height = min(height, h - y)
        
        return image[y:y+height, x:x+width]
    
    def visualize_bbox_selection(self, x, y, width, height):
        """Show the GT or baseline image with bounding box overlay for verification"""
        # Use GT if available, otherwise use Lanczos (baseline), or first available image
        if self.gt_image is not None:
            reference_img = self.gt_image
            reference_name = self.model_mapping['GT']
            title_suffix = " (Ground Truth)"
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.imshow(reference_img)
        
        # Draw bounding box
        rect = patches.Rectangle((x, y), width, height, 
                               linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        ax.set_title(f'{reference_name} with Selected Region{title_suffix}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'Bounding Box: ({x}, {y}) - {width}×{height}')
        ax.axis('on')
        
        
        plt.show()
    
    def create_side_by_side_comparison(self, x, y, width, height, save_path=None, dataset_type='vimeo90k'):
        """
        Create side-by-side comparison with full GT on left and 4x2 grid of crops on right
        
        Args:
            x, y (int): Top-left corner coordinates of bounding box
            width, height (int): Dimensions of bounding box
            save_path (str): Path to save the figure (optional)
            dataset_type (str): 'vimeo90k' or 'icme' - controls which models to show
        """
        if self.gt_image is None:
            raise ValueError("Ground Truth image is required for this comparison layout")
        
        # Extract regions from all images
        regions = {}
        
        # Always use GT as the full reference image
        reference_img = self.gt_image
        reference_name = "Ground Truth"
        
        # Define model order and what to show based on dataset type
        if dataset_type.lower() == 'vimeo90k':
            # For Vimeo90K: Show GT crop + models (exclude ICME)
            regions['GT'] = self.extract_bbox_region(self.gt_image, x, y, width, height)
            
            # Add other models, excluding ICME
            for model_key in self.model_order:
                if 'icme' not in model_key.lower():
                    img = self.images[model_key]
                    regions[model_key] = self.extract_bbox_region(img, x, y, width, height)
        
        elif dataset_type.lower() == 'icme':
            # For ICME: Show ICME model + other models (exclude GT crop)
            for model_key in self.model_order:
                img = self.images[model_key]
                regions[model_key] = self.extract_bbox_region(img, x, y, width, height)
        
        else:
            raise ValueError("dataset_type must be 'vimeo90k' or 'icme'")
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 10))
        
        # Create grid layout: left side for full GT, right side for 4x2 crop grid
        from matplotlib import gridspec
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 2], wspace=0.05)
        
        # Left side: Full GT image with bounding box
        ax_ref = fig.add_subplot(gs[0, 0])
        ax_ref.imshow(reference_img)
        
        # Draw bounding box on reference
        rect = patches.Rectangle((x, y), width, height, 
                               linewidth=4, edgecolor='red', facecolor='none')
        ax_ref.add_patch(rect)
        
        ax_ref.set_title(reference_name, fontsize=16, fontweight='bold', pad=20)
        ax_ref.axis('off')
        
        # Right side: 4x2 grid of cropped regions
        gs_right = gridspec.GridSpecFromSubplotSpec(
                                                    2, 4, gs[0, 1], hspace=0.02, wspace=0.02
                                                )
        
        # Sort regions for consistent display
        sorted_regions = []
        
        if dataset_type.lower() == 'vimeo90k':
            # Order: GT, then models alphabetically (excluding ICME)
            if 'GT' in regions:
                sorted_regions.append(('GT', regions['GT']))
            
            # Add models in a specific order
            model_priority = ['Lanczos', 'dcn_12', 'edvr_10', 'frvsr_12_bicubic', 
                            'frvsr_12_gan_vanilla', 'frvsr_12_transp_conv', 'srnet_12_bicubic']
            
            for model_key in model_priority:
                if model_key in regions:
                    sorted_regions.append((model_key, regions[model_key]))
            
            # Add any remaining models not in priority list
            for model_key in regions:
                if model_key not in ['GT'] + model_priority:
                    sorted_regions.append((model_key, regions[model_key]))
        
        elif dataset_type.lower() == 'icme':
            # Order: ICME first, then other models
            icme_key = None
            for model_key in self.model_order:
                if 'icme' in model_key.lower():
                    icme_key = model_key
                    break
            
            if icme_key and icme_key in regions:
                sorted_regions.append((icme_key, regions[icme_key]))
            
            # Add other models
            model_priority = ['Lanczos', 'dcn_12', 'edvr_10', 'frvsr_12_bicubic', 
                            'frvsr_12_gan_vanilla', 'frvsr_12_transp_conv', 'srnet_12_bicubic']
            
            for model_key in model_priority:
                if model_key in regions and model_key != icme_key:
                    sorted_regions.append((model_key, regions[model_key]))
        
        # Display crops in 4x2 grid (limit to 8 total)
        for idx, (key, region) in enumerate(sorted_regions[:8]):
            row = idx // 4
            col = idx % 4
            
            ax_crop = fig.add_subplot(gs_right[row, col])
            ax_crop.imshow(region)
            
            # Determine display name and styling
            if key == 'GT':
                display_name = 'GT'
                title_color = 'darkgreen'
                bbox_props = dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.8)
                border_color = 'darkgreen'
                border_width = 2
            elif 'icme' in key.lower():
                display_name = self.model_mapping.get(key, key)
                title_color = 'darkblue'
                bbox_props = dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8)
                border_color = 'darkblue'
                border_width = 2
            else:
                display_name = self.model_mapping.get(key, key)
                title_color = 'black'
                bbox_props = None
                border_color = 'gray'
                border_width = 1
            
            ax_crop.set_title(display_name, fontsize=10, fontweight='bold', 
                            color=title_color, bbox=bbox_props)
            ax_crop.axis('off')
            
            # Add border around crop
            for spine in ax_crop.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(border_width)
        
        # Add main title
        dataset_label = "Vimeo90K" if dataset_type.lower() == 'vimeo90k' else "ICME"
        main_title = f"VSR Comparison ({dataset_label}) - Region: ({x}, {y}) - {width}×{height}"
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
        
        
        
        if save_path:
            dataset_suffix = f"_{dataset_type.lower()}"
            save_path_with_suffix = save_path.replace('.png', f'{dataset_suffix}.png')
            plt.savefig(save_path_with_suffix, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path_with_suffix}")
        
        plt.show()
    
    def create_comparison_plot(self, x, y, width, height, save_path=None):
        """
        Create comparison plot showing GT (if available) and all models with their cropped regions
        
        Args:
            x, y (int): Top-left corner coordinates of bounding box
            width, height (int): Dimensions of bounding box
            save_path (str): Path to save the figure (optional)
        """
        # Extract regions from all images
        regions = {}
        full_images = {}
        
        # Add GT if available
        if self.gt_image is not None:
            full_images['GT'] = self.gt_image
            regions['GT'] = self.extract_bbox_region(self.gt_image, x, y, width, height)
        
        # Add model images
        for model_key in self.model_order:
            img = self.images[model_key]
            full_images[model_key] = img
            regions[model_key] = self.extract_bbox_region(img, x, y, width, height)
        
        # Determine layout
        all_keys = (['GT'] if self.gt_image is not None else []) + self.model_order
        n_models = len(all_keys)
        
        # Create figure - each model gets full image + crop
        n_cols = 2
        n_rows = (n_models + 1) // 2
        fig = plt.figure(figsize=(20, 4 * n_rows))
        
        from matplotlib import gridspec
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
        
        for idx, key in enumerate(all_keys):
            row = idx // n_cols
            col = idx % n_cols
            
            # Create subplot with nested grid for full image + crop
            gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, gs[row, col], 
                                                     width_ratios=[2, 1], wspace=0.1)
            
            # Full image with bounding box
            ax_full = fig.add_subplot(gs_sub[0, 0])
            ax_full.imshow(full_images[key])
            rect = patches.Rectangle((x, y), width, height, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax_full.add_patch(rect)
            
            model_name = self.model_mapping.get(key, key)
            # Add special styling for GT
            if key == 'GT':
                ax_full.set_title(f'{model_name}', fontsize=12, fontweight='bold', 
                                color='darkgreen', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
            else:
                ax_full.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax_full.axis('off')
            
            # Cropped region
            ax_crop = fig.add_subplot(gs_sub[0, 1])
            ax_crop.imshow(regions[key])
            ax_crop.set_title('Detail', fontsize=10)
            ax_crop.axis('off')
            
            # Add border around crop for clarity
            border_color = 'darkgreen' if key == 'GT' else 'black'
            border_width = 2 if key == 'GT' else 1
            for spine in ax_crop.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(border_width)
        
        # Add main title
        title_prefix = "VSR Model Comparison with Ground Truth" if self.gt_image is not None else "VSR Model Comparison"
        fig.suptitle(f'{title_prefix} - Region: ({x}, {y}) - {width}×{height}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def create_crops_only_comparison(self, x, y, width, height, save_path=None):
        """
        Create a compact comparison showing only the cropped regions including GT
        
        Args:
            x, y (int): Top-left corner coordinates of bounding box
            width, height (int): Dimensions of bounding box
            save_path (str): Path to save the figure (optional)
        """
        # Extract regions from all images
        regions = {}
        
        # Add GT if available
        if self.gt_image is not None:
            regions['GT'] = self.extract_bbox_region(self.gt_image, x, y, width, height)
        
        # Add model regions
        for model_key in self.model_order:
            img = self.images[model_key]
            regions[model_key] = self.extract_bbox_region(img, x, y, width, height)
        
        # Determine layout
        all_keys = (['GT'] if self.gt_image is not None else []) + self.model_order
        n_models = len(all_keys)
        
        # Create figure for crops only
        fig = plt.figure(figsize=(16, 8))
        
        # Calculate grid layout
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        
        for idx, key in enumerate(all_keys):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            ax.imshow(regions[key])
            
            model_name = self.model_mapping.get(key, key)
            
            # Special styling for GT
            if key == 'GT':
                ax.set_title(model_name, fontsize=11, fontweight='bold', 
                           color='darkgreen', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
                border_color = 'darkgreen'
                border_width = 2
            else:
                ax.set_title(model_name, fontsize=10, fontweight='bold')
                border_color = 'black'
                border_width = 1
            
            ax.axis('off')
            
            # Add border for clarity
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(border_width)
        
        # Add main title
        title_prefix = "VSR Comparison with GT (Crops Only)" if self.gt_image is not None else "VSR Model Comparison (Crops Only)"
        fig.suptitle(f'{title_prefix} - Region: ({x}, {y}) - {width}×{height}', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        
        
        if save_path:
            crops_save_path = save_path.replace('.png', '_crops_only.png')
            plt.savefig(crops_save_path, dpi=300, bbox_inches='tight')
            print(f"Crops-only figure saved to: {crops_save_path}")
        
        plt.show()
    
    def update_model_names(self, model_dict):
        """Update model names for captions"""
        self.model_mapping.update(model_dict)

    def crop_gt_image_to_bbox(self, x, y, width, height, margin=20):
        """
        Crop the GT image to a region around the bounding box to make plots more compact.

        Args:
            x, y (int): Top-left corner of the bounding box
            width, height (int): Size of the bounding box
            margin (int): Extra pixels to include around the bounding box
        """
        if self.gt_image is None:
            return

        h, w = self.gt_image.shape[:2]
        
        # Compute cropping area with margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + width + margin)
        y2 = min(h, y + height + margin)

        self.gt_image = self.gt_image[y1:y2, x1:x2]
        print(f"GT image cropped to: ({x1}, {y1}) - ({x2}, {y2})")

# Convenience function
def run_multi_model_vsr_comparison(base_dir, bbox_coords, image_filename=None, 
                                 gt_dir=None, gt_filename=None, custom_model_names=None, 
                                 save_path=None, show_crops_only=True, use_side_by_side=True,
                                 dataset_type='vimeo90k'):
    """
    Convenience function to run multi-model VSR comparison with Ground Truth
    
    Args:
        base_dir (str): Base directory containing model subdirectories
        bbox_coords (tuple): (x, y, width, height) for bounding box
        image_filename (str): Specific image filename to load (optional)
        gt_dir (str): Directory containing ground truth images (optional)
        gt_filename (str): Specific ground truth filename (optional)
        custom_model_names (dict): Custom model names (optional)
        save_path (str): Path to save figure (optional)
        show_crops_only (bool): Whether to show crops-only comparison
        use_side_by_side (bool): Whether to use the new side-by-side layout
        dataset_type (str): 'vimeo90k' or 'icme' - controls which models to display
    """
    # Initialize comparison
    vsr = VSRMultiModelComparison(base_dir, image_filename, gt_dir, gt_filename)
    
    # Crop GT to make plot more compact, but preserve full bbox
    vsr.crop_gt_image_to_bbox(*bbox_coords, margin=50)
    
    # Update model names if provided
    if custom_model_names:
        vsr.update_model_names(custom_model_names)
    
    # Extract coordinates
    x, y, width, height = bbox_coords
    
    # Show bbox selection
    print("Showing bounding box selection...")
    vsr.visualize_bbox_selection(x, y, width, height)
    
    # Create the new side-by-side comparison if requested
    if use_side_by_side:
        print(f"Creating side-by-side comparison plot for {dataset_type}...")
        vsr.create_side_by_side_comparison(x, y, width, height, save_path, dataset_type)
    
    # Create comparison plot
    print("Creating full comparison plot...")
    vsr.create_comparison_plot(x, y, width, height, save_path)
    
    # Create crops-only comparison if requested
    if show_crops_only:
        print("Creating crops-only comparison...")
        vsr.create_crops_only_comparison(x, y, width, height, save_path)
    
    return vsr

# Example usage
if __name__ == "__main__":
    # Configuration
    BASE_DIR = "./00002_0112"  # Change to your base directory containing model folders
    BBOX_COORDS = (250, 160, 100, 100)  # (x, y, width, height) - adjust as needed
    IMAGE_FILENAME = None  # Set to specific filename if all folders have the same image name, e.g., "frame_001.png"
    
    # Ground Truth configuration
    GT_DIR = "./00002_0112/GT"  # Directory containing ground truth images
    GT_FILENAME = None  # Set to specific GT filename if needed, e.g., "gt_frame_001.png"
    
    # Custom model names (optional) - only override if you want different names
    CUSTOM_MODEL_NAMES = {
        'dcn_12': 'DCN-12',
        'edvr_10': 'EDVR-10',
        'frvsr_12_bicubic': 'FRVSR-12 (Bicubic)',
        'frvsr_12_bicubic_icme': 'FRVSR-12 (ICME)',
        'frvsr_12_gan_vanilla': 'FRVSR-12 (GAN)',
        'frvsr_12_transp_conv': 'FRVSR-12 (TranspConv)',
        'srnet_12_bicubic': 'SRNet-12 (Bicubic)'
    }
    
    # Output path (optional)
    SAVE_PATH = "vsr_multi_model_comparison.png"
    
    print("=== Running Vimeo90K Comparison ===")
    vsr_tool_vimeo = run_multi_model_vsr_comparison(
        base_dir=BASE_DIR,
        bbox_coords=BBOX_COORDS,
        image_filename=IMAGE_FILENAME,
        gt_dir=GT_DIR,
        gt_filename=GT_FILENAME,
        custom_model_names=CUSTOM_MODEL_NAMES,
        save_path=SAVE_PATH,
        show_crops_only=False,  
        use_side_by_side=True,
        dataset_type='vimeo90k'
        )
    



