import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# TABLE I - ENHANCED VISUALIZATIONS
# ============================================================================

# Model names (Lanczos first as baseline)
models = [
    "Lanczos",
    "SRNet", 
    "FNet + SRNet",
    "DCNAlignNet + SRNet",
    "EDVR"
]

# Metric values
metrics = {
    # Vimeo90K metrics
    "PSNR (Vimeo90K)": [30.14, 32.10, 32.94, 32.28, 33.68],
    "SSIM (Vimeo90K)": [0.8636, 0.898, 0.9028, 0.901, 0.929],
    "LPIPS (Vimeo90K)": [0.1638, 0.0618, 0.0469, 0.0589, 0.0410],
    "tOF (Vimeo90K)": [2.0013, 0.5948, 0.4194, 0.6021, 0.3916],
    
    # Vid4 metrics
    "PSNR (Vid4)": [22.04, 22.98, 24.51, 23.22, 25.35],
    "SSIM (Vid4)": [0.6313, 0.689, 0.7805, 0.696, 0.812],
    "LPIPS (Vid4)": [0.2431, 0.1116, 0.0772, 0.1057, 0.0663],
    "tOF (Vid4)": [2.4997, 1.2300, 0.6739, 1.1954, 0.4843],
    
    # Performance metrics
    "Params (M)": [0, 0.88, 2.66, 1.82, 3.55],
    "FPS": [292, 48.16, 30.74, 14.23, 4.24]
}

# Colors
baseline_color = '#FF6B6B'
model_colors = sns.color_palette("viridis", len(models)-1)
colors = [baseline_color] + list(model_colors)

def add_metric_arrows(ax, metric_name, position='top'):
    """Add directional arrows to indicate better performance"""
    if any(x in metric_name.upper() for x in ["PSNR", "SSIM", "FPS"]):
        arrow = "↑ Better"
        color = 'darkgreen'
    else:
        arrow = "↓ Better"  
        color = 'darkred'
    
    if position == 'top':
        ax.text(0.02, 0.98, arrow, transform=ax.transAxes, 
                fontsize=11, va='top', ha='left', color=color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    else:
        ax.text(0.98, 0.02, arrow, transform=ax.transAxes, 
                fontsize=11, va='bottom', ha='right', color=color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ============================================================================
# TABLE I - VERSION 1: All 4 Quality Metrics in One Chart (EDVR Style)
# ============================================================================

def create_combined_metrics_plot(dataset="Vimeo90K"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Select metrics for the dataset
    if dataset == "Vimeo90K":
        metric_names = ["PSNR (Vimeo90K)", "SSIM (Vimeo90K)", "LPIPS (Vimeo90K)", "tOF (Vimeo90K)"]
        title = f"Quality Metrics Comparison - {dataset} Dataset"
    else:
        metric_names = ["PSNR (Vid4)", "SSIM (Vid4)", "LPIPS (Vid4)", "tOF (Vid4)"]
        title = f"Quality Metrics Comparison - {dataset} Dataset"
    
    x = np.arange(len(models))
    width = 0.2
    
    # Create bars for each metric
    for i, metric in enumerate(metric_names):
        values = metrics[metric]
        
        # Normalize values for visualization (keep original for labels)
        if "PSNR" in metric:
            normalized_values = [(v-20)/15 for v in values]  # Scale PSNR to 0-1 range
            color = '#2E8B57'
        elif "SSIM" in metric:
            normalized_values = values  # Already 0-1
            color = '#4169E1'
        elif "LPIPS" in metric:
            normalized_values = [1-v for v in values]  # Invert for better=higher
            color = '#DC143C'
        else:  # tOF
            normalized_values = [1/(1+v) for v in values]  # Transform to 0-1, higher=better
            color = '#FF8C00'
        
        bars = ax.bar(x + i*width, normalized_values, width, 
                     label=metric.split('(')[0].strip(), color=color, alpha=0.8)
        
        # Add value labels (original values)
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if val < 1 and val > 0:
                label = f'{val:.3f}'
            else:
                label = f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   label, ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Performance', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.show()

# Create combined plots
create_combined_metrics_plot("Vimeo90K")
create_combined_metrics_plot("Vid4")

# ============================================================================
# TABLE I - VERSION 2: Efficiency vs Performance Scatter Plot
# ============================================================================

models = [
    "SRNet", 
    "FNet + SRNet",
    "DCNAlignNet + SRNet",
    "EDVR"
]

# Metric values
metrics = {
    # Vimeo90K metrics
    "PSNR (Vimeo90K)": [32.10, 32.94, 32.28, 33.68],
    "SSIM (Vimeo90K)": [0.898, 0.9028, 0.901, 0.929],
    "LPIPS (Vimeo90K)": [0.0618, 0.0469, 0.0589, 0.0410],
    "tOF (Vimeo90K)": [0.5948, 0.4194, 0.6021, 0.3916],
    
    # Vid4 metrics
    "PSNR (Vid4)": [22.98, 24.51, 23.22, 25.35],
    "SSIM (Vid4)": [0.689, 0.7805, 0.696, 0.812],
    "LPIPS (Vid4)": [0.1116, 0.0772, 0.1057, 0.0663],
    "tOF (Vid4)": [1.2300, 0.6739, 1.1954, 0.4843],
    
    # Performance metrics
    "Params (M)": [0.88, 2.66, 1.82, 3.55],
    "FPS": [48.16, 30.74, 14.23, 4.24]
}

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Use PSNR as performance metric and FPS as efficiency
perf_metric = metrics["PSNR (Vimeo90K)"]
efficiency = metrics["FPS"]
params = metrics["Params (M)"]

# Create scatter plot with bubble size based on parameters
for i, (model, psnr, fps, param) in enumerate(zip(models, perf_metric, efficiency, params)):
    size = max(100, param * 100) if param > 0 else 200  # Lanczos gets special size
    
    if model == "Lanczos":
        ax.scatter(fps, psnr, s=size, c=baseline_color, alpha=0.7, 
                  edgecolors='black', linewidth=2, marker='s', label='Baseline')
    else:
        ax.scatter(fps, psnr, s=size, c=colors[i], alpha=0.7, 
                  edgecolors='black', linewidth=1, label=model)
    
    # Add model labels
    ax.annotate(model, (fps, psnr), xytext=(5, 5), textcoords='offset points',
               fontsize=9, fontweight='bold')

ax.set_xlabel('FPS (Frames Per Second) ↑', fontsize=12, fontweight='bold')
ax.set_ylabel('PSNR (dB) ↑', fontsize=12, fontweight='bold')
ax.set_title('Efficiency vs Performance Trade-off\n(Bubble size = Model Parameters)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add quadrant lines
ax.axhline(y=np.mean(perf_metric[1:]), color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=np.mean(efficiency[1:]), color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Model names (Lanczos first as baseline)
models = [
    "Lanczos",
    "SRNet", 
    "FNet + SRNet",
    "DCNAlignNet + SRNet",
    "EDVR"
]

# Metric values
metrics = {
    # Vimeo90K metrics
    "PSNR (Vimeo90K)": [30.14, 32.10, 32.94, 32.28, 33.68],
    "SSIM (Vimeo90K)": [0.8636, 0.898, 0.9028, 0.901, 0.929],
    "LPIPS (Vimeo90K)": [0.1638, 0.0618, 0.0469, 0.0589, 0.0410],
    "tOF (Vimeo90K)": [2.0013, 0.5948, 0.4194, 0.6021, 0.3916],
    
    # Vid4 metrics
    "PSNR (Vid4)": [22.04, 22.98, 24.51, 23.22, 25.35],
    "SSIM (Vid4)": [0.6313, 0.689, 0.7805, 0.696, 0.812],
    "LPIPS (Vid4)": [0.2431, 0.1116, 0.0772, 0.1057, 0.0663],
    "tOF (Vid4)": [2.4997, 1.2300, 0.6739, 1.1954, 0.4843],
    
    # Performance metrics
    "Params (M)": [0, 0.88, 2.66, 1.82, 3.55],
    "FPS": [292, 48.16, 30.74, 14.23, 4.24]
}

# ============================================================================
# TABLE II - ABLATION STUDY VISUALIZATIONS
# ============================================================================

# Model names for ablation study
models = [
    "FRVSR+ w/o Skip",
    "FNet + SRResNet", 
    "FRVSR+",
    "FRVSR+ w/ TC",
    "FRVSR+ Small",
    "FRVSR+ Large"
]

# Metric values from ablation study
metrics = {
    # Vimeo90K metrics
    "PSNR (Vimeo90K)": [32.64, 32.74, 32.94, 33.03, 32.72, 33.24],
    "SSIM (Vimeo90K)": [0.9158, 0.918, 0.9028, 0.921, 0.917, 0.924],
    "LPIPS (Vimeo90K)": [0.0469, 0.0534, 0.0469, 0.0455, 0.0536, 0.0426],
    "tOF (Vimeo90K)": [0.4346, 0.4509, 0.4194, 0.4299, 0.4627, 0.3940],
    
    # Vid4 metrics
    "PSNR (Vid4)": [24.38, 24.35, 24.51, 24.59, 24.22, 24.83],
    "SSIM (Vid4)": [0.769, 0.769, 0.7805, 0.782, 0.761, 0.7937],
    "LPIPS (Vid4)": [0.0781, 0.086, 0.0772, 0.0747, 0.0898, 0.0691],
    "tOF (Vid4)": [0.6782, 0.7306, 0.6739, 0.6692, 0.7700, 0.5942],
    
    # Performance metrics
    "Params (M)": [2.66, 2.66, 2.66, 2.73, 2.36, 2.95],
    "FPS": [30.74, 30.74, 30.74, 21.88, 38.26, 26.02]
}

# Colors for ablation study variants
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

def add_metric_arrows(ax, metric_name, position='top'):
    """Add directional arrows to indicate better performance"""
    if any(x in metric_name.upper() for x in ["PSNR", "SSIM", "FPS"]):
        arrow = "↑ Better"
        color = 'darkgreen'
    else:
        arrow = "↓ Better"  
        color = 'darkred'
    
    if position == 'top':
        ax.text(0.02, 0.98, arrow, transform=ax.transAxes, 
                fontsize=11, va='top', ha='left', color=color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    else:
        ax.text(0.98, 0.02, arrow, transform=ax.transAxes, 
                fontsize=11, va='bottom', ha='right', color=color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ============================================================================
# TABLE II - VERSION 1: All 4 Quality Metrics in One Chart (Ablation Study)
# ============================================================================

def create_ablation_metrics_plot(dataset="Vimeo90K"):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Select metrics for the dataset
    if dataset == "Vimeo90K":
        metric_names = ["PSNR (Vimeo90K)", "SSIM (Vimeo90K)", "LPIPS (Vimeo90K)", "tOF (Vimeo90K)"]
        title = f"Ablation Study - Quality Metrics Comparison - {dataset} Dataset"
    else:
        metric_names = ["PSNR (Vid4)", "SSIM (Vid4)", "LPIPS (Vid4)", "tOF (Vid4)"]
        title = f"Ablation Study - Quality Metrics Comparison - {dataset} Dataset"
    
    x = np.arange(len(models))
    width = 0.18
    
    # Create bars for each metric
    for i, metric in enumerate(metric_names):
        values = metrics[metric]
        
        # Normalize values for visualization (keep original for labels)
        if "PSNR" in metric:
            normalized_values = [(v-24)/10 for v in values]  # Scale PSNR to 0-1 range
            color = '#2E8B57'
        elif "SSIM" in metric:
            normalized_values = [(v-0.75)/0.2 for v in values]  # Scale SSIM for better visibility
            color = '#4169E1'
        elif "LPIPS" in metric:
            normalized_values = [1-v*10 for v in values]  # Invert and scale for better visibility
            color = '#DC143C'
        else:  # tOF
            normalized_values = [1/(1+v*2) for v in values]  # Transform to 0-1, higher=better
            color = '#FF8C00'
        
        bars = ax.bar(x + i*width, normalized_values, width, 
                     label=metric.split('(')[0].strip(), color=color, alpha=0.8)
        
        # Add value labels (original values)
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if val < 1 and val > 0:
                label = f'{val:.3f}'
            else:
                label = f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   label, ha='center', va='bottom', fontsize=7, fontweight='bold', rotation=90)
    
    ax.set_xlabel('Ablation Study Variants', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Performance', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.3)
    
    plt.tight_layout()
    plt.show()

# Create combined plots for ablation study
create_ablation_metrics_plot("Vimeo90K")
create_ablation_metrics_plot("Vid4")

# ============================================================================
# TABLE II - VERSION 2: Efficiency vs Performance Scatter Plot (Ablation)
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Use PSNR as performance metric and FPS as efficiency for ablation study
perf_metric = metrics["PSNR (Vimeo90K)"]
efficiency = metrics["FPS"]
params = metrics["Params (M)"]

# Create scatter plot with bubble size based on parameters
for i, (model, psnr, fps, param) in enumerate(zip(models, perf_metric, efficiency, params)):
    size = max(150, param * 80)  # Scale bubble size based on parameters
    
    # Highlight the main FRVSR+ variant
    if model == "FRVSR+":
        ax.scatter(fps, psnr, s=size, c=colors[i], alpha=0.8, 
                  edgecolors='black', linewidth=3, marker='*', label=model, zorder=5)
    else:
        ax.scatter(fps, psnr, s=size, c=colors[i], alpha=0.7, 
                  edgecolors='black', linewidth=1.5, label=model)
    
    # Add model labels with offset to avoid overlap
    offset_x = 1 if fps > 30 else -3
    offset_y = 0.02 if i % 2 == 0 else -0.02
    ax.annotate(model, (fps, psnr), xytext=(offset_x, offset_y), textcoords='offset points',
               fontsize=8, fontweight='bold', ha='left' if fps > 30 else 'right')

ax.set_xlabel('FPS (Frames Per Second) ↑', fontsize=12, fontweight='bold')
ax.set_ylabel('PSNR (dB) ↑', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study: Efficiency vs Performance Trade-off\n(Bubble size = Model Parameters)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add quadrant lines based on main FRVSR+ model
frvsr_idx = models.index("FRVSR+")
ax.axhline(y=perf_metric[frvsr_idx], color='gray', linestyle='--', alpha=0.5, label='FRVSR+ baseline')
ax.axvline(x=efficiency[frvsr_idx], color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ============================================================================
# TABLE II - VERSION 3: Component Analysis Bar Chart
# ============================================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# PSNR comparison
ax1.bar(range(len(models)), metrics["PSNR (Vimeo90K)"], color=colors, alpha=0.8)
ax1.set_title('PSNR Comparison (Vimeo90K)', fontsize=14, fontweight='bold')
ax1.set_ylabel('PSNR (dB) ↑', fontweight='bold')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(metrics["PSNR (Vimeo90K)"]):
    ax1.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

# SSIM comparison
ax2.bar(range(len(models)), metrics["SSIM (Vimeo90K)"], color=colors, alpha=0.8)
ax2.set_title('SSIM Comparison (Vimeo90K)', fontsize=14, fontweight='bold')
ax2.set_ylabel('SSIM ↑', fontweight='bold')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(metrics["SSIM (Vimeo90K)"]):
    ax2.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# FPS comparison
ax3.bar(range(len(models)), metrics["FPS"], color=colors, alpha=0.8)
ax3.set_title('Speed Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('FPS ↑', fontweight='bold')
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, rotation=45, ha='right')
ax3.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(metrics["FPS"]):
    ax3.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

# Parameters comparison
ax4.bar(range(len(models)), metrics["Params (M)"], color=colors, alpha=0.8)
ax4.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax4.set_ylabel('Parameters (M)', fontweight='bold')
ax4.set_xticks(range(len(models)))
ax4.set_xticklabels(models, rotation=45, ha='right')
ax4.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(metrics["Params (M)"]):
    ax4.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Ablation Study: Component-wise Performance Analysis', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()


# ============================================================================
# TABLES III & IV - QP Analysis (Complete Dataset)
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np

# Updated models list with all models from the tables
models_qp = ["FNet + SRNet", "Finetuned", "Finetuned Adv.", "EDVR", "EDVR Finetuned", "BasicVSR"]

# Complete QP data from both tables
qp_low_metrics = {
    # QP = 17
    "PSNR (QP=17)": [27.45, 28.90, 27.06, 27.06, 28.9, 29.46],
    "SSIM (QP=17)": [0.744, 0.801, 0.752, 0.751, 0.799, 0.818],
    "LPIPS (QP=17)": [0.138, 0.084, 0.079, 0.079, 0.084, 0.097],
    "tOF (QP=17)": [2.810, 1.892, 1.578, 1.578, 2.006, 1.413],
    
    # QP = 22
    "PSNR (QP=22)": [27.22, 28.43, 26.79, 26.79, 28.45, 28.69],
    "SSIM (QP=22)": [0.724, 0.773, 0.729, 0.729, 0.772, 0.778],
    "LPIPS (QP=22)": [0.145, 0.092, 0.087, 0.087, 0.093, 0.111],
    "tOF (QP=22)": [3.395, 2.716, 2.432, 2.432, 2.782, 2.368],
    
    # QP = 27
    "PSNR (QP=27)": [26.61, 27.36, 26.01, 26.01, 27.44, 27.32],
    "SSIM (QP=27)": [0.682, 0.716, 0.674, 0.674, 0.718, 0.710],
    "LPIPS (QP=27)": [0.160, 0.109, 0.103, 0.103, 0.110, 0.137],
    "tOF (QP=27)": [4.435, 3.965, 3.435, 3.435, 4.065, 3.698]
}

qp_high_metrics = {
    # QP = 32
    "PSNR (QP=32)": [25.46, 25.72, 24.51, 24.51, 25.80, 25.60],
    "SSIM (QP=32)": [0.618, 0.637, 0.584, 0.584, 0.640, 0.625],
    "LPIPS (QP=32)": [0.183, 0.134, 0.136, 0.136, 0.135, 0.167],
    "tOF (QP=32)": [5.888, 5.574, 4.286, 4.286, 5.755, 5.313],
    
    # QP = 34
    "PSNR (QP=34)": [24.91, 25.07, 23.92, 23.92, 25.11, 24.94],
    "SSIM (QP=34)": [0.591, 0.606, 0.550, 0.550, 0.607, 0.593],
    "LPIPS (QP=34)": [0.194, 0.145, 0.153, 0.153, 0.147, 0.179],
    "tOF (QP=34)": [6.532, 6.183, 4.599, 4.599, 6.442, 5.962],
    
    # QP = 37
    "PSNR (QP=37)": [24.09, 24.17, 23.21, 23.21, 24.21, 24.06],
    "SSIM (QP=37)": [0.555, 0.566, 0.509, 0.509, 0.568, 0.553],
    "LPIPS (QP=37)": [0.208, 0.122, 0.179, 0.179, 0.164, 0.198],
    "tOF (QP=37)": [7.488, 7.062, 5.091, 5.092, 7.422, 6.880]
}

# Enhanced color palette for all 6 models
colors_qp = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC', '#228B22']
linestyles = ['-', '-', '--', ':', '-.', '-']
markers = ['o', 's', '^', 'D', 'v', 'h']

def add_metric_arrows(ax, metric, position='top'):
    """Add directional indicators for metrics"""
    if metric in ['PSNR', 'SSIM']:
        direction = '↑ Better'
        color = 'green'
    else:  # LPIPS, tOF
        direction = '↓ Better'
        color = 'red'
    
    ax.text(0.02, 0.98 if position == 'top' else 0.02, direction, 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top' if position == 'top' else 'bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))

# ============================================================================
# Enhanced QP Trend Analysis with All Models
# ============================================================================

def create_qp_trends_complete():
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    qp_values_all = [17, 22, 27, 32, 34, 37]
    metrics_for_trend = ["PSNR", "SSIM", "LPIPS", "tOF"]
    
    for idx, metric in enumerate(metrics_for_trend):
        ax = axes[idx // 2, idx % 2]
        
        for model_idx in range(len(models_qp)):
            values = []
            for qp in qp_values_all:
                if qp <= 27:
                    values.append(qp_low_metrics[f"{metric} (QP={qp})"][model_idx])
                else:
                    values.append(qp_high_metrics[f"{metric} (QP={qp})"][model_idx])
            
            # Plot with enhanced styling
            ax.plot(qp_values_all, values, 
                   marker=markers[model_idx], 
                   linewidth=2.5, 
                   markersize=8,
                   linestyle=linestyles[model_idx],
                   label=models_qp[model_idx], 
                   color=colors_qp[model_idx],
                   alpha=0.8)
            
            # Add subtle confidence bands for visual appeal
            values_array = np.array(values)
            noise_level = np.std(values_array) * 0.05
            ax.fill_between(qp_values_all, 
                           values_array - noise_level, 
                           values_array + noise_level,
                           alpha=0.1, color=colors_qp[model_idx])
        
        # Styling improvements
        ax.set_xlabel('QP Value (Higher = More Compression)', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{metric} Score', fontsize=13, fontweight='bold')
        ax.set_title(f'{metric} Performance vs Compression Level', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        # Add compression level background shading
        ax.axvspan(17, 27, alpha=0.05, color='green', label='Low Compression' if idx == 0 else "")
        ax.axvspan(32, 37, alpha=0.05, color='red', label='High Compression' if idx == 0 else "")
        
        # Set x-axis ticks to QP values
        ax.set_xticks(qp_values_all)
        ax.set_xticklabels(qp_values_all)
        
        add_metric_arrows(ax, metric, 'top')
    
    plt.suptitle("Complete Model Performance Analysis vs JPEG Compression Level", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

# ============================================================================
# Model Ranking Analysis
# ============================================================================

def create_model_ranking_analysis():
    """Create ranking analysis across different QP levels"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
    qp_values = [17, 22, 27, 32, 34, 37]
    metrics = ["PSNR", "SSIM", "LPIPS", "tOF"]
    
    for qp_idx, qp in enumerate(qp_values):
        ax = axes[qp_idx // 3, qp_idx % 3]
        
        # Get data for this QP level
        metric_data = {}
        for metric in metrics:
            if qp <= 27:
                metric_data[metric] = qp_low_metrics[f"{metric} (QP={qp})"]
            else:
                metric_data[metric] = qp_high_metrics[f"{metric} (QP={qp})"]
        
        # Create grouped bar chart
        x = np.arange(len(models_qp))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = metric_data[metric]
            # Normalize values for better comparison (0-1 scale)
            if metric in ['PSNR', 'SSIM']:
                norm_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
            else:  # Lower is better for LPIPS, tOF
                norm_values = [1 - (v - min(values)) / (max(values) - min(values)) for v in values]
            
            ax.bar(x + i * width, norm_values, width, 
                  label=metric, alpha=0.8, color=colors_qp[i])
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Normalized Performance (0-1)', fontweight='bold')
        ax.set_title(f'QP = {qp} Performance Comparison', fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([m.replace(' ', '\n') for m in models_qp], rotation=0, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Model Performance Rankings Across Different Compression Levels", 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# Statistical Summary Analysis
# ============================================================================

def create_statistical_summary():
    """Create statistical summary of model performance"""
    
    # Calculate average performance across all QP levels
    model_stats = {}
    
    for model_idx, model in enumerate(models_qp):
        model_stats[model] = {
            'PSNR_avg': 0, 'SSIM_avg': 0, 'LPIPS_avg': 0, 'tOF_avg': 0,
            'PSNR_std': 0, 'SSIM_std': 0, 'LPIPS_std': 0, 'tOF_std': 0
        }
        
        # Collect all values for this model
        psnr_vals, ssim_vals, lpips_vals, tof_vals = [], [], [], []
        
        for qp in [17, 22, 27, 32, 34, 37]:
            if qp <= 27:
                psnr_vals.append(qp_low_metrics[f"PSNR (QP={qp})"][model_idx])
                ssim_vals.append(qp_low_metrics[f"SSIM (QP={qp})"][model_idx])
                lpips_vals.append(qp_low_metrics[f"LPIPS (QP={qp})"][model_idx])
                tof_vals.append(qp_low_metrics[f"tOF (QP={qp})"][model_idx])
            else:
                psnr_vals.append(qp_high_metrics[f"PSNR (QP={qp})"][model_idx])
                ssim_vals.append(qp_high_metrics[f"SSIM (QP={qp})"][model_idx])
                lpips_vals.append(qp_high_metrics[f"LPIPS (QP={qp})"][model_idx])
                tof_vals.append(qp_high_metrics[f"tOF (QP={qp})"][model_idx])
        
        # Calculate statistics
        model_stats[model]['PSNR_avg'] = np.mean(psnr_vals)
        model_stats[model]['SSIM_avg'] = np.mean(ssim_vals)
        model_stats[model]['LPIPS_avg'] = np.mean(lpips_vals)
        model_stats[model]['tOF_avg'] = np.mean(tof_vals)
        
        model_stats[model]['PSNR_std'] = np.std(psnr_vals)
        model_stats[model]['SSIM_std'] = np.std(ssim_vals)
        model_stats[model]['LPIPS_std'] = np.std(lpips_vals)
        model_stats[model]['tOF_std'] = np.std(tof_vals)
    
    # Print summary table
    print("="*80)
    print("STATISTICAL SUMMARY - Average Performance Across All QP Levels")
    print("="*80)
    print(f"{'Model':<15} {'PSNR ↑':<12} {'SSIM ↑':<12} {'LPIPS ↓':<12} {'tOF ↓':<12}")
    print("-"*80)
    
    for model in models_qp:
        stats = model_stats[model]
        print(f"{model:<15} {stats['PSNR_avg']:<12.2f} {stats['SSIM_avg']:<12.3f} "
              f"{stats['LPIPS_avg']:<12.3f} {stats['tOF_avg']:<12.2f}")
    
    return model_stats

# ============================================================================
# Execute Analysis
# ============================================================================

if __name__ == "__main__":
    print("Creating comprehensive QP analysis...")
    
    # Run trend analysis
    create_qp_trends_complete()
    
    # Run ranking analysis
    create_model_ranking_analysis()
    
    # Generate statistical summary
    stats = create_statistical_summary()