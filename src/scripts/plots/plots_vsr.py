# # Table I

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Model names (Lanczos first as baseline)
# models = [
#     "Lanczos",
#     "SRNet", 
#     "FNet + SRNet",
#     "DCNAlignNet + SRNet",
#     "EDVR"
# ]

# # Metric values
# metrics = {
#     # Vimeo90K metrics
#     "PSNR (Vimeo90K)": [30.14, 32.10, 32.94, 32.28, 33.68],
#     "SSIM (Vimeo90K)": [0.8636, 0.898, 0.9028, 0.901, 0.929],
#     "LPIPS (Vimeo90K)": [0.1638, 0.0618, 0.0469, 0.0589, 0.0410],
#     "tOF (Vimeo90K)": [2.0013, 0.5948, 0.4194, 0.6021, 0.3916],
    
#     # Vid4 metrics
#     "PSNR (Vid4)": [22.04, 22.98, 24.51, 23.22, 25.35],
#     "SSIM (Vid4)": [0.6313, 0.689, 0.7805, 0.696, 0.812],
#     "LPIPS (Vid4)": [0.2431, 0.1116, 0.0772, 0.1057, 0.0663],
#     "tOF (Vid4)": [2.4997, 1.2300, 0.6739, 1.1954, 0.4843],
    
#     # Performance metrics
#     "Params (M)": [0, 0.88, 2.66, 1.82, 3.55],  # Lanczos has no parameters
#     "FPS": [292, 48.16, 30.74, 14.23, 4.24]
# }

# # Create color palette with distinctive color for Lanczos (baseline)
# baseline_color = '#FF6B6B'  # Red for Lanczos baseline
# model_colors = sns.color_palette("muted", len(models)-1)
# colors = [baseline_color] + list(model_colors)

# # Helper function to add value labels
# def add_labels(ax, bars, values, metric_name=""):
#     for bar, val in zip(bars, values):
#         height = bar.get_height()
#         # Format based on metric type
#         if "Params" in metric_name and val == 0:
#             label = "N/A"
#         elif val < 1 and val > 0:
#             label = f'{val:.3f}'
#         elif val >= 100:
#             label = f'{val:.0f}'
#         else:
#             label = f'{val:.2f}'
            
#         ax.text(bar.get_x() + bar.get_width() / 2, height,
#                 label, ha='center', va='bottom', fontsize=9, fontweight='bold')

# # Create grouped plots for Vimeo90K
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# vimeo_metrics = ["PSNR (Vimeo90K)", "SSIM (Vimeo90K)", "LPIPS (Vimeo90K)", "tOF (Vimeo90K)"]

# for ax, metric in zip(axes.flat, vimeo_metrics):
#     values = metrics[metric]
#     bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.8)
    
#     # Special formatting for baseline bar
#     bars[0].set_alpha(0.8)
#     bars[0].set_hatch('//')
    
#     ax.set_title(metric, fontsize=14, fontweight='bold', pad=20)
#     ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
#     add_labels(ax, bars, values, metric)
#     ax.grid(True, axis='y', alpha=0.3)
    
#     # Add arrows for better/worse direction
#     if "PSNR" in metric or "SSIM" in metric:
#         ax.text(0.02, 0.98, "↑ Better", transform=ax.transAxes, 
#                 fontsize=10, va='top', ha='left', color='green', fontweight='bold')
#     else:
#         ax.text(0.02, 0.98, "↓ Better", transform=ax.transAxes, 
#                 fontsize=10, va='top', ha='left', color='green', fontweight='bold')

# plt.tight_layout()
# plt.suptitle("Vimeo90K Dataset Comparison (Lanczos Baseline)", fontsize=18, y=0.98, fontweight='bold')
# plt.show()

# # Create grouped plots for Vid4
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# vid4_metrics = ["PSNR (Vid4)", "SSIM (Vid4)", "LPIPS (Vid4)", "tOF (Vid4)"]

# for ax, metric in zip(axes.flat, vid4_metrics):
#     values = metrics[metric]
#     bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.8)
    
#     # Special formatting for baseline bar
#     bars[0].set_alpha(0.8)
#     bars[0].set_hatch('//')
    
#     ax.set_title(metric, fontsize=14, fontweight='bold', pad=20)
#     ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
#     add_labels(ax, bars, values, metric)
#     ax.grid(True, axis='y', alpha=0.3)
    
#     # Add arrows for better/worse direction
#     if "PSNR" in metric or "SSIM" in metric:
#         ax.text(0.02, 0.98, "↑ Better", transform=ax.transAxes, 
#                 fontsize=10, va='top', ha='left', color='green', fontweight='bold')
#     else:
#         ax.text(0.02, 0.98, "↓ Better", transform=ax.transAxes, 
#                 fontsize=10, va='top', ha='left', color='green', fontweight='bold')

# plt.tight_layout()
# plt.suptitle("Vid4 Dataset Comparison (Lanczos Baseline)", fontsize=18, y=0.98, fontweight='bold')
# plt.show()

# # Plot for Performance metrics (Params and FPS)
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# performance_metrics = ["Params (M)", "FPS"]

# for ax, metric in zip(axes, performance_metrics):
#     values = metrics[metric]
#     bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.8)
    
#     # Special formatting for baseline bar
#     bars[0].set_alpha(0.8)
#     bars[0].set_hatch('//')
    
#     ax.set_title(metric, fontsize=14, fontweight='bold', pad=20)
#     ax.set_xticklabels(models, rotation=25, ha='right', fontsize=10)
#     add_labels(ax, bars, values, metric)
#     ax.grid(True, axis='y', alpha=0.3)
    
#     # Add direction indicator
#     if "FPS" in metric:
#         ax.text(0.02, 0.98, "↑ Better", transform=ax.transAxes, 
#                 fontsize=10, va='top', ha='left', color='green', fontweight='bold')

# plt.tight_layout()
# plt.suptitle("Model Complexity and Speed Comparison (Lanczos Baseline)", fontsize=18, y=1.02, fontweight='bold')

# # Add legend
# legend_elements = [
#     plt.Rectangle((0,0),1,1, facecolor=baseline_color, alpha=0.8, hatch='//', 
#                   edgecolor='black', linewidth=0.8, label='Lanczos (Baseline)'),
#     plt.Rectangle((0,0),1,1, facecolor=model_colors[0], 
#                   edgecolor='black', linewidth=0.8, label='Learning-based Models')
# ]
# plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
#               ncol=2, fontsize=12, frameon=True, fancybox=True, shadow=True)

# plt.show()

# # Table II

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Model names
# models = [
#     "FNet + SRNet w/o Skip",
#     "FNet + SRResNet",
#     "FNet + SRNet",
#     "FNet + SRNet w/ TC",
#     "FNet + SRNet Small",
#     "FNet + SRNet Large"
# ]

# # Metric values
# metrics = {
#     "PSNR-Y (Vimeo90K)": [32.64, 32.74, 32.94, 33.03, 32.72, 33.24],
#     "SSIM (Vimeo90K)":   [0.9158, 0.918, 0.9028, 0.921, 0.917, 0.924],
#     "LPIPS (Vimeo90K)":  [0.0469, 0.0534, 0.0469, 0.0455, 0.0536, 0.0426],
#     "tOF (Vimeo90K)":    [0.4346, 0.4509, 0.4194, 0.4299, 0.4627, 0.3940],
#     "PSNR-Y (Vid4)":     [24.38, 24.35, 24.51, 24.59, 24.22, 24.83],
#     "SSIM (Vid4)":       [0.769, 0.769, 0.7805, 0.782, 0.761, 0.7937],
#     "LPIPS (Vid4)":      [0.0781, 0.086, 0.0772, 0.0747, 0.0898, 0.0691],
#     "tOF (Vid4)":        [0.6782, 0.7306, 0.6739, 0.6692, 0.7700, 0.5942],
#     "Params (M)":        [2.66, 2.66, 2.66, 2.73, 2.36, 2.95],
#     "FPS":               [30.74, 30.74, 30.74, 21.88, 38.26, 26.02]
# }

# # Seaborn color palette
# colors = sns.color_palette("muted", len(models))

# # Helper function to add value labels
# def add_labels(ax, bars, values):
#     for bar, val in zip(bars, values):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2, height,
#                 f'{val:.3f}' if val < 1 else f'{val:.2f}',
#                 ha='center', va='bottom', fontsize=8)

# # Create grouped plots for Vimeo90K
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# vimeo_metrics = ["PSNR-Y (Vimeo90K)", "SSIM (Vimeo90K)", "LPIPS (Vimeo90K)", "tOF (Vimeo90K)"]

# for ax, metric in zip(axes.flat, vimeo_metrics):
#     values = metrics[metric]
#     bars = ax.bar(models, values, color=colors)
#     ax.set_title(metric, fontsize=12, fontweight='bold')
#     ax.set_xticklabels(models, rotation=15, ha='right')
#     add_labels(ax, bars, values)
#     ax.grid(True, axis='y', alpha=0.3)

# plt.tight_layout()
# plt.suptitle("Vimeo90K Metrics", fontsize=16, y=1.02)
# plt.show()

# # Create grouped plots for Vid4
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# vid4_metrics = ["PSNR-Y (Vid4)", "SSIM (Vid4)", "LPIPS (Vid4)", "tOF (Vid4)"]

# for ax, metric in zip(axes.flat, vid4_metrics):
#     values = metrics[metric]
#     bars = ax.bar(models, values, color=colors)
#     ax.set_title(metric, fontsize=12, fontweight='bold')
#     ax.set_xticklabels(models, rotation=15, ha='right')
#     add_labels(ax, bars, values)
#     ax.grid(True, axis='y', alpha=0.3)

# plt.tight_layout()
# plt.suptitle("Vid4 Metrics", fontsize=16, y=1.02)
# plt.show()

# # Plot for Params and Time
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# performance_metrics = ["Params (M)", "FPS"]

# for ax, metric in zip(axes, performance_metrics):
#     values = metrics[metric]
#     bars = ax.bar(models, values, color=colors)
#     ax.set_title(metric, fontsize=12, fontweight='bold')
#     ax.set_xticklabels(models, rotation=45, ha='right')
#     add_labels(ax, bars, values)
#     ax.grid(True, axis='y', alpha=0.3)

# plt.tight_layout()
# plt.suptitle("Model Parameters and Inference Time", fontsize=16, y=1.02)
# plt.show()

# Table III and IV

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Model names
models = [
    "FNet + SRNet",
    "Finetune", 
    "Finetune + D"
]

# QP 17-27 data
qp_low_metrics = {
    # QP = 17
    "PSNR (QP=17)": [27.45, 28.90, 0],  # 0 as placeholder
    "SSIM (QP=17)": [0.744, 0.801, 0],
    "LPIPS (QP=17)": [0.138, 0.084, 0],
    "tOF (QP=17)": [2.810, 1.892, 0],
    
    # QP = 22
    "PSNR (QP=22)": [27.221, 28.43, 0],
    "SSIM (QP=22)": [0.724, 0.773, 0],
    "LPIPS (QP=22)": [0.145, 0.092, 0],
    "tOF (QP=22)": [3.395, 2.716, 0],
    
    # QP = 27
    "PSNR (QP=27)": [26.611, 27.36, 0],
    "SSIM (QP=27)": [0.682, 0.716, 0],
    "LPIPS (QP=27)": [0.160, 0.109, 0],
    "tOF (QP=27)": [4.435, 3.965, 0]
}

# QP 32-37 data
qp_high_metrics = {
    # QP = 32
    "PSNR (QP=32)": [25.464, 25.719, 0],
    "SSIM (QP=32)": [0.618, 0.637, 0],
    "LPIPS (QP=32)": [0.183, 0.134, 0],
    "tOF (QP=32)": [5.888, 5.574, 0],
    
    # QP = 34
    "PSNR (QP=34)": [24.908, 25.065, 0],
    "SSIM (QP=34)": [0.591, 0.606, 0],
    "LPIPS (QP=34)": [0.194, 0.145, 0],
    "tOF (QP=34)": [6.532, 6.183, 0],
    
    # QP = 37
    "PSNR (QP=37)": [24.088, 24.174, 0],
    "SSIM (QP=37)": [0.555, 0.566, 0],
    "LPIPS (QP=37)": [0.208, 0.122, 0],
    "tOF (QP=37)": [7.488, 7.062, 0]
}

# Create color palette with special handling for incomplete model
base_colors = sns.color_palette("Set2", 2)  # Colors for completed models
incomplete_color = '#CCCCCC'  # Gray for incomplete model
colors = list(base_colors) + [incomplete_color]

# Helper function to add value labels
def add_labels(ax, bars, values, metric_name=""):
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        
        # Handle placeholder values
        if val == 0:
            label = "TBD"
            color = 'red'
            fontweight = 'bold'
        else:
            # Format based on metric type
            if val < 1 and val > 0:
                label = f'{val:.3f}'
            else:
                label = f'{val:.2f}'
            color = 'black'
            fontweight = 'normal'
            
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                label, ha='center', va='bottom', fontsize=9, 
                fontweight=fontweight, color=color)

# Helper function to create plots
def create_qp_plots(metrics_dict, title_prefix, qp_values):
    # Create 4x3 subplot grid (4 metrics x 3 QP values)
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    
    metric_types = ["PSNR", "SSIM", "LPIPS", "tOF"]
    
    for row, metric_type in enumerate(metric_types):
        for col, qp in enumerate(qp_values):
            metric_key = f"{metric_type} (QP={qp})"
            values = metrics_dict[metric_key]
            
            ax = axes[row, col]
            bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.8)
            
            # Special formatting for incomplete model bar
            bars[2].set_alpha(0.5)
            bars[2].set_hatch('xxx')
            
            ax.set_title(f"{metric_type} (QP={qp})", fontsize=12, fontweight='bold', pad=15)
            ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
            add_labels(ax, bars, values, metric_key)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add direction indicator
            if metric_type in ["PSNR", "SSIM"]:
                ax.text(0.02, 0.98, "↑ Better", transform=ax.transAxes, 
                        fontsize=9, va='top', ha='left', color='green', fontweight='bold')
            else:
                ax.text(0.02, 0.98, "↓ Better", transform=ax.transAxes, 
                        fontsize=9, va='top', ha='left', color='green', fontweight='bold')
            
            # Set y-axis to start from 0 or appropriate minimum
            if max(values) > 0:  # Only if there are non-zero values
                if metric_type in ["LPIPS", "tOF"]:
                    ax.set_ylim(0, max(values) * 1.2)
                else:
                    ax.set_ylim(0, max(values) * 1.1)
    
    plt.tight_layout()
    plt.suptitle(title_prefix, fontsize=20, y=0.98, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=base_colors[0], 
                      edgecolor='black', linewidth=0.8, label='FNet + SRNet'),
        plt.Rectangle((0,0),1,1, facecolor=base_colors[1], 
                      edgecolor='black', linewidth=0.8, label='Finetune'),
        plt.Rectangle((0,0),1,1, facecolor=incomplete_color, alpha=0.5, hatch='xxx',
                      edgecolor='black', linewidth=0.8, label='Finetune + D (Training in Progress)')
    ]
    plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
                  ncol=3, fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.show()

# Create plots for QP 17-27
create_qp_plots(qp_low_metrics, "Performance on Images of QP 17-27", [17, 22, 27])

# Create plots for QP 32-37  
create_qp_plots(qp_high_metrics, "Performance on Images of QP 32-37", [32, 34, 37])

# Create summary comparison across all QP values
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Collect data for trend analysis
qp_values_all = [17, 22, 27, 32, 34, 37]
metrics_for_trend = ["PSNR", "SSIM", "LPIPS", "tOF"]

for idx, metric in enumerate(metrics_for_trend):
    ax = axes[idx // 2, idx % 2]
    
    # Extract data for each model across all QP values
    for model_idx, model in enumerate(models[:2]):  # Only plot completed models
        values = []
        for qp in qp_values_all:
            if qp <= 27:
                values.append(qp_low_metrics[f"{metric} (QP={qp})"][model_idx])
            else:
                values.append(qp_high_metrics[f"{metric} (QP={qp})"][model_idx])
        
        ax.plot(qp_values_all, values, marker='o', linewidth=2, markersize=8, 
                label=model, color=colors[model_idx])
        
        # Add value labels
        for qp, val in zip(qp_values_all, values):
            ax.annotate(f'{val:.2f}' if val >= 1 else f'{val:.3f}', 
                       (qp, val), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
    
    ax.set_xlabel('QP Value', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} vs QP Value', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add direction indicator
    if metric in ["PSNR", "SSIM"]:
        ax.text(0.02, 0.98, "↑ Better", transform=ax.transAxes, 
                fontsize=10, va='top', ha='left', color='green', fontweight='bold')
    else:
        ax.text(0.02, 0.98, "↓ Better", transform=ax.transAxes, 
                fontsize=10, va='top', ha='left', color='green', fontweight='bold')

plt.tight_layout()
plt.suptitle("Performance Trends Across QP Values", fontsize=18, y=0.98, fontweight='bold')
plt.show()