import matplotlib.pyplot as plt
import numpy as np

# QP values
qp_values = [17, 22, 27, 32, 34, 37]

# PSNR values for both models across QPs
psnr_baseline = [27.45, 27.22, 26.61, 25.46, 24.91, 24.09]
psnr_improved = [28.90, 28.43, 27.36, 25.72, 25.07, 24.17]

# Determine y-axis limits and ticks
psnr_min = min(min(psnr_baseline), min(psnr_improved)) - 0.2
psnr_max = max(max(psnr_baseline), max(psnr_improved)) + 0.2
psnr_ticks = np.arange(np.floor(psnr_min * 5) / 5, np.ceil(psnr_max * 5) / 5 + 0.01, 1)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(qp_values, psnr_baseline, marker='o', label='FNet + SRNet 12 Baseline')
plt.plot(qp_values, psnr_improved, marker='o', linestyle='--', label='FNet + SRNet 12 Improved')

plt.xlabel("QP")
plt.ylabel("PSNR (↑) [dB]")
plt.title("PSNR vs QP")
plt.grid(True)
plt.xticks(qp_values)  # QP axis as integers only
plt.yticks(psnr_ticks)  # PSNR axis in steps of 0.2
plt.legend()
plt.tight_layout()

plt.show()
