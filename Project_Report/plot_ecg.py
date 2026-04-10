import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Generate synthetic ECG
np.random.seed(42)
t = np.linspace(0, 3, 1080) # 3 seconds at 360 Hz
# Basic beat
p_wave = 0.15 * np.exp(-((t % 1 - 0.2)**2) / (2 * 0.02**2))
qrs_q = -0.1 * np.exp(-((t % 1 - 0.4)**2) / (2 * 0.01**2))
qrs_r = 1.0 * np.exp(-((t % 1 - 0.45)**2) / (2 * 0.015**2))
qrs_s = -0.2 * np.exp(-((t % 1 - 0.5)**2) / (2 * 0.01**2))
t_wave = 0.3 * np.exp(-((t % 1 - 0.7)**2) / (2 * 0.04**2))
clean_ecg = p_wave + qrs_q + qrs_r + qrs_s + t_wave

# 1. Raw Signal (add baseline wander and high freq noise)
baseline = 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.sin(2 * np.pi * 0.1 * t)
noise = np.random.normal(0, 0.08, len(t))
raw_ecg = clean_ecg + baseline + noise

# 2. Bandpass Filtered (clean ECG with mild noise but no baseline)
filtered_ecg = clean_ecg + np.random.normal(0, 0.02, len(t))

# 3. Segmented Beat Window (0.7s around a beat -> from t=1.15 to t=1.85)
# One beat centers at 1.45
start_idx = int(1.15 * 360)
end_idx = int(1.85 * 360)
t_seg = t[start_idx:end_idx]
seg_ecg = filtered_ecg[start_idx:end_idx]

fig, axs = plt.subplots(3, 1, figsize=(8, 6))

axs[0].plot(t, raw_ecg, color='#7f7f7f', lw=1)
axs[0].set_title('Raw Signal (with Baseline Wander and Noise)', fontsize=11, pad=5)
axs[0].axis('off')

axs[1].plot(t, filtered_ecg, color='#1f77b4', lw=1.2)
axs[1].set_title('Bandpass Filtered Signal (0.5 – 40 Hz)', fontsize=11, pad=5)
axs[1].axis('off')

axs[2].plot(np.linspace(0, 0.7, len(seg_ecg)), seg_ecg, color='#d62728', lw=1.5)
axs[2].set_title('Segmented Beat Window (0.3s pre-R, 0.4s post-R)', fontsize=11, pad=5)
axs[2].axis('off')

plt.tight_layout()
os.makedirs('/Users/harshgoyal/Desktop/capstone/Project_Report/images', exist_ok=True)
plt.savefig('/Users/harshgoyal/Desktop/capstone/Project_Report/images/ecg_preprocessing.png', dpi=150, bbox_inches='tight')
print("Saved ecg_preprocessing.png")
