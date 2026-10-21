import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs("results/plots", exist_ok=True)

print(">> Loading cached datasets...")

# -------------------------------------------------------------
# 1. LOAD UCI-HAR
# -------------------------------------------------------------
uci_cache = np.load("Data/UCI_HAR/UCI HAR Dataset/cache_6ch.npz")
x_uci = uci_cache["x_all"]       # [N, 6, 128]
y_uci = uci_cache["y_all"]       # [N]
sub_uci = uci_cache["subj_all"]   # [N]

uci_labels_map = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

# -------------------------------------------------------------
# 2. LOAD HHAR
# -------------------------------------------------------------
hhar_cache = np.load("Data/heterogeneity/Activity recognition exp/cache/hhar_s3_to_nexus4_w128.npz")
x_hhar_s3 = hhar_cache["x_src"]      # [N_s, 6, 128] (Samsung S3)
y_hhar_s3 = hhar_cache["y_src"]
x_hhar_nexus = hhar_cache["x_tgt"]   # [N_t, 6, 128] (LG Nexus 4)
y_hhar_nexus = hhar_cache["y_tgt"]

hhar_labels_map = {
    0: "Bike",
    1: "Sit",
    2: "Stand",
    3: "Walk",
    4: "Stairs Up",
    5: "Stairs Down"
}

# -------------------------------------------------------------
# 3. LOAD PAMAP2
# -------------------------------------------------------------
pamap_cache = np.load("Data/PAMAP2_Dataset/cache/pamap2_chest_to_wrist_w128.npz")
x_pamap_chest = pamap_cache["x_src"]  # [N, 6, 128] (Chest IMU)
x_pamap_wrist = pamap_cache["x_tgt"]  # [N, 6, 128] (Wrist IMU)
y_pamap = pamap_cache["y_all"]

pamap_labels_map = {
    0: "Lying", 1: "Sitting", 2: "Standing", 3: "Walking", 4: "Running", 5: "Cycling",
    6: "Nordic Walk", 7: "Stairs Up", 8: "Stairs Down", 9: "Vacuum", 10: "Ironing", 11: "Rope Jump"
}

print(">> Datasets loaded successfully:")
print(f"   UCI-HAR: {x_uci.shape[0]} windows, 6 channels, 128 timesteps")
print(f"   HHAR: S3={x_hhar_s3.shape[0]} windows, Nexus4={x_hhar_nexus.shape[0]} windows")
print(f"   PAMAP2: {x_pamap_chest.shape[0]} synchronized windows")

# Standard color palette for 3 axes
c_acc = ['#1f77b4', '#ff7f0e', '#2ca02c'] # x, y, z
c_gyro = ['#d62728', '#9467bd', '#8c564b'] # x, y, z

# =============================================================
# PLOT 1: UCI-HAR - 6-AXIS SIGNALS (WALKING vs SITTING)
# =============================================================
print(">> Generating Plot 1: UCI-HAR Signals (Pure presentation, English)...")
fig = plt.figure(figsize=(15, 8), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.18)

# Representative samples for Walking (0) and Sitting (3)
walk_idx = np.where(y_uci == 0)[0][10]
sit_idx = np.where(y_uci == 3)[0][10]
time_axis_uci = np.arange(128) / 50.0  # 50 Hz -> seconds

# Top Left: Walking - Accelerometer
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_axis_uci, x_uci[walk_idx, 0], label=r'Acc $X$', color=c_acc[0], lw=1.6)
ax1.plot(time_axis_uci, x_uci[walk_idx, 1], label=r'Acc $Y$', color=c_acc[1], lw=1.6)
ax1.plot(time_axis_uci, x_uci[walk_idx, 2], label=r'Acc $Z$', color=c_acc[2], lw=1.6)
ax1.set_title("Walking — Accelerometer", fontsize=12, fontweight='bold')
ax1.set_ylabel("Acceleration (g)", fontsize=11)
ax1.legend(loc='upper right', frameon=True, fontsize=9, ncol=3)
ax1.set_xlim(0, 128 / 50.0)

# Bottom Left: Walking - Gyroscope
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_axis_uci, x_uci[walk_idx, 3], label=r'Gyro $X$', color=c_gyro[0], lw=1.6)
ax2.plot(time_axis_uci, x_uci[walk_idx, 4], label=r'Gyro $Y$', color=c_gyro[1], lw=1.6)
ax2.plot(time_axis_uci, x_uci[walk_idx, 5], label=r'Gyro $Z$', color=c_gyro[2], lw=1.6)
ax2.set_title("Walking — Gyroscope", fontsize=12, fontweight='bold')
ax2.set_xlabel("Time (seconds)", fontsize=11)
ax2.set_ylabel("Angular Velocity (rad/s)", fontsize=11)
ax2.legend(loc='upper right', frameon=True, fontsize=9, ncol=3)
ax2.set_xlim(0, 128 / 50.0)

# Top Right: Sitting - Accelerometer
ax3 = fig.add_subplot(gs[0, 1])
ax3.plot(time_axis_uci, x_uci[sit_idx, 0], label=r'Acc $X$', color=c_acc[0], lw=1.6)
ax3.plot(time_axis_uci, x_uci[sit_idx, 1], label=r'Acc $Y$', color=c_acc[1], lw=1.6)
ax3.plot(time_axis_uci, x_uci[sit_idx, 2], label=r'Acc $Z$', color=c_acc[2], lw=1.6)
ax3.set_title("Sitting — Accelerometer", fontsize=12, fontweight='bold')
ax3.set_ylabel("Acceleration (g)", fontsize=11)
ax3.legend(loc='upper right', frameon=True, fontsize=9, ncol=3)
ax3.set_xlim(0, 128 / 50.0)

# Bottom Right: Sitting - Gyroscope
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time_axis_uci, x_uci[sit_idx, 3], label=r'Gyro $X$', color=c_gyro[0], lw=1.6)
ax4.plot(time_axis_uci, x_uci[sit_idx, 4], label=r'Gyro $Y$', color=c_gyro[1], lw=1.6)
ax4.plot(time_axis_uci, x_uci[sit_idx, 5], label=r'Gyro $Z$', color=c_gyro[2], lw=1.6)
ax4.set_title("Sitting — Gyroscope", fontsize=12, fontweight='bold')
ax4.set_xlabel("Time (seconds)", fontsize=11)
ax4.set_ylabel("Angular Velocity (rad/s)", fontsize=11)
ax4.legend(loc='upper right', frameon=True, fontsize=9, ncol=3)
ax4.set_xlim(0, 128 / 50.0)

plt.suptitle("UCI-HAR Dataset: 6-Axis Inertial Signals (Walking vs. Sitting)", fontsize=15, fontweight='bold', y=0.98)
plt.savefig("results/plots/dataset_uci_har_signals.png", bbox_inches='tight')
plt.close()

# =============================================================
# PLOT 2: HHAR - DEVICE COMPARISON (SAMSUNG S3 vs LG NEXUS 4)
# =============================================================
print(">> Generating Plot 2: HHAR Device Comparison (Pure presentation, English)...")
fig, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=300, sharex=True)

walk_s3_idx = np.where(y_hhar_s3 == 3)[0][15]
walk_nexus_idx = np.where(y_hhar_nexus == 3)[0][15]

# Top-Left: S3 Accelerometer
axes[0, 0].plot(x_hhar_s3[walk_s3_idx, 0], label=r'Acc $X$', color=c_acc[0], lw=1.6)
axes[0, 0].plot(x_hhar_s3[walk_s3_idx, 1], label=r'Acc $Y$', color=c_acc[1], lw=1.6)
axes[0, 0].plot(x_hhar_s3[walk_s3_idx, 2], label=r'Acc $Z$', color=c_acc[2], lw=1.6)
axes[0, 0].set_title("Samsung Galaxy S3 — Accelerometer", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("Normalized Amplitude (Z-Score)", fontsize=11)
axes[0, 0].legend(loc='upper right', fontsize=9, ncol=3)

# Bottom-Left: S3 Gyroscope
axes[1, 0].plot(x_hhar_s3[walk_s3_idx, 3], label=r'Gyro $X$', color=c_gyro[0], lw=1.6)
axes[1, 0].plot(x_hhar_s3[walk_s3_idx, 4], label=r'Gyro $Y$', color=c_gyro[1], lw=1.6)
axes[1, 0].plot(x_hhar_s3[walk_s3_idx, 5], label=r'Gyro $Z$', color=c_gyro[2], lw=1.6)
axes[1, 0].set_title("Samsung Galaxy S3 — Gyroscope", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Sample Index (128 samples)", fontsize=11)
axes[1, 0].set_ylabel("Normalized Amplitude (Z-Score)", fontsize=11)
axes[1, 0].legend(loc='upper right', fontsize=9, ncol=3)

# Top-Right: Nexus 4 Accelerometer
axes[0, 1].plot(x_hhar_nexus[walk_nexus_idx, 0], label=r'Acc $X$', color=c_acc[0], lw=1.6)
axes[0, 1].plot(x_hhar_nexus[walk_nexus_idx, 1], label=r'Acc $Y$', color=c_acc[1], lw=1.6)
axes[0, 1].plot(x_hhar_nexus[walk_nexus_idx, 2], label=r'Acc $Z$', color=c_acc[2], lw=1.6)
axes[0, 1].set_title("LG Nexus 4 — Accelerometer", fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel("Normalized Amplitude (Z-Score)", fontsize=11)
axes[0, 1].legend(loc='upper right', fontsize=9, ncol=3)

# Bottom-Right: Nexus 4 Gyroscope
axes[1, 1].plot(x_hhar_nexus[walk_nexus_idx, 3], label=r'Gyro $X$', color=c_gyro[0], lw=1.6)
axes[1, 1].plot(x_hhar_nexus[walk_nexus_idx, 4], label=r'Gyro $Y$', color=c_gyro[1], lw=1.6)
axes[1, 1].plot(x_hhar_nexus[walk_nexus_idx, 5], label=r'Gyro $Z$', color=c_gyro[2], lw=1.6)
axes[1, 1].set_title("LG Nexus 4 — Gyroscope", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Sample Index (128 samples)", fontsize=11)
axes[1, 1].set_ylabel("Normalized Amplitude (Z-Score)", fontsize=11)
axes[1, 1].legend(loc='upper right', fontsize=9, ncol=3)

plt.suptitle("HHAR Dataset: Signal Comparison Between Devices (Walking)", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("results/plots/dataset_hhar_cross_device.png", bbox_inches='tight')
plt.close()

# =============================================================
# PLOT 3: PAMAP2 - PLACEMENT COMPARISON (CHEST vs WRIST)
# =============================================================
print(">> Generating Plot 3: PAMAP2 Placement Comparison (Pure presentation, English)...")
fig, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=300, sharex=True)

walk_pamap_idx = np.where(y_pamap == 3)[0][25]

# Top-Left: Chest Accelerometer
axes[0, 0].plot(x_pamap_chest[walk_pamap_idx, 0], label=r'Acc $X$', color=c_acc[0], lw=1.6)
axes[0, 0].plot(x_pamap_chest[walk_pamap_idx, 1], label=r'Acc $Y$', color=c_acc[1], lw=1.6)
axes[0, 0].plot(x_pamap_chest[walk_pamap_idx, 2], label=r'Acc $Z$', color=c_acc[2], lw=1.6)
axes[0, 0].set_title("Chest Placement — Accelerometer", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("Normalized Amplitude (Z-Score)", fontsize=11)
axes[0, 0].legend(loc='upper right', fontsize=9, ncol=3)

# Bottom-Left: Chest Gyroscope
axes[1, 0].plot(x_pamap_chest[walk_pamap_idx, 3], label=r'Gyro $X$', color=c_gyro[0], lw=1.6)
axes[1, 0].plot(x_pamap_chest[walk_pamap_idx, 4], label=r'Gyro $Y$', color=c_gyro[1], lw=1.6)
axes[1, 0].plot(x_pamap_chest[walk_pamap_idx, 5], label=r'Gyro $Z$', color=c_gyro[2], lw=1.6)
axes[1, 0].set_title("Chest Placement — Gyroscope", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Sample Index (128 samples)", fontsize=11)
axes[1, 0].set_ylabel("Normalized Amplitude (Z-Score)", fontsize=11)
axes[1, 0].legend(loc='upper right', fontsize=9, ncol=3)

# Top-Right: Wrist Accelerometer
axes[0, 1].plot(x_pamap_wrist[walk_pamap_idx, 0], label=r'Acc $X$', color=c_acc[0], lw=1.6)
axes[0, 1].plot(x_pamap_wrist[walk_pamap_idx, 1], label=r'Acc $Y$', color=c_acc[1], lw=1.6)
axes[0, 1].plot(x_pamap_wrist[walk_pamap_idx, 2], label=r'Acc $Z$', color=c_acc[2], lw=1.6)
axes[0, 1].set_title("Wrist Placement — Accelerometer", fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel("Normalized Amplitude (Z-Score)", fontsize=11)
axes[0, 1].legend(loc='upper right', fontsize=9, ncol=3)

# Bottom-Right: Wrist Gyroscope
axes[1, 1].plot(x_pamap_wrist[walk_pamap_idx, 3], label=r'Gyro $X$', color=c_gyro[0], lw=1.6)
axes[1, 1].plot(x_pamap_wrist[walk_pamap_idx, 4], label=r'Gyro $Y$', color=c_gyro[1], lw=1.6)
axes[1, 1].plot(x_pamap_wrist[walk_pamap_idx, 5], label=r'Gyro $Z$', color=c_gyro[2], lw=1.6)
axes[1, 1].set_title("Wrist Placement — Gyroscope", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Sample Index (128 samples)", fontsize=11)
axes[1, 1].set_ylabel("Normalized Amplitude (Z-Score)", fontsize=11)
axes[1, 1].legend(loc='upper right', fontsize=9, ncol=3)

plt.suptitle("PAMAP2 Dataset: Signal Comparison Across Body Placements (Walking)", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("results/plots/dataset_pamap2_cross_placement.png", bbox_inches='tight')
plt.close()

# =============================================================
# PLOT 4: COMPREHENSIVE OVERVIEW DASHBOARD
# =============================================================
print(">> Generating Plot 4: Comprehensive Master Dashboard (English)...")
fig = plt.figure(figsize=(18, 10), dpi=300)
gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

# Row 1, Col 1: UCI-HAR Representative Waveform
ax_w1 = fig.add_subplot(gs[0, 0])
ax_w1.plot(time_axis_uci, x_uci[walk_idx, 0], label=r'$a_x$', color=c_acc[0], lw=1.5)
ax_w1.plot(time_axis_uci, x_uci[walk_idx, 1], label=r'$a_y$', color=c_acc[1], lw=1.5)
ax_w1.plot(time_axis_uci, x_uci[walk_idx, 2], label=r'$a_z$', color=c_acc[2], lw=1.5)
ax_w1.plot(time_axis_uci, x_uci[walk_idx, 3], label=r'$g_x$', color=c_gyro[0], lw=1.2, ls='--')
ax_w1.plot(time_axis_uci, x_uci[walk_idx, 4], label=r'$g_y$', color=c_gyro[1], lw=1.2, ls='--')
ax_w1.plot(time_axis_uci, x_uci[walk_idx, 5], label=r'$g_z$', color=c_gyro[2], lw=1.2, ls='--')
ax_w1.set_title("1. UCI-HAR (Waist Placement - Walking)", fontsize=11, fontweight='bold')
ax_w1.set_xlabel("Time (seconds) — 50 Hz", fontsize=10)
ax_w1.set_ylabel("Sensor Signals (6 Axes)", fontsize=10)
ax_w1.legend(loc='upper right', fontsize=8, ncol=3)

# Row 1, Col 2: HHAR Waveform Comparison (S3 vs Nexus 4)
ax_w2 = fig.add_subplot(gs[0, 1])
ax_w2.plot(x_hhar_s3[walk_s3_idx, 0], label='S3 Acc X', color='#1f77b4', lw=1.5)
ax_w2.plot(x_hhar_nexus[walk_nexus_idx, 0], label='Nexus 4 Acc X', color='#d62728', lw=1.5, ls='--')
ax_w2.plot(x_hhar_s3[walk_s3_idx, 1], label='S3 Acc Y', color='#2ca02c', lw=1.5)
ax_w2.plot(x_hhar_nexus[walk_nexus_idx, 1], label='Nexus 4 Acc Y', color='#ff7f0e', lw=1.5, ls='--')
ax_w2.set_title("2. HHAR (Samsung Galaxy S3 vs. LG Nexus 4)", fontsize=11, fontweight='bold')
ax_w2.set_xlabel("Sample Index (128 samples)", fontsize=10)
ax_w2.set_ylabel("Normalized Acceleration", fontsize=10)
ax_w2.legend(loc='upper right', fontsize=8, ncol=2)

# Row 1, Col 3: PAMAP2 Waveform Comparison (Chest vs Wrist)
ax_w3 = fig.add_subplot(gs[0, 2])
ax_w3.plot(x_pamap_chest[walk_pamap_idx, 0], label='Chest Acc X', color='#2ca02c', lw=1.6)
ax_w3.plot(x_pamap_wrist[walk_pamap_idx, 0], label='Wrist Acc X', color='#9467bd', lw=1.6, ls='--')
ax_w3.plot(x_pamap_chest[walk_pamap_idx, 2], label='Chest Acc Z', color='#1f77b4', lw=1.6)
ax_w3.plot(x_pamap_wrist[walk_pamap_idx, 2], label='Wrist Acc Z', color='#e377c2', lw=1.6, ls='--')
ax_w3.set_title("3. PAMAP2 (Chest vs. Wrist Placements)", fontsize=11, fontweight='bold')
ax_w3.set_xlabel("Sample Index (100 Hz)", fontsize=10)
ax_w3.set_ylabel("Normalized Acceleration", fontsize=10)
ax_w3.legend(loc='upper right', fontsize=8, ncol=2)

# Row 2, Col 1: Class Distribution UCI-HAR
ax_c1 = fig.add_subplot(gs[1, 0])
classes_uci, counts_uci = np.unique(y_uci, return_counts=True)
uci_names = [uci_labels_map[c] for c in classes_uci]
bars1 = ax_c1.barh(uci_names, counts_uci, color='#4575b4', alpha=0.85, edgecolor='black', height=0.6)
ax_c1.set_title(f"Class Distribution: UCI-HAR (Total: {len(y_uci)} samples)", fontsize=11, fontweight='bold')
ax_c1.set_xlabel("Number of Windows", fontsize=10)
for bar in bars1:
    ax_c1.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}", va='center', fontsize=8)

# Row 2, Col 2: Class Distribution HHAR
ax_c2 = fig.add_subplot(gs[1, 1])
classes_hhar, counts_hhar = np.unique(y_hhar_s3, return_counts=True)
hhar_names = [hhar_labels_map[c] for c in classes_hhar]
bars2 = ax_c2.barh(hhar_names, counts_hhar, color='#d73027', alpha=0.85, edgecolor='black', height=0.6)
ax_c2.set_title(f"Class Distribution: HHAR (Total: {len(y_hhar_s3)} samples)", fontsize=11, fontweight='bold')
ax_c2.set_xlabel("Number of Windows", fontsize=10)
for bar in bars2:
    ax_c2.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}", va='center', fontsize=8)

# Row 2, Col 3: Class Distribution PAMAP2
ax_c3 = fig.add_subplot(gs[1, 2])
classes_pamap, counts_pamap = np.unique(y_pamap, return_counts=True)
pamap_names = [pamap_labels_map[c] for c in classes_pamap]
bars3 = ax_c3.barh(pamap_names, counts_pamap, color='#313695', alpha=0.85, edgecolor='black', height=0.6)
ax_c3.set_title(f"Class Distribution: PAMAP2 (Total: {len(y_pamap)} samples)", fontsize=11, fontweight='bold')
ax_c3.set_xlabel("Number of Windows", fontsize=10)
for bar in bars3:
    ax_c3.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}", va='center', fontsize=8)

plt.suptitle("Benchmark HAR Datasets: Representative Waveforms and Class Distributions", fontsize=15, fontweight='bold', y=0.98)
plt.savefig("results/plots/har_datasets_benchmark_infographic.png", bbox_inches='tight')
plt.close()

print("[SUCCESS] All 4 plots have been successfully regenerated in English without subjective evaluation text!")
print(" - results/plots/dataset_uci_har_signals.png")
print(" - results/plots/dataset_hhar_cross_device.png")
print(" - results/plots/dataset_pamap2_cross_placement.png")
print(" - results/plots/har_datasets_benchmark_infographic.png")
