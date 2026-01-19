# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# CSV_FILE = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data/HiCInterpolateV2/output/config_64_set_18_local/hicinterpolate_64_val_metrics.csv"

# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial']
# nature_colors = ["#d55e00", "#009e74", "#f0e442", "#0072b2",
#                  "#56b3e9", "#e69f00", "#cc79a7", "#000000"]

# df = pd.read_csv(CSV_FILE)

# df_filtered = df[df['epoch'] <= 300].copy()

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# # fig.suptitle('Validation Metrics', fontsize=16, y=0.995)

# epoch_92_data = df_filtered[df_filtered['epoch'] == 92].iloc[0]

# # Plot 1: PSNR
# ax1 = axes[0, 0]
# ax1.plot(df_filtered['epoch'], df_filtered['val_psnr'],
#          color=nature_colors[0], linewidth=2)
# ax1.axvline(x=92, color=nature_colors[6], linestyle='--', linewidth=1.5, alpha=0.7)
# ax1.text(92, 0.70, '92', transform=ax1.get_xaxis_transform(), rotation='vertical', color=nature_colors[7], fontweight='bold')
# ax1.plot(92, epoch_92_data['val_psnr'], 'o', color=nature_colors[0],
#          markersize=8, markeredgecolor=nature_colors[7], markeredgewidth=1.5)

# ax1.set_xlabel('Epoch', fontsize=11)
# ax1.set_ylabel('PSNR (dB)', fontsize=11)
# ax1.set_title('Peak Signal-to-Noise Ratio (PSNR)', fontsize=12)
# ax1.grid(True, alpha=0.3)
# ax1.set_xlim([df_filtered['epoch'].min(), df_filtered['epoch'].max()])

# # Plot 2: SSIM
# ax2 = axes[0, 1]
# ax2.plot(df_filtered['epoch'], df_filtered['val_ssim'],
#          color=nature_colors[1], linewidth=2)
# ax2.axvline(x=92, color=nature_colors[6], linestyle='--', linewidth=1.5, alpha=0.7)
# ax2.text(92, 0.70, '92', transform=ax2.get_xaxis_transform(), rotation='vertical', color=nature_colors[7], fontweight='bold')
# ax2.plot(92, epoch_92_data['val_ssim'], 'o', color=nature_colors[1],
#          markersize=8, markeredgecolor=nature_colors[7], markeredgewidth=1.5)
# ax2.set_xlabel('Epoch', fontsize=11)
# ax2.set_ylabel('SSIM', fontsize=11)
# ax2.set_title('Structural Similarity Index (SSIM)', fontsize=12)
# ax2.grid(True, alpha=0.3)
# ax2.set_xlim([df_filtered['epoch'].min(), df_filtered['epoch'].max()])
# ax2.set_ylim([0, 1])

# # Plot 3: GenomeDISCO
# ax3 = axes[1, 0]
# ax3.plot(df_filtered['epoch'], df_filtered['val_genome_disco'],
#          color=nature_colors[2], linewidth=2)
# ax3.axvline(x=92, color=nature_colors[6], linestyle='--', linewidth=1.5, alpha=0.7)
# ax3.text(92, 0.70, '92', transform=ax3.get_xaxis_transform(), rotation='vertical', color=nature_colors[7], fontweight='bold')
# ax3.plot(92, epoch_92_data['val_genome_disco'], 'o', color=nature_colors[2],
#          markersize=8, markeredgecolor=nature_colors[7], markeredgewidth=1.5)
# ax3.set_xlabel('Epoch', fontsize=11)
# ax3.set_ylabel('GenomeDISCO', fontsize=11)
# ax3.set_title('GenomeDISCO Score', fontsize=12)
# ax3.grid(True, alpha=0.3)
# ax3.set_xlim([df_filtered['epoch'].min(), df_filtered['epoch'].max()])
# ax3.set_ylim([0, 1])

# # Plot 4: HiCRep
# ax4 = axes[1, 1]
# ax4.plot(df_filtered['epoch'], df_filtered['val_hicrep'],
#          color=nature_colors[3], linewidth=2)
# ax4.axvline(x=92, color=nature_colors[6], linestyle='--', linewidth=1.5, alpha=0.7)
# ax4.text(92, 0.70, '92', transform=ax4.get_xaxis_transform(), rotation='vertical', color=nature_colors[7], fontweight='bold')
# ax4.plot(92, epoch_92_data['val_hicrep'], 'o', color=nature_colors[3],
#          markersize=8, markeredgecolor=nature_colors[7], markeredgewidth=1.5)
# ax4.set_xlabel('Epoch', fontsize=11)
# ax4.set_ylabel('HiCRep', fontsize=11)
# ax4.set_title('HiCRep Score', fontsize=12)
# ax4.grid(True, alpha=0.3)
# ax4.set_xlim([df_filtered['epoch'].min(), df_filtered['epoch'].max()])
# ax4.set_ylim([0, 1])

# plt.tight_layout()
# plt.savefig('validation_metrics.png', dpi=300, bbox_inches='tight')
# plt.savefig('validation_metrics.pdf', dpi=300, bbox_inches='tight')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
CSV_FILE = "/home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data/HiCInterpolateV2/output/config_64_set_18_local/hicinterpolate_64_val_metrics.csv"
highlight_epoch = 92  # the epoch to highlight

# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial']
# nature_colors = ["#d55e00", "#009e74", "#f0e442", "#0072b2",
#                  "#56b3e9", "#e69f00", "#cc79a7", "#000000"]

# # --- Load data ---
# df = pd.read_csv(CSV_FILE)
# df_filtered = df[df['epoch'] <= 300].copy()

# # --- Extract metrics at highlight epoch ---
# epoch_data = df_filtered[df_filtered['epoch'] == highlight_epoch].iloc[0]

# # --- Create figure ---
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# # fig.suptitle('Validation Metrics', fontsize=16, y=0.995)

# # --- Helper function to plot each metric ---
# def plot_metric(ax, metric_name, y_label, color):
#     val = epoch_data[metric_name]
#     ax.plot(df_filtered['epoch'], df_filtered[metric_name], color=color, linewidth=2)
#     # vertical line
#     ax.axvline(x=highlight_epoch, color=nature_colors[6], linestyle='--', linewidth=1.5, alpha=0.7)
#     # text label on top of vertical line (epoch number)
#     ax.text(highlight_epoch, 0.70, str(highlight_epoch), transform=ax.get_xaxis_transform(),
#             rotation='vertical', color=nature_colors[7], fontweight='bold')
#     # marker at highlight epoch
#     ax.plot(highlight_epoch, val, 'o', color=color, markersize=8,
#             markeredgecolor=nature_colors[7], markeredgewidth=1.5)
#     # display metric value next to the marker with some space and black color
#     ax.text(highlight_epoch + 4, val, f"{val:.3f}", color='black', fontsize=10, fontweight='bold')

#     ax.set_xlabel('Epoch', fontsize=11)
#     ax.set_ylabel(y_label, fontsize=11)
#     ax.set_title(y_label, fontsize=12)
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim([df_filtered['epoch'].min(), df_filtered['epoch'].max()])
#     if metric_name in ['val_ssim', 'val_genome_disco', 'val_hicrep']:
#         ax.set_ylim([0, 1])

# # --- Plot all metrics ---
# plot_metric(axes[0, 0], 'val_psnr', 'Peak Signal-to-Noise Ratio (PSNR)', nature_colors[0])
# plot_metric(axes[0, 1], 'val_ssim', 'Structural Similarity Index (SSIM)', nature_colors[1])
# plot_metric(axes[1, 0], 'val_genome_disco', 'GenomeDISCO Score', nature_colors[2])
# plot_metric(axes[1, 1], 'val_hicrep', 'HiCRep Score', nature_colors[3])

# plt.tight_layout()
# plt.savefig('validation_metrics.png', dpi=300, bbox_inches='tight')
# plt.savefig('validation_metrics.pdf', dpi=300, bbox_inches='tight')
# plt.show()


# --- Global style ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

nature_colors = [
    "#d55e00", "#009e74", "#f0e442", "#0072b2",
    "#56b3e9", "#e69f00", "#cc79a7", "#000000"
]

# --- Load data ---
df = pd.read_csv(CSV_FILE)
df_filtered = df[df['epoch'] <= 300].copy()

# --- Extract metrics at highlight epoch ---
epoch_data = df_filtered[df_filtered['epoch'] == highlight_epoch].iloc[0]

# --- Helper function ---


def plot_and_save(metric_name, y_label, color, y_min, y_max, filename):
    fig, ax = plt.subplots(figsize=(7, 5))

    val = epoch_data[metric_name]

    ax.plot(df_filtered['epoch'], df_filtered[metric_name],
            color=color, linewidth=2)

    ax.axvline(x=highlight_epoch, color=nature_colors[6],
               linestyle='--', linewidth=1.5, alpha=0.7)

    ax.text(highlight_epoch, 0.20, str(highlight_epoch),
            fontsize=18,
            transform=ax.get_xaxis_transform(),
            rotation='horizontal', color=nature_colors[7],
            fontweight='normal')

    ax.plot(highlight_epoch, val, 'o', color=color, markersize=8,
            markeredgecolor=nature_colors[7], markeredgewidth=1.5)

    ax.text(highlight_epoch + 4, val, f"{val:.3f}",
            color='black', fontsize=22, fontweight='normal')

    ax.set_xlabel('Epoch', fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(df_filtered['epoch'].min(), df_filtered['epoch'].max())

    # if metric_name in ['val_ssim', 'val_genome_disco', 'val_hicrep']:
    #     ax.set_ylim([0, 1])

    ax.tick_params(axis='both', which='major', labelsize=22)

    ax.set_ylim(y_min , y_max)

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


# --- Generate individual plots ---
plot_and_save('val_psnr',
              'dB',
              nature_colors[0],
              37, 38.5,
              'val_psnr')

plot_and_save('val_ssim',
              'SSIM',
              nature_colors[1],
              0.90, 0.93,
              'val_ssim')

plot_and_save('val_genome_disco',
              'GenomeDISCO',
              nature_colors[2],
              0.78, 0.83,
              'val_genomedisco')

plot_and_save('val_hicrep',
              'SCC',
              nature_colors[3],
                            0.20, 0.23,
              'val_hicrep')
