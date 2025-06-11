import os
import sys
import copy

import time
import timeit
import os, copy

import umap
from sklearn.decomposition import PCA

import learning_repo.general_utils as lrgu
import learning_repo.geometric_utils as lrgeo
from structure_index import compute_structure_index
import elephant
from scipy.signal import convolve
from elephant.kernels import GaussianKernel
from quantities import ms
import quantities as pq
import numpy as np
from elephant.statistics import instantaneous_rate
import quantities as pq
import neo
from scipy.ndimage import gaussian_filter1d

def smooth_calcium_signals(calcium_data, sigma=4):
    """
    Apply Gaussian smoothing to each row of the calcium imaging data.
    Each row represents the calcium signal of one neuron.

    Parameters:
    - calcium_data: 2D numpy array, where each row is a neuron's time series.
    - sigma: Standard deviation for Gaussian kernel. The value is in terms of index units.

    Returns:
    - smoothed_signals: 2D numpy array of smoothed signals.
    """
    # Initialize an array to store the smoothed signals
    smoothed_signals = np.zeros_like(calcium_data)

    # Apply Gaussian smoothing to each row (neuron)
    for i in range(calcium_data.shape[1]):
        smoothed_signals[:,i] = gaussian_filter1d(calcium_data[:,i], sigma=sigma)

    return smoothed_signals

from sklearn.metrics import pairwise_distances

def filter_noisy_outliers(data):
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,1), axis=1) - 1
    noiseIdx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    signalIdx = np.where(nn_dist >= np.percentile(nn_dist, 20))[0]
    return noiseIdx, signalIdx

base_dir =  '/home/melma31/Documents/deepsup_project/'

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'

data_dir = os.path.join(base_dir, 'MIR')
save_dir = os.path.join(base_dir, 'SI_Filters')
if not os.path.isdir(save_dir): os.makedirs(save_dir)

params1 = {
    'n_bins': 10,
    'discrete_label': False,
    'continuity_kernel': None,
    'perc_neigh': 1,
    'num_shuffles': 0,
    'verbose': False
}


params2 = {
    'n_bins': 3,
    'discrete_label': True,
    'continuity_kernel': None,
    'n_neighbors': 50,
    'num_shuffles': 0,
    'verbose': False
}

si_neigh = 50
si_beh_params = {}
for beh in ['pos', 'speed', 'trial_id_mat','time','(pos,dir)']:
    si_beh_params[beh] = copy.deepcopy(params1)
for beh in ['dir']:
    si_beh_params[beh] = copy.deepcopy(params2)

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        print(mouse)

        global_time = timeit.default_timer()  # initialize time
        local_time = timeit.default_timer()
        mdata_dir = os.path.join(data_dir, mouse)  # mouse save dir
        msave_dir = os.path.join(save_dir, mouse)  # mouse save dir
        if not os.path.isdir(msave_dir): os.makedirs(msave_dir)
        f = open(os.path.join(msave_dir, f"{mouse}_linearity_{signal_name}_logFile.txt"), 'w')
        original = sys.stdout
        sys.stdout = lrgu.Tee(sys.stdout, f)

        mi_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_mi_cluster_all_{signal_name}_dict.pkl")
        session_names = list(mi_dict.keys())
        session_names.sort()
        si_dict = dict()
        for idx, session in enumerate(session_names):
            if 'lt' in session:
                si_dict[session] = dict()
                print(f"Working on session {session} ({idx + 1}/{len(session_names)}): ", sep='', end='')
                mov_dir = mi_dict[session]['behaviour']['mov_dir']
                valid_index =  mi_dict[session]['valid_index']

                valid_index = np.arange(0,mov_dir.shape[0])
                beh_variables = {
                    'pos': mi_dict[session]['behaviour']['position'][valid_index].copy(),
                    '(pos,dir)': mi_dict[session]['behaviour']['(pos,dir)'][valid_index].copy(),
                    'speed': mi_dict[session]['behaviour']['speed'][valid_index].copy(),
                    'trial_id_mat': mi_dict[session]['behaviour']['trial_id'][valid_index].copy(),
                    'dir':  mi_dict[session]['behaviour']['mov_dir'][valid_index].copy(),
                    'time': mi_dict[session]['behaviour']['time'][valid_index].copy()
                    }

                signal = mi_dict[session]['signal']
                signal = signal[valid_index, :]
                selected_indices = np.random.choice(signal.shape[1], 20, replace=False)
                signal = signal[:, selected_indices]

                noise_idx, signal_idx = filter_noisy_outliers(signal)
                print(f"Working on session {session} ({idx + 1}/{len(session_names)}):",end='')

                for beh_name, beh_val in beh_variables.items():
                    if 'trial_id_mat' in beh_name:
                        si_beh_params[beh_name]['min_label'] = np.min(beh_val)
                        si_beh_params[beh_name]['max_label'] = np.max(beh_val)
                    # si_beh_params[beh_name]['n_neighbors'] = int(signal.shape[0]* si_beh_params[beh_name]['perc_neigh']/100)
                    si_beh_params[beh_name]['n_neighbors'] = si_neigh
                    si_dict[session][beh_name] = dict()

                    si, process_info, overlap_mat, _ = compute_structure_index(signal[signal_idx],
                                                                                   beh_val[signal_idx],
                                                                                   **si_beh_params[beh_name])

                    si_dict[session][beh_name][str(0)] = {
                        'si': copy.deepcopy(si),
                        'process_info': copy.deepcopy(process_info),
                        'overlap_mat': copy.deepcopy(overlap_mat),
                        'beh_params': copy.deepcopy(si_beh_params[beh_name]),
                        'valid_idx': valid_index.copy(),
                        'signal_idx': signal_idx.copy(),
                        'signal': signal.copy()

                    }
                    kernels = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
                    for index , filter_size in enumerate(kernels):
                        rates = smooth_calcium_signals(signal, filter_size)
                        si, process_info, overlap_mat, _ = compute_structure_index(rates[signal_idx],
                                                                                   beh_val[signal_idx],
                                                                                   **si_beh_params[beh_name])

                        si_dict[session][beh_name][str(filter_size)] = {
                            'si': copy.deepcopy(si),
                            'process_info': copy.deepcopy(process_info),
                            'overlap_mat': copy.deepcopy(overlap_mat),
                            'beh_params': copy.deepcopy(si_beh_params[beh_name]),
                            'valid_idx': valid_index.copy(),
                            'signal_idx': signal_idx.copy(),
                            'signal': rates.copy()
                        }

                        print(f" {beh_name}_{filter_size}={si:.4f} |", end='', sep='', flush='True')
                        print()

        lrgu.save_pickle(msave_dir, f"{mouse}_si_filters_{signal_name}_{si_neigh}dict.pkl", si_dict)
        lrgu.print_time_verbose(local_time, global_time)
        sys.stdout = original


#####################################################

# PLOT SI VALUES#

####################################################
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats

data_dir = os.path.join(base_dir, 'SI')
mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'
mov_cond = 'all_mov'

df_rows = []
for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        si_dict = lrgu.load_pickle(os.path.join(save_dir, mouse),
                                   f"{mouse}_si_filters_{signal_name}_{si_neigh}dict.pkl")
        for session, beh_data in si_dict.items():
            for beh_label, filter_data in beh_data.items():
                for filter_id,data in filter_data.items():
                    row = {
                        'area': area,
                        'session': session,
                        'mouse': mouse,
                        'session': session,
                        'filter': int(filter_id),
                        'behavioral_label': beh_label,
                        'si': data['si']
                    }
                    df_rows.append(row)
# Create DataFrame
df = pd.DataFrame(df_rows)
# Optional: sort it for readability
df.sort_values(by=['area','mouse', 'session', 'behavioral_label', 'filter'], inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
# Set up style and palette
palette = {'superficial': 'purple', 'deep': 'gold'}
# Convert filter size to time in seconds (sampling rate = 20 Hz)
sampling_rate = 20  # Hz
df['filter_time'] = df['filter'] / sampling_rate
# Get unique behavioral labels
behavior_labels = df['behavioral_label'].unique()
# Create subplots: one row per behavior
fig, axes = plt.subplots(len(behavior_labels), 1, figsize=(8, 2 * len(behavior_labels)), sharex=True)
# If only one behavior, keep axes as list
if len(behavior_labels) == 1:
    axes = [axes]
# Plot each behavior separately
for i, beh_label in enumerate(behavior_labels):
    ax = axes[i]
    # Filter data for current behavioral label
    beh_df = df[df['behavioral_label'] == beh_label]
    # Lineplot with error bars per area
    sns.lineplot(data=beh_df,
                 x='filter_time', y='si', hue='area',
                 errorbar='sd', ax=ax, palette=palette, marker='o')
    # Labels and formatting
    ax.set_title(f'Structure Index vs Filter Time ({beh_label})')
    ax.set_ylabel('Structure Index (SI)')
    ax.set_xlabel('Filter Size (seconds)' if i == len(behavior_labels) - 1 else '')
    ax.legend(title='Area', loc='upper right')
    ax.set_ylim(0, 1)  # Optional: standardize y axis across all plots
    ax.grid(False)
# Final layout
plt.tight_layout()
plt.show()
# Save figure
fig.savefig(os.path.join(save_dir, f"SI_filter_{signal_name}_time_axis.png"), dpi=400,
            bbox_inches="tight")


df= df[df['behavioral_label'].isin(['pos', 'time'])]
# Get unique behavioral labels
behavior_labels = df['behavioral_label'].unique()
# Create subplots: one row per behavior
fig, axes = plt.subplots(len(behavior_labels), 1, figsize=(8, 2 * len(behavior_labels)), sharex=True)
# If only one behavior, keep axes as list
if len(behavior_labels) == 1:
    axes = [axes]
# Plot each behavior separately
for i, beh_label in enumerate(behavior_labels):
    ax = axes[i]
    # Filter data for current behavioral label
    beh_df = df[df['behavioral_label'] == beh_label]
    # Lineplot with error bars per area
    sns.lineplot(data=beh_df,
                 x='filter_time', y='si', hue='area',
                 errorbar='sd', ax=ax, palette=palette, marker='o')
    # Labels and formatting
    ax.set_title(f'Structure Index vs Filter Time ({beh_label})')
    ax.set_ylabel('Structure Index (SI)')
    ax.set_xlabel('Filter Size (seconds)' if i == len(behavior_labels) - 1 else '')
    ax.legend(title='Area', loc='upper right')
    ax.set_ylim(0, 1)  # Optional: standardize y axis across all plots
    ax.grid(False)
# Final layout
plt.tight_layout()
plt.show()
# Save figure
fig.savefig(os.path.join(save_dir, f"SI_filter_pos_time_{signal_name}_time_axis.png"), dpi=400,
            bbox_inches="tight")


##############################################################33
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os

# Convert filter size to time in seconds (sampling rate = 20 Hz)
sampling_rate = 20
df['filter_time'] = df['filter'] / sampling_rate

# Filter data for filter_time = 0.2 and behavioral labels 'pos' and 'time'
df_plot = df[(df['filter_time'] == 0.2) & (df['behavioral_label'].isin(['pos', 'time']))]

# Plot settings
palette = {'superficial': 'purple', 'deep': 'gold'}
alpha_levels = [(0.001, '***'), (0.01, '**'), (0.05, '*')]

# Create barplot: x = behavior, bars = area
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=df_plot, x='behavioral_label', y='si', hue='area',
                 palette=palette, errorbar='sd', capsize=0.1)

# Significance test (one-sided: deep > superficial) for each behavior
y_max = df_plot['si'].max()
increment = 0.05

for i, beh in enumerate(['pos', 'time']):
    beh_df = df_plot[df_plot['behavioral_label'] == beh]
    si_sup = beh_df[beh_df['area'] == 'superficial']['si']
    si_deep = beh_df[beh_df['area'] == 'deep']['si']

    # Two-sided t-test
    stat, pval_two_sided = ttest_ind(si_deep, si_sup, equal_var=False)

    # Convert to one-sided p-value (H1: deep > superficial)
    if stat > 0:
        pval = pval_two_sided / 2
    else:
        pval = 1.0  # not in direction of interest

    # Determine significance symbol
    signif = ''
    for alpha, symbol in alpha_levels:
        if pval < alpha:
            signif = symbol
            break

    # Add significance to plot
    if signif:
        x1, x2 = i - 0.2, i + 0.2
        y = y_max + (i + 1) * increment
        ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=1.3, color='black')
        ax.text((x1 + x2) / 2, y + 0.015, signif, ha='center', va='bottom', fontsize=12)

# Final formatting
ax.set_title('Structure Index (SI) at Filter Time = 0.2s\n(One-sided test: Deep > Superficial)')
ax.set_ylabel('Structure Index (SI)')
ax.set_xlabel('Behavioral Condition')
ax.set_ylim(0, y_max + 0.2)
ax.legend(title='Area')
plt.tight_layout()
plt.show()

# Save figure
fig_path = os.path.join(save_dir, f"SI_bar_comparison_0.2s_{signal_name}_one_sided.png")
plt.savefig(fig_path, dpi=400, bbox_inches="tight")
print(f"Saved figure to {fig_path}")
