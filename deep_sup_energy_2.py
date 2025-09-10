import os
import sys
import copy
import timeit
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.signal import argrelextrema
import learning_repo.general_utils as lrgu
from scipy.stats import pearsonr
import pickle
import matplotlib.pyplot as plt

# Parameters
percentage = 10
percentage_text = str(percentage)
minima_values = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 300]
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

# Directories
base_dir = '/home/melma31/Documents/deepsup_project/'
data_dir = os.path.join(base_dir, 'data')
save_dir = os.path.join(base_dir, 'EnergyFilter')
os.makedirs(save_dir, exist_ok=True)

# Mouse groups
coloring_condition = ['single_color']
general_mice_list = {
    'lt_and_rot': {
        'Calb': ['CGrin1', 'CZ3', 'CZ6', 'CZ8', 'CZ9'],
        'Chrna7': ['ChZ4'],
        'Thy1': ['GC2', 'GC3', 'GC5_nvista', 'TGrin1']
    },
    'only_lt': {
        'Calb': ['CZ4'],
        'Chrna7': ['ChZ7', 'ChZ8'],
        'Thy1': ['GC7']
    }
}

signal_name = 'clean_traces'
models = ['Calb', 'Chrna7', 'Thy1']
R_summary = {}

# Energy computation functions
def compute_hopfield_energy(X, pattern_indices, eval_indices):
    patterns = X[pattern_indices]
    W = np.sum([np.outer(p, p) for p in patterns], axis=0)
    np.fill_diagonal(W, 0)
    energy = -0.5 * np.einsum('ij,ij->i', X[eval_indices] @ W, X[eval_indices])
    return energy, eval_indices

def get_energy_segments(signal):
    X = signal.copy()
    T = X.shape[0]
    ten_percent = int(percentage / 100 * T)
    segments = {
        'First_' + percentage_text: {
            'train': np.arange(0, ten_percent),
            'eval': np.arange(ten_percent, T)
        },
        #'Last_' + percentage_text: {
        #    'train': np.arange(T - ten_percent, T),
        #    'eval': np.arange(0, T - ten_percent)
        #}
    }
    energies = {}
    for label, idxs in segments.items():
        energy, eval_idx = compute_hopfield_energy(X, pattern_indices=idxs['train'],
                                                   eval_indices=idxs['eval'])
        energies[label] = {'energy': energy, 'eval_idx': eval_idx}
    return energies

# Main loop
for coloring in coloring_condition:
    paradigm_list = ['lt_and_rot', 'only_lt'] if coloring == 'single_color' else ['only_lt']
    for paradigm_ in paradigm_list:
        mice_list = general_mice_list[paradigm_]
        for model in models:
            for mouse in mice_list.get(model, []):
                print(mouse)
                global_time = timeit.default_timer()
                mdata_dir = os.path.join(data_dir, coloring, paradigm_, 'processed_data', model, mouse)
                msave_dir = os.path.join(save_dir, mouse)
                os.makedirs(msave_dir, exist_ok=True)

                f = open(os.path.join(msave_dir, f"{mouse}_linearity_{signal_name}_logFile.txt"), 'w')
                original_stdout = sys.stdout
                sys.stdout = lrgu.Tee(sys.stdout, f)

                print(f"\t#### {mouse}: | {signal_name} ####")
                print(f'\t{datetime.now():%Y-%m-%d %H:%M}\n')

                file_name = mouse + '_df_dict.pkl'
                mouse_dict = lrgu.load_pickle(mdata_dir, file_name)
                fnames = list(mouse_dict.keys())
                if len(fnames) > 2:
                    mouse_dict = {mouse + '_lt': copy.deepcopy(mouse_dict)}
                    maze_state = [mouse + '_lt']
                else:
                    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
                    maze_state = [fname_pre]

                for maze in maze_state:
                    print(f"\t#### {mouse}: | {maze} ####\n")
                    session_pd = copy.deepcopy(mouse_dict[maze])
                    params = {'Fs': 'sf', 'pos': 'position', 'vel': 'speed'}
                    for old, new in params.items():
                        if old in session_pd.columns:
                            session_pd.rename(columns={old: new}, inplace=True)

                    session_pd = lrgu.add_mov_direction_mat_field(session_pd)
                    session_pd = lrgu.add_inner_trial_time_field(session_pd)
                    session_pd = lrgu.add_trial_id_mat_field(session_pd)

                    mov_dir = lrgu.get_signal(session_pd, 'mov_direction').copy()
                    pos = lrgu.get_signal(session_pd, 'position')
                    speed = lrgu.get_signal(session_pd, 'speed')

                    temp_mov = np.diff(pos, axis=0) * 20
                    temp_mov = np.concatenate([temp_mov[0, :].reshape(-1, 2), temp_mov], axis=0)[:, 0]
                    temp_mov = gaussian_filter1d(temp_mov, sigma=5)
                    temp_mov[temp_mov < 0] = -1
                    temp_mov = np.sign(temp_mov)
                    mov_dir[mov_dir == 0] = temp_mov[np.where(mov_dir == 0)[0]]
                    valid_index = np.arange(0, mov_dir.shape[0])

                    filter_size = 16
                    signal = lrgu.get_signal(session_pd, signal_name)
                    signal = smooth_calcium_signals(signal, filter_size)

                    energies = get_energy_segments(signal)

                    for top_k in minima_values:
                        R_summary.setdefault(top_k, {}).setdefault(model, {}).setdefault(mouse, {})
                        for label in energies:
                            energy = energies[label]['energy']
                            x = energies[label]['eval_idx']
                            x_rel = np.arange(len(energy))
                            minima_idx = argrelextrema(energy, np.less, order=1)[0]
                            minima_values_array = energy[minima_idx]
                            if len(minima_idx) > top_k:
                                top_indices = np.argsort(minima_values_array)[:top_k]
                                minima_idx = minima_idx[top_indices]
                                minima_values_array = minima_values_array[top_indices]
                            minima_x = x_rel[minima_idx]
                            minima_y = minima_values_array
                            if len(minima_x) > 1:
                                slope, intercept, r_value, p_value, std_err = linregress(minima_x, minima_y)
                                r_squared = r_value ** 2  # Calculate R²
                            else:
                                r_value = np.nan
                                r_squared = np.nan  # No R² value when there's not enough data
                            R_summary[top_k][model][mouse][label] = {'r_value': r_value, 'r_squared': r_squared, 'slope': slope}

                sys.stdout = original_stdout
                f.close()

# Optionally save R_summary
with open(os.path.join(save_dir, 'R_summary_all_minima.pkl'), 'wb') as f:
    pickle.dump(R_summary, f)

print("Analysis complete.")

# Plotting
# Custom color palette
custom_palette = {
    'superficial': '#9900ff',  # RGB: 153, 0, 255
    'deep': '#cc9900',         # RGB: 204, 153, 0
}

r_summary_path = '/home/melma31/Documents/deepsup_project/EnergyFilter/R_summary_all_minima.pkl'
mice_dict = {
    'superficial': ['CGrin1', 'CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9'],
    'deep': ['ChZ4', 'ChZ8', 'ChZ7', 'GC2', 'GC3', 'GC7', 'GC5_nvista', 'TGrin1']
}

if os.path.exists(r_summary_path):
    with open(r_summary_path, 'rb') as f:
        R_summary = pickle.load(f)

    minima_values = sorted(R_summary.keys())
    segment_labels = ['First_10']#, 'Last_10']
    depth_types = ['superficial', 'deep']

    summary_stats = {seg: {depth: {'mean': [], 'std': []} for depth in depth_types} for seg in segment_labels}

    for top_k in minima_values:
        for segment in segment_labels:
            for depth in depth_types:
                r_vals = []
                for model_dict in R_summary[top_k].values():
                    for mouse, segments in model_dict.items():
                        if mouse in mice_dict[depth] and segment in segments:
                            r = segments[segment]['slope']
                            if r is not None and not np.isnan(r):
                                r_vals.append(r)
                if r_vals:
                    summary_stats[segment][depth]['mean'].append(np.mean(r_vals))
                    summary_stats[segment][depth]['std'].append(np.std(r_vals))
                else:
                    summary_stats[segment][depth]['mean'].append(np.nan)
                    summary_stats[segment][depth]['std'].append(np.nan)

    # Plot using custom colors
    for segment in segment_labels:
        plt.figure(figsize=(10, 6))
        for depth in depth_types:
            plt.errorbar(
                minima_values,
                summary_stats[segment][depth]['mean'],
                yerr=summary_stats[segment][depth]['std'],
                fmt='-o',
                capsize=5,
                label=f'{depth.capitalize()}',
                color=custom_palette[depth]
            )
        plt.xlabel('Number of Minima (top_k)')
        plt.ylabel('Average R-value')
        plt.title(f'R-value vs. Number of Minima — {segment}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'enery_number_of_minima_stability_first.png'),
                    dpi=400, bbox_inches="tight")
        plt.savefig(os.path.join(save_dir, f'enery_number_of_minima_stability_first.svg'),
                    dpi=400, bbox_inches="tight")
        plt.show()
else:
    print("❌ File not found. Please ensure the .pkl file exists at the specified path.")

# Plotting R-value and R-squared using custom colors

for segment in segment_labels:
    # Plot for R-value
    plt.figure(figsize=(10, 6))
    for depth in depth_types:
        plt.errorbar(
            minima_values,
            summary_stats[segment][depth]['mean'],  # Mean R-value
            yerr=summary_stats[segment][depth]['std'],
            fmt='-o',
            capsize=5,
            label=f'{depth.capitalize()}',
            color=custom_palette[depth]
        )
    plt.xlabel('Number of Minima (top_k)')
    plt.ylabel('Average Slope-value')
    plt.title(f'Slope-value vs. Number of Minima — {segment}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'enery_number_of_minima_stability_slope_value.png'),
                dpi=400, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f'enery_number_of_minima_stability_slope_value.svg'),
                dpi=400, bbox_inches="tight")
    plt.show()

    # Plot for R-squared (R²)
    plt.figure(figsize=(10, 6))
    for depth in depth_types:
        plt.errorbar(
            minima_values,
            summary_stats[segment][depth]['mean'],  # Mean R² value
            yerr=summary_stats[segment][depth]['std'],
            fmt='-o',
            capsize=5,
            label=f'{depth.capitalize()}',
            color=custom_palette[depth]
        )
    plt.xlabel('Number of Minima (top_k)')
    plt.ylabel('Average R-squared (R²)')
    plt.title(f'R-squared (R²) vs. Number of Minima — {segment}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'enery_number_of_minima_stability_r_squared.png'),
                dpi=400, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f'enery_number_of_minima_stability_r_squared.svg'),
                dpi=400, bbox_inches="tight")
    plt.show()


###################load SI #############################################
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load SI data
data_dir = os.path.join(base_dir, 'results')
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'

mouse_name_list = []
si_time_list = []
si_umap_time_list = []
area_list = []
session_list = []

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = data_dir
        si_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_si_{signal_name}_dict.pkl")
        session_names = list(si_dict.keys())
        session_names.sort()
        for idx, session in enumerate(session_names):
            area_list.append(area)
            session_type = 'lt' if 'lt' in session else 'rot' if 'rot' in session else 'unknown'
            mouse_name_list.append(mouse)
            session_list.append(session_type)
            si_time_list.append(si_dict[session]['time']['si'])
            si_umap_time_list.append(si_dict[session]['time']['si_umap'])

si_pd = pd.DataFrame(data={'mouse': mouse_name_list,
                           'area': area_list,
                           'session_type': session_list,
                           'si_time': si_time_list,
                           'si_time_umap': si_umap_time_list})

# Custom color palette
custom_palette = {
    'superficial': '#9900ff',  # purple
    'deep': '#cc9900',         # gold
}


# Re-run correlation and plotting split by depth
if os.path.exists(r_summary_path):
    with open(r_summary_path, 'rb') as f:
        R_summary = pickle.load(f)

    # Mice list definition for each depth (superficial and deep)
    mice_dict = {
        'superficial': ['CGrin1', 'CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9'],
        'deep': ['ChZ4', 'ChZ8', 'ChZ7', 'GC2', 'GC3', 'GC7', 'GC5_nvista', 'TGrin1']
    }

    segment_labels = ['First_10', 'Last_10']
    minima_values = sorted(R_summary.keys())
    correlation_by_depth = {
        'slope': {seg: {depth: [] for depth in mice_dict} for seg in segment_labels},
        'r_squared': {seg: {depth: [] for depth in mice_dict} for seg in segment_labels}
    }

    # Compute correlation between SI and R-value and R-squared
    for seg in segment_labels:
        for top_k in minima_values:
            for depth, mice in mice_dict.items():
                si_vals = []
                r_vals = []
                r_squared_vals = []

                for model_dict in R_summary[top_k].values():
                    for mouse, segments in model_dict.items():
                        if mouse in mice and seg in segments:
                            r = segments[seg]['slope']
                            r_squared = segments[seg]['r_squared']
                            si_match = si_pd[si_pd['mouse'] == mouse]
                            if not si_match.empty:
                                si_mean = si_match['si_time'].mean()
                                # Collecting values
                                si_vals.append(si_mean)
                                r_vals.append(r)
                                r_squared_vals.append(r_squared)

                if len(si_vals) > 1:
                    # Compute Pearson correlation for R-value
                    corr_r, _ = pearsonr(si_vals, r_vals) if len(r_vals) > 1 else (np.nan, np.nan)
                    # Compute Pearson correlation for R-squared
                    corr_r_squared, _ = pearsonr(si_vals, r_squared_vals) if len(r_squared_vals) > 1 else (np.nan, np.nan)
                else:
                    corr_r = corr_r_squared = np.nan

                # Store the results for both R-value and R-squared
                correlation_by_depth['slope'][seg][depth].append(corr_r)
                correlation_by_depth['r_squared'][seg][depth].append(corr_r_squared)

    # Plotting one figure per segment
    for seg in segment_labels:
        plt.figure(figsize=(10, 6))
        for depth in mice_dict:
            plt.plot(
                minima_values,
                correlation_by_depth['slope'][seg][depth],
                '-o',
                label=f'Slope - {depth.capitalize()}',
                color=custom_palette[depth]
            )
        plt.xlabel('Number of Minima (top_k)')
        plt.ylabel('Correlation Slope (SI vs Energy R-value)')
        plt.title(f'SI vs Energy Slope Correlation — {seg}')
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'correlation_si_vs_energy_slope_value_{seg}.png'), dpi=400, bbox_inches="tight")
        plt.savefig(os.path.join(save_dir, f'correlation_si_vs_energy_slope_value_{seg}.svg'), dpi=400, bbox_inches="tight")
        plt.show()

        # Plotting R-squared (R²) correlation
        plt.figure(figsize=(10, 6))
        for depth in mice_dict:
            plt.plot(
                minima_values,
                correlation_by_depth['r_squared'][seg][depth],
                '-o',
                label=f'R-squared - {depth.capitalize()}',
                color=custom_palette[depth]
            )
        plt.xlabel('Number of Minima (top_k)')
        plt.ylabel('Correlation R² (SI vs Energy R²)')
        plt.title(f'SI vs Energy R-squared Correlation — {seg}')
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'correlation_si_vs_energy_r_squared_{seg}.png'), dpi=400, bbox_inches="tight")
        plt.savefig(os.path.join(save_dir, f'correlation_si_vs_energy_r_squared_{seg}.svg'), dpi=400, bbox_inches="tight")
        plt.show()

else:
    print("❌ R_summary file not found. Make sure it exists at the given path.")


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Configuration
selected_minima = [30,40,50]
segment = 'First_10'

custom_palette = {
    'superficial': '#9900ff',  # purple
    'deep': '#cc9900',         # gold
}

mice_dict = {
    'superficial': ['CGrin1', 'CZ3', 'CZ4', 'CZ6', 'CZ8', 'CZ9'],
    'deep': ['ChZ4', 'ChZ8', 'ChZ7', 'GC2', 'GC3', 'GC7', 'GC5_nvista', 'TGrin1']
}

# Plot
fig, axs = plt.subplots(2, 3, figsize=(9, 8))  # Adjusted number of subplots
for i, top_k in enumerate(selected_minima):
    ax = axs[0,i]
    ax_2 = axs[1,i]

    depth_data = {'superficial': {'si': [], 'slope': [], 'r_squared': []}, 'deep': {'si': [], 'slope': [], 'r_squared': []}}

    for depth, mice in mice_dict.items():
        for model_dict in R_summary[top_k].values():
            for mouse, segments in model_dict.items():
                if mouse in mice and segment in segments:
                    r_val = segments[segment]['slope']
                    r_squared = segments[segment]['r_squared']
                    si_match = si_pd[si_pd['mouse'] == mouse]
                    if not si_match.empty:
                        si_val = si_match['si_time'].mean()
                        depth_data[depth]['si'].append(si_val)
                        depth_data[depth]['slope'].append(r_val)
                        depth_data[depth]['r_squared'].append(r_squared)

    for depth in ['superficial', 'deep']:
        si_vals = np.array(depth_data[depth]['si'])
        r_vals = np.array(depth_data[depth]['slope'])
        r_squared_vals = np.array(depth_data[depth]['r_squared'])
        color = custom_palette[depth]

        if len(si_vals) > 1:
            ax.scatter(si_vals, r_vals, color=color, edgecolor='k', s=50, alpha=0.7, label=f'{depth.capitalize()} R')
            ax_2.scatter(si_vals, r_squared_vals, color=color, marker='x', edgecolor='k', s=50, alpha=0.7, label=f'{depth.capitalize()} R²')

            # Linear regression for R-value
            slope, intercept, r_val, _, _ = linregress(si_vals, r_vals)
            x_line = np.linspace(min(si_vals), max(si_vals), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linestyle='--', label=f'{depth} R={r_val:.2f}')

            #Linear regression for R-squared
            slope_r2, intercept_r2, r_squared_val, _, _ = linregress(si_vals, r_squared_vals)
            y_line_r2 = slope_r2 * x_line + intercept_r2
            ax_2.plot(x_line, y_line_r2, color=color, linestyle=':', label=f'{depth} R²={r_squared_val:.2f}')

    ax.set_title(f'top_k = {top_k}')
    ax.set_xlabel('SI (Time)')
    if i == 0:
        ax.set_ylabel('Energy Slope')
    ax.legend(fontsize='small')

plt.suptitle(f'Scatter: SI vs Energy Slope & R² ({segment}) — Separate Regression by Depth', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(save_dir, 'regression2.png'))
plt.savefig(os.path.join(save_dir, 'regression2.svg'))
plt.show()



import pandas as pd

# Collect all data
all_regression_data = []

for top_k in selected_minima:
    depth_data = {'superficial': {'si': [], 'slope': [], 'r_squared': []}, 'deep': {'si': [], 'slope': [], 'r_squared': []}}

    for depth, mice in mice_dict.items():
        for model_dict in R_summary[top_k].values():
            for mouse, segments in model_dict.items():
                if mouse in mice and segment in segments:
                    r_val = segments[segment]['slope']
                    r_squared = segments[segment]['r_squared']
                    si_match = si_pd[si_pd['mouse'] == mouse]
                    if not si_match.empty:
                        si_val = si_match['si_time'].mean()
                        depth_data[depth]['si'].append(si_val)
                        depth_data[depth]['slope'].append(r_val)
                        depth_data[depth]['r_squared'].append(r_squared)

                        # Add to long-format list
                        all_regression_data.append({
                            'Mouse': mouse,
                            'Depth': depth,
                            'TopK': top_k,
                            'SI': si_val,
                            'Slope': r_val,
                            'R2': r_squared
                        })

# Create DataFrame
df_regression = pd.DataFrame(all_regression_data)

import seaborn as sns
import os

from scipy.stats import ttest_ind

def add_p_values(ax, df, metric, y_offset=0.05):
    """Add p-values between superficial and deep per TopK group."""
    topks = sorted(df['TopK'].unique())
    for i, top_k in enumerate(topks):
        df_k = df[df['TopK'] == top_k]
        deep_vals = df_k[df_k['Depth'] == 'deep'][metric]
        sup_vals = df_k[df_k['Depth'] == 'superficial'][metric]
        if len(deep_vals) > 1 and len(sup_vals) > 1:
            tstat, pval = ttest_ind(sup_vals, deep_vals, equal_var=False)
            p_text = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"

            # X positions for annotation
            x1, x2 = i - 0.2, i + 0.2
            y = max(df_k[metric]) + y_offset
            ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], lw=1.5, c='k')
            ax.text((x1 + x2)/2, y + y_offset * 1.3, p_text, ha='center', va='bottom', fontsize=9)

plt.figure(figsize=(8, 5))
ax = sns.boxplot(data=df_regression, x='TopK', y='Slope', hue='Depth', palette=custom_palette)
plt.title('Slope vs TopK by Depth')
plt.ylabel('Energy Slope')
add_p_values(ax, df_regression, metric='Slope', y_offset=0.05)
plt.tight_layout()
plt.show()
plt.savefig(save_dir + '/slope_box_by_depth.png')
plt.close()

plt.figure(figsize=(8, 5))
ax = sns.boxplot(data=df_regression, x='TopK', y='R2', hue='Depth', palette=custom_palette)
plt.title('R² vs TopK by Depth')
plt.ylabel('Energy R²')
add_p_values(ax, df_regression, metric='R2', y_offset=0.05)
plt.tight_layout()
plt.show()
plt.savefig(save_dir + '/r2_box_by_depth.png')
plt.close()
