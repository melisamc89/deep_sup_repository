import os
import sys
import copy

import time
import timeit
from datetime import datetime

import numpy as np

import learning_repo.general_utils as lrgu
from structure_index import compute_structure_index
from scipy.ndimage import gaussian_filter1d
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

data_dir = os.path.join(base_dir, 'MIR','Transfer')
save_dir = os.path.join(base_dir, 'SI')
if not os.path.isdir(save_dir): os.makedirs(save_dir)

k = 3
params1 = {
    'n_bins': 10,
    'discrete_label': False,
    'continuity_kernel': None,
    'perc_neigh': 1,
    'num_shuffles': 20,
    'verbose': False
}


params2 = {
    'n_bins': 3,
    'discrete_label': True,
    'continuity_kernel': None,
    'n_neighbors': 50,
    'num_shuffles': 20,
    'verbose': False
}

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

        mi_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_mi_transferred_cluster_{k}_all_{signal_name}_dict.pkl")
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

                noise_idx, signal_idx = filter_noisy_outliers(signal)
                clusters_id = mi_dict[session]['cluster_id']
                clusters_names = np.unique(clusters_id)

                print(f"Working on session {session} ({idx + 1}/{len(session_names)}):",end='')

                for beh_name, beh_val in beh_variables.items():
                    if 'trial_id_mat' in beh_name:
                        si_beh_params[beh_name]['min_label'] = np.min(beh_val)
                        si_beh_params[beh_name]['max_label'] = np.max(beh_val)
                    # si_beh_params[beh_name]['n_neighbors'] = int(signal.shape[0]* si_beh_params[beh_name]['perc_neigh']/100)
                    si_beh_params[beh_name]['n_neighbors'] = 50
                    si_dict[session][beh_name] = dict()

                    si, process_info, overlap_mat, si_shuff = compute_structure_index(signal[signal_idx],
                                                                                   beh_val[signal_idx],
                                                                                   **si_beh_params[beh_name])

                    si_dict[session][beh_name][str(-1)] = {
                        'si': copy.deepcopy(si),
                        'si_shuffled': copy.deepcopy(si_shuff),
                        'process_info': copy.deepcopy(process_info),
                        'overlap_mat': copy.deepcopy(overlap_mat),
                        'beh_params': copy.deepcopy(si_beh_params[beh_name]),
                        'valid_idx': valid_index.copy(),
                        'signal_idx': signal_idx.copy(),
                        'signal': signal.copy()

                    }
                    for cluster_idx in clusters_names:
                        if cluster_idx == -10: continue
                        cluster_signal = signal[:, clusters_id == cluster_idx]
                        si, process_info, overlap_mat, si_shuff = compute_structure_index(cluster_signal[signal_idx],
                                                                                       beh_val[signal_idx],
                                                                                       **si_beh_params[beh_name])

                        si_dict[session][beh_name][str(cluster_idx)] = {
                            'si': copy.deepcopy(si),
                            'si_shuffled': copy.deepcopy(si_shuff),
                            'process_info': copy.deepcopy(process_info),
                            'overlap_mat': copy.deepcopy(overlap_mat),
                            'beh_params': copy.deepcopy(si_beh_params[beh_name]),
                            'valid_idx': valid_index.copy(),
                            'signal_idx': signal_idx.copy(),
                            'signal': cluster_signal.copy()

                        }

                        print(f" {beh_name}_{cluster_idx}={si:.4f} |", end='', sep='', flush='True')
                        print()
                idx += 1

        lrgu.save_pickle(msave_dir, f"{mouse}_si_tranferred_cluster_kmeans_{k}_all_{signal_name}_all_mov_dict_zscored.pkl", si_dict)
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
                                   f"{mouse}_si_tranferred_cluster_kmeans_{k}_all_{signal_name}_all_mov_dict_zscored.pkl")
        for session, beh_data in si_dict.items():
            for beh_label, cluster_data in beh_data.items():
                for cluster_id, si_info in cluster_data.items():
                    #si_shuffled = si_dict[session][beh_label][cluster_id]['si_shuffled']
                    #if si_info['si'] > np.sort(si_shuffled)[int(0.05*len(si_shuffled))]:
                    #    sig = 's'
                    #else:
                    #    sig = 'ns'
                    row = {
                        'area': area,
                        'session': session,
                        'mouse': mouse,
                        'session': session,
                        'cluster': int(cluster_id),
                        'behavioral_label': beh_label,
                        'si': si_info['si'],
                        'significance': 's'
                    }
                    df_rows.append(row)
# Create DataFrame
df = pd.DataFrame(df_rows)
# Optional: sort it for readability
df.sort_values(by=['area','mouse', 'session', 'behavioral_label', 'cluster'], inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Filter for session 4
df_session = df.copy()

# Compute mean and std for session 4
summary_df = df_session.groupby(['cluster', 'behavioral_label'])['si'].agg(['mean', 'std']).reset_index()

# Plot
plt.figure(figsize=(10, 6))

# Raw scatter points
sns.stripplot(data=df_session, x='cluster', y='si', hue='behavioral_label',
              dodge=True, jitter=True, alpha=0.3, linewidth=0.5, palette='Set2')

# Overlay mean ± std as error bars
for i, (beh_label, group) in enumerate(summary_df.groupby('behavioral_label')):
    x = group['cluster'] + (i - len(summary_df['behavioral_label'].unique())/2) * 0.15
    y = group['mean']
    yerr = group['std']
    plt.errorbar(x, y, yerr=yerr, fmt='o', color=sns.color_palette('Set2')[i],
                 label=f"{beh_label} (mean±std)", capsize=4, markersize=5, lw=1.5)

# Decorate plot
plt.title("Structure Index per Cluster")
plt.xlabel("Cluster ID")
plt.ylabel("Structure Index (SI)")
plt.legend(title="Behavioral Label", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

for area in  ['deep','superficial']:
    df_session = df[df['area'] == area]
    #df_session = df.copy()
    # Get all behavioral labels
    behavior_labels = df_session['behavioral_label'].unique()
    # Set color palette
    mouse_palette = sns.color_palette('husl', len(df_session['mouse'].unique()))
    mouse_color_map = dict(zip(sorted(df_session['mouse'].unique()), mouse_palette))
    # Create subplots: one per behavioral label
    n = len(behavior_labels)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3), sharey=True)
    # Loop through behavioral labels
    for i, beh in enumerate(behavior_labels):
        ax = axes[i] if n > 1 else axes  # handle case of a single subplot
        beh_df = df_session[df_session['behavioral_label'] == beh]
        # Plot boxplot (mean and std)
        sns.violinplot(data = beh_df, x='cluster', y='si', ax=ax, color='lightgray', width=0.6)
        # Overlay individual mouse points (no lines)
        #sns.stripplot(data=beh_df, x='cluster', y='si', ax=ax,
        #              hue='mouse', palette=mouse_color_map,
        #              dodge=True, jitter=True, size=6, alpha=0.7, edgecolor='k', linewidth=0.5)
        ax.set_title(f"Behavior: {beh}")
        ax.set_xlabel("Cluster ID")
        if i == 0:
            ax.set_ylabel("Structure Index (SI)")
        else:
            ax.set_ylabel("")
        ax.set_ylim([0, 1])
        ax.grid(False, linestyle='--', alpha=0.3)
        # Only add legend once
        if i == 0:
            ax.legend(title="Mouse", loc='lower right', fontsize=8, frameon=False)
        #else:
            #ax.get_legend().remove()
    # Final formatting
    fig.suptitle(f"Structure Index per Cluster {area} ", fontsize=16)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(save_dir, f"SI_tranferred_cluster_{signal_name}_{area}_{mov_cond}.png"), dpi=400,
                bbox_inches="tight")
####################################################################
#### when all clustered together

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats
k = 3
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
                                   f"{mouse}_si_tranferred_cluster_kmeans_{k}_all_{signal_name}_all_mov_dict_zscored.pkl")
        for session, beh_data in si_dict.items():
            for beh_label, cluster_data in beh_data.items():
                for cluster_id, si_info in cluster_data.items():
                    row = {
                        'area': area,
                        'session': session,
                        'mouse': mouse,
                        'session': session,
                        'cluster': int(cluster_id),
                        'behavioral_label': beh_label,
                        'si': si_info['si']
                    }
                    df_rows.append(row)
# Create DataFrame
df = pd.DataFrame(df_rows)
# Optional: sort it for readability
df.sort_values(by=['area','mouse', 'session', 'behavioral_label', 'cluster'], inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np

# Prepare for stats
df_session = df.copy()
behavior_labels = df_session['behavioral_label'].unique()
clusters = sorted(df_session['cluster'].unique())

# Collect p-values per behavior + cluster
pval_dict = {}
for beh in behavior_labels:
    for clust in clusters:
        sup_vals = df_session[(df_session['behavioral_label'] == beh) &
                              (df_session['cluster'] == clust) &
                              (df_session['area'] == 'superficial')]['si']
        deep_vals = df_session[(df_session['behavioral_label'] == beh) &
                               (df_session['cluster'] == clust) &
                               (df_session['area'] == 'deep')]['si']
        if len(sup_vals) > 0 and len(deep_vals) > 0:
            stat, p = mannwhitneyu(sup_vals, deep_vals, alternative='two-sided')
        else:
            p = 1.0
        pval_dict[(beh, clust)] = p

# Correct p-values
keys = list(pval_dict.keys())
pvals = [pval_dict[k] for k in keys]
#corrected = multipletests(pvals, method='fdr_bh')[1]
corrected = pvals
corrected_pvals = dict(zip(keys, corrected))

# --- Plot ---
n = len(behavior_labels)
fig, axes = plt.subplots(1, n, figsize=(2.5 * len(clusters), 3), sharey=True)

for i, beh in enumerate(behavior_labels):
    ax = axes[i] if n > 1 else axes
    beh_df = df_session[df_session['behavioral_label'] == beh]

    # Plot violins by area
    sns.violinplot(data=beh_df, x='cluster', y='si', hue='area',
                   ax=ax, palette=['#cc9900', '#9900ff'], width=0.8, inner=None, split = True)

    # Set aesthetics
    ax.set_title(f"Behavior: {beh}")
    ax.set_xlabel("Cluster")
    if i == 0:
        ax.set_ylabel("Structure Index (SI)")
    else:
        ax.set_ylabel("")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(False)

    # Add significance stars
    for clust in clusters:
        p_corr = corrected_pvals.get((beh, clust), 1.0)
        if p_corr < 0.001:
            sig = '***'
        elif p_corr < 0.01:
            sig = '**'
        elif p_corr < 0.05:
            sig = '*'
        elif p_corr < 0.1:
            sig = '0.1'
        else:
            sig = '-'

        if sig:
            # Estimate max y for this cluster across areas
            cluster_vals = beh_df[beh_df['cluster'] == clust]['si']
            y_star = 1.2 * cluster_vals.max() + 0.02  # 75% of max, with padding
            ax.text(clust+1, y_star, sig)

    if i == 0:
        ax.legend(title="Area", loc='lower right', frameon=False)
    else:
        ax.get_legend().remove()

# Final touch
fig.suptitle("Structure Index (SI) by Cluster and Area with Significance", fontsize=16, y=1.1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"SI_transferred_cluster_{k}_{signal_name}_depth_comparison_fdr_corrected.png"),
            dpi=400, bbox_inches='tight')
plt.show()
###############################################################################3
################################################################################


#########################################
# PLOTING SI VALUES
#######################################
import matplotlib.pyplot as plt
import numpy as np
# Prepare data
si_time = df[df['behavioral_label'] == 'time']['si'].values
si_pos = df[df['behavioral_label'] == 'pos']['si'].values
area = df[df['behavioral_label'] == 'pos']['area'].values
cluster = df[df['behavioral_label'] == 'pos']['cluster'].values
# Remove unassigned cluster -1
valid_mask = cluster != -1
si_time = si_time[valid_mask]
si_pos = si_pos[valid_mask]
area = area[valid_mask]
cluster = cluster[valid_mask]
# Area color map
area_colors = {'superficial': '#9900ff', 'deep': '#cc9900'}
marker_list = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']
# Plot
fig, ax = plt.subplots(figsize=(6, 4))
unique_clusters = sorted(np.unique(cluster))
for i, clust in enumerate(unique_clusters):
    mask = cluster == clust
    for area_type in ['superficial', 'deep']:
        area_mask = (area == area_type) & mask
        ax.scatter(
            si_time[area_mask],
            si_pos[area_mask],
            c=area_colors[area_type],
            marker=marker_list[i % len(marker_list)],
            s=60,
            label=f'Cluster {clust} ({area_type})',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
ax.set_xlabel('SI Time')
ax.set_ylabel('SI Pos')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(False)
ax.set_title('SI Time vs Pos (Colored by Area, Marker = Cluster)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Cluster/Area')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"SI_transferred_cluster_{k}_{signal_name}_pos_vs_time.png"),
            dpi=400, bbox_inches='tight')
plt.savefig(os.path.join(save_dir, f"SI_transferred_cluster_{k}_{signal_name}_pos_vs_time.svg"),
            dpi=400, bbox_inches='tight')
plt.show()
###########################################################################

import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare data
si_time = df[df['behavioral_label'] == 'time']['si'].values
si_pos = df[df['behavioral_label'] == 'pos']['si'].values
area = df[df['behavioral_label'] == 'pos']['area'].values
cluster = df[df['behavioral_label'] == 'pos']['cluster'].values

# Remove unassigned cluster -1
valid_mask = cluster != -1
si_time = si_time[valid_mask]
si_pos = si_pos[valid_mask]
area = area[valid_mask]
cluster = cluster[valid_mask]

# Color maps
cluster_colors = {0: '#bce784', 1: '#66cef4', 2: '#ec8ef8'}
area_edge_colors = {'superficial': '#9900ff', 'deep': '#cc9900'}

marker_list = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
unique_clusters = sorted(np.unique(cluster))

for i, clust in enumerate(unique_clusters):
    mask = cluster == clust
    for area_type in ['superficial', 'deep']:
        area_mask = (area == area_type) & mask
        ax.scatter(
            si_time[area_mask],
            si_pos[area_mask],
            c=cluster_colors.get(clust, 'gray'),
            edgecolor=area_edge_colors[area_type],
            marker=marker_list[i % len(marker_list)],
            s=60,
            label=f'Cluster {clust} ({area_type})',
            linewidth=0.8,
            alpha=0.9
        )

ax.set_xlabel('SI Time')
ax.set_ylabel('SI Pos')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(False)
ax.set_title('SI Time vs Pos (Cluster Color, Area Border)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Cluster/Area')
plt.tight_layout()

# Save plots
plt.savefig(os.path.join(save_dir, f"SI_transferred_cluster_{k}_{signal_name}_pos_vs_time.png"),
            dpi=400, bbox_inches='tight')
plt.savefig(os.path.join(save_dir, f"SI_transferred_cluster_{k}_{signal_name}_pos_vs_time.svg"),
            dpi=400, bbox_inches='tight')

plt.show()

###############################################################################3
################################################################################



all_neurons_df = df[df['cluster' == '-1']]
time_values = all_neurons_df[all_neurons_df['behavioral_label'] == 'time']
### max_si_time_deep =  GC2 (0.61)
### max_si_time_sup  =  CZ3 (0.53)


##### COUTING
### mising piece of code...

# Step 1: Neuron counts per mouse and cluster (excluding cluster -1)
cluster_df = df[df['cluster'] != -1]

# Step 2: Count neurons per mouse/area/cluster
mouse_counts = cluster_df.groupby(['area', 'mouse', 'cluster']).size().reset_index(name='count')

# Step 3: Get total neurons per mouse from cluster -1
mouse_totals = df[df['cluster'] == -1].groupby(['area', 'mouse']).size().reset_index(name='total_neurons')

# Step 4: Merge and normalize
mouse_counts = pd.merge(mouse_counts, mouse_totals, on=['area', 'mouse'])
mouse_counts['normalized'] = mouse_counts['count'] / mouse_counts['total_neurons']

# Step 5: Aggregate by area and cluster
area_means = mouse_counts.groupby(['area', 'cluster'])['normalized'].mean().reset_index()
area_sems = mouse_counts.groupby(['area', 'cluster'])['normalized'].sem().reset_index()
area_stats = pd.merge(area_means, area_sems, on=['area', 'cluster'], suffixes=('_mean', '_sem'))

# --- Plotting ---
sns.set(style="whitegrid")
palette = {'superficial': '#9900ff', 'deep': '#cc9900'}

plt.figure(figsize=(10, 6))

# Bar plot with error bars (no ci, using sem if desired)
sns.barplot(data=area_stats, x='cluster', y='normalized_mean', hue='area',
            palette=palette, ci=None, edgecolor='k')

# Overlay individual points per mouse
for i, cluster in enumerate(sorted(cluster_df['cluster'].unique())):
    for area in ['superficial', 'deep']:
        dots = mouse_counts[(mouse_counts['area'] == area) & (mouse_counts['cluster'] == cluster)]
        x_pos = i - 0.2 if area == 'superficial' else i + 0.2
        plt.scatter([x_pos] * len(dots), dots['normalized'], color='black', alpha=0.8, zorder=10)

# Labels and layout
plt.title('Normalized Neuron Counts per Cluster (Excluding cluster -1)')
plt.ylabel('Fraction of Neurons in Cluster')
plt.xlabel('Cluster ID')
plt.xticks(sorted(cluster_df['cluster'].unique()))
plt.legend(title='Area', loc='upper right')
plt.tight_layout()
plt.show()


