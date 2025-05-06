import os
import sys
import copy
import umap

import time
import timeit
from datetime import datetime

import numpy as np

import learning_repo.general_utils as lrgu
from structure_index import compute_structure_index
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import mutual_info_regression

base_dir =  '/home/melma31/Documents/deepsup_project/'
#__________________________________________________________________________
#|                                                                        |#
#|                           Mutual Information (MIR)                     |#
#|________________________________________________________________________|#
data_dir = os.path.join(base_dir, 'data')
save_dir = os.path.join(base_dir, 'MIR')
if not os.path.isdir(save_dir): os.makedirs(save_dir)
coloring_condition = ['single_color']#,'dual_color']
general_mice_list = {'lt_and_rot':{'Calb': ['CGrin1','CZ3','CZ6','CZ8','CZ9'],
              'Chrna7': ['ChZ4'],
              'Thy1':['GC2','GC3','GC5_nvista','TGrin1']},
              'only_lt':{
                  'Calb':['CZ4'],
                  'Chrna7': ['ChZ7','ChZ8'],
                   'Thy1':['GC7']}
              }
signal_name = 'clean_traces'
models = ['Calb', 'Chrna7', 'Thy1']

for coloring in coloring_condition:
    if coloring == 'single_color':
        paradigm =['lt_and_rot','only_lt']
    else:
        paradigm =['only_lt']
    for paradigm_ in paradigm:
        mice_list = general_mice_list[paradigm_]
        for model in models:
            mice_model_list = general_mice_list[paradigm_][model]
            for mouse in mice_model_list:
                print(mouse)
                global_time = timeit.default_timer()     #initialize time
                local_time = timeit.default_timer()

                mdata_dir = os.path.join(data_dir,coloring,paradigm_,'processed_data',model ,mouse) #mouse data dir
                msave_dir = os.path.join(save_dir, mouse) #mouse save dir
                if not os.path.isdir(msave_dir): os.makedirs(msave_dir)
                print(mdata_dir)

                f = open(os.path.join(msave_dir, f"{mouse}_linearity_{signal_name}_logFile.txt"), 'w')
                original = sys.stdout
                sys.stdout = lrgu.Tee(sys.stdout, f)

                print(f"\t#### {mouse}: | {signal_name} ####")
                print(f'\t{datetime.now():%Y-%m-%d %H:%M}\n')

                mi_dict = dict()

                file_name =  mouse+'_df_dict.pkl'
                mouse_dict = lrgu.load_pickle(mdata_dir,file_name)
                fnames = list(mouse_dict.keys())
                if len(fnames) >2 :
                    mouse_copy = copy.deepcopy(mouse_dict)
                    mouse_dict = {}
                    mouse_dict[mouse +'_lt'] = copy.deepcopy(mouse_copy)
                    maze_state = [mouse+ '_lt']
                else:
                    fname_pre = [fname for fname in fnames if 'lt' in fname][0]
                    fname_rot = [fname for fname in fnames if 'rot' in fname][0]
                    maze_state = [fname_pre,fname_rot]
                    maze_state = [fname_pre]

                for idx, maze in enumerate(maze_state):
                    print(f"\t#### {mouse}: | {maze} ####")
                    print()
                    mi_dict[maze] = dict()
                    mice_maze = mouse_dict[maze]
                    params = {'columns_to_rename': {'Fs': 'sf', 'pos': 'position', 'vel': 'speed'}}
                    session_pd = copy.deepcopy(mice_maze)
                    for old, new in params['columns_to_rename'].items():
                        if old in session_pd.columns: session_pd.rename(columns={old: new}, inplace=True)
                    session_pd = lrgu.add_mov_direction_mat_field(session_pd)
                    session_pd = lrgu.add_inner_trial_time_field(session_pd)
                    session_pd = lrgu.add_trial_id_mat_field(session_pd)

                    mov_dir = lrgu.get_signal(session_pd, 'mov_direction').copy()
                    pos = lrgu.get_signal(session_pd, 'position')
                    speed = lrgu.get_signal(session_pd, 'speed')

                    # clean mov dir to avoid gaps in ring
                    temp_mov = np.diff(pos, axis=0) * 20
                    temp_mov = np.concatenate([temp_mov[0, :].reshape(-1, 2), temp_mov], axis=0)[:, 0]
                    temp_mov = gaussian_filter1d(temp_mov, sigma=5, axis=0)
                    temp_mov[temp_mov < 0] = -1
                    temp_mov = np.sign(temp_mov)
                    mov_dir[mov_dir == 0] = temp_mov[np.where(mov_dir == 0)[0]]
                    valid_index = mov_dir != 0
                    valid_index = np.arange(0,mov_dir.shape[0])

                    pos_dir = pos[:, 0] * mov_dir
                    inner_time = lrgu.get_signal(session_pd, 'inner_trial_time')
                    trial_id = lrgu.get_signal(session_pd, 'trial_id_mat')
                    time = np.arange(0,pos.shape[0])

                    signal = lrgu.get_signal(session_pd, signal_name)
                    #noise_idx, signal_idx = filter_noisy_outliers(signal)

                    behaviours_list = [pos[:,0], pos_dir, mov_dir, speed, time, inner_time, trial_id]
                    beh_names = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID']
                    behaviour_dict = {
                        'position':behaviours_list[0],
                        '(pos,dir)':behaviours_list[1],
                        'mov_dir':behaviours_list[2],
                        'speed':behaviours_list[3],
                        'time':behaviours_list[4],
                        'inner_time':behaviours_list[5],
                        'trial_id':behaviours_list[6]
                    }

                    mi_all = []
                    for beh_index, beh in enumerate(behaviours_list):
                        print('MI for variable:' + beh_names[beh_index])
                        mi = []
                        for neui in range(signal.shape[1]):
                            neuron_mi = \
                                mutual_info_regression(signal[valid_index, neui].reshape(-1, 1), beh[valid_index], n_neighbors=50,
                                                       random_state=16)[0]
                            mi.append(neuron_mi)
                        # mi_stack = np.vstack([mi,mi2,mi3]).T
                        mi_all.append(mi)
                    # mi_final = np.hstack(mi_all)
                    mi_dict[maze]['behaviour'] = behaviour_dict
                    mi_dict[maze]['signal'] = signal
                    mi_dict[maze]['valid_index'] = valid_index
                    mi_dict[maze]['MIR'] = mi_all

                lrgu.save_pickle(msave_dir,f"{mouse}_mi_{signal_name}_dict_alldir.pkl", mi_dict)
                lrgu.print_time_verbose(local_time, global_time)
                sys.stdout = original
                f.close()


################################################################################################
#
#                           PLOTTING MIR                                                        #
#
################################################################################################


import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import zscore
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


base_dir =  '/home/melma31/Documents/deepsup_project/'
#__________________________________________________________________________
#|                                                                        |#
#|                           Mutual Information (MIR)                     |#
#|________________________________________________________________________|#
data_dir = os.path.join(base_dir, 'MIR')
save_dir = os.path.join(base_dir, 'MIR')
if not os.path.isdir(save_dir): os.makedirs(save_dir)
topvalues = 10

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9',
                             'CalbEphys1GRIN1', 'CalbEphys1GRIN2'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1',
                    'Thy1Ephys1GRIN1', 'Thy1Ephys1GRIN2']
            }

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'
# Initialize lists
mouse_name_list, area_list, session_list = [], [], []
raw_mi_values = {key: [] for key in ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']}
z_mi_values = {f'z_{key}': [] for key in raw_mi_values}

# Collect MI data
for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = os.path.join(data_dir, mouse)
        mi_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_mi_{signal_name}_dict_alldir.pkl")
        for session in sorted(mi_dict):
            session_type = 'lt' if 'lt' in session else 'rot' if 'rot' in session else None
            data = np.array(mi_dict[session]['MIR'])
            data_z = zscore(data, axis=1)

            for neuron in range(data.shape[1]):
                mouse_name_list.append(mouse)
                area_list.append(area)
                session_list.append(session_type)
                for i, key in enumerate(raw_mi_values):
                    raw_mi_values[key].append(data[i][neuron])
                    z_mi_values[f'z_{key}'].append(data_z[i][neuron])

# Create DataFrame
mi_pd = pd.DataFrame({
    'mouse': mouse_name_list,
    'area': area_list,
    'session_type': session_list,
    **raw_mi_values,
    **z_mi_values
})
# 1. Sum of raw MI values (across all behavioral labels)
behavior_keys = list(raw_mi_values.keys())  # ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']
mi_pd['total_MI'] = mi_pd[behavior_keys].sum(axis=1)

# 2. Mark top 20% of cells
#threshold = np.percentile(mi_pd['total_MI'], topvalues)
#mi_pd['topcells'] = mi_pd['total_MI'].apply(lambda x: 'yes' if x >= threshold else 'no')
threshold = 0.5
mi_pd['topcells'] = mi_pd['total_MI'].apply(lambda x: 'yes' if x >= threshold else 'no')


mi_pd_lt = mi_pd[mi_pd['session_type'] == 'lt']
mi_pd_lt = mi_pd_lt[mi_pd_lt['topcells'] == 'yes']
palette = ['purple', 'yellow']  # Define your own list of colors
# Features to plot


raw_features = ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']
#raw_features = ['z_pos', 'z_posdir', 'z_dir', 'z_speed', 'z_time', 'z_inner_trial_time', 'z_trial_id']

palette = ['purple', 'gold']

# Create subplots
fig, axes = plt.subplots(1, len(raw_features), figsize=(3 * len(raw_features), 3))
if len(raw_features) == 1:
    axes = [axes]
# Collect p-values
pvals = []
for feature in raw_features:
    sup_vals = mi_pd_lt[mi_pd_lt['area'] == 'superficial'][feature]
    deep_vals = mi_pd_lt[mi_pd_lt['area'] == 'deep'][feature]

    if len(sup_vals) > 0 and len(deep_vals) > 0:
        stat, p = mannwhitneyu(sup_vals, deep_vals, alternative='two-sided')
    else:
        p = 1.0
    pvals.append(p)

# FDR correction
corrected_pvals = multipletests(pvals, method='fdr_bh')[1]

# Plot and annotate
for i, (feature, p_corr) in enumerate(zip(raw_features, corrected_pvals)):
    ax = axes[i]
    sns.violinplot(data=mi_pd_lt, x='area', y=feature, palette=palette, ax=ax, cut=0)

    # Y limits
    ymax = mi_pd_lt[feature].max()
    y_annot = 0.75 * ymax

    # Significance label
    if p_corr < 0.001:
        sig = '***'
    elif p_corr < 0.01:
        sig = '**'
    elif p_corr < 0.05:
        sig = '*'
    else:
        sig = 'no'

    # Add line + star if significant
    if sig:
        ax.plot([0, 1], [y_annot, y_annot], color='black', linewidth=1.2)
        ax.text(0.5, y_annot + 0.02 * ymax, sig, ha='center', va='bottom', fontsize=14)

    ax.set_title(feature)

# Final layout
fig.suptitle('MIR Features by Area Depth', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_{signal_name}_violin_significance_raw_MIthreshold _{threshold}.png'), dpi=400, bbox_inches="tight")
plt.show()


###################################################################################################
from sklearn.preprocessing import StandardScaler

df = mi_pd_lt.copy()
z_cols = [col for col in df.columns if col.startswith('z_')]
#z_cols = ['pos', 'posdir', 'dir', 'speed',
#       'time', 'inner_trial_time', 'trial_id']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[z_cols].values)

# Clustering
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
initial_clusters = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Initialize with -10 (unassigned)
final_cluster_labels = np.full(X_scaled.shape[0], -10)

# Keep only top 75% closest to centroid
for cid in range(k):
    cluster_indices = np.where(initial_clusters == cid)[0]
    if len(cluster_indices) == 0:
        continue
    cluster_points = X_scaled[cluster_indices]
    centroid = centroids[cid]
    dists = np.linalg.norm(cluster_points - centroid, axis=1)
    threshold = np.percentile(dists, 50)
    keep_mask = dists <= threshold
    keep_indices = cluster_indices[keep_mask]
    final_cluster_labels[keep_indices] = cid

# Add final cluster labels to df (mi_pd_lt)
df['cluster'] = final_cluster_labels
# Now update the original dataframe with these labels
mi_pd['cluster'] = -10  # default: unassigned
mi_pd.loc[df.index, 'cluster'] = df['cluster'].values  # assign only valid cluster labels

mice_list = ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9',
             'ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1',
             'CalbEphys1GRIN1', 'CalbEphys1GRIN2', 'Thy1Ephys1GRIN1', 'Thy1Ephys1GRIN2']
mice_list = ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9',
             'ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']

# Save clusters to each session
for mouse in mice_list:
    mdata_dir = os.path.join(data_dir, mouse)
    msave_dir = os.path.join(save_dir, mouse)
    if not os.path.isdir(msave_dir): os.makedirs(msave_dir)
    mi_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_mi_{signal_name}_dict_alldir.pkl")
    for session in sorted(mi_dict):
        if 'lt' in session:
            clusters_mouse = mi_pd[mi_pd['mouse'] == mouse]['cluster'].values
            mi_dict[session]['cluster_id'] = clusters_mouse
    lrgu.save_pickle(msave_dir, f"{mouse}_mi_cluster_{k}_all_{signal_name}_dict.pkl", mi_dict)

tsne = TSNE(n_components=2, perplexity=15, learning_rate='auto', init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
df['tsne_1'], df['tsne_2'] = X_tsne[:, 0], X_tsne[:, 1]

#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#X_pca = pca.fit_transform(X_scaled)
#df['tsne_1'], df['tsne_2']= X_pca[:, 0], X_pca[:, 1]
#import umap
#umap_model = umap.UMAP(n_components=2, random_state=42)
#X_umap = umap_model.fit_transform(X_scaled)
#df['tsne_1'], df['tsne_2']= X_umap[:, 0], X_umap[:, 1]

# Separate assigned vs. unassigned
df_assigned = df[df['cluster'] != -10]
df_unassigned = df[df['cluster'] == -10]

# Plotting
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
axes = axes.flatten()

# --- Plot 1: Cluster ID ---
ax = axes[0]
# Gray background for discarded cells
ax.scatter(df_unassigned['tsne_1'], df_unassigned['tsne_2'], color='lightgray', s=20, label='Unassigned', alpha=0.5)
# Colored overlay for clustered cells
sns.scatterplot(data=df_assigned, x='tsne_1', y='tsne_2', hue='cluster',
                ax=ax, palette='tab10', s=40, alpha=0.8)
ax.set_title('t-SNE (colored by cluster)')
ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- Plot 2: Area ---
ax = axes[1]
ax.scatter(df_unassigned['tsne_1'], df_unassigned['tsne_2'], color='lightgray', s=20, alpha=0.5)
sns.scatterplot(data=df_assigned, x='tsne_1', y='tsne_2', hue='area',
                ax=ax, palette='Set2', s=40, alpha=0.8)
ax.set_title('t-SNE (colored by area)')
ax.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- Plot 3: Mouse ID ---
ax = axes[2]
ax.scatter(df_unassigned['tsne_1'], df_unassigned['tsne_2'], color='lightgray', s=20, alpha=0.5)
sns.scatterplot(data=df_assigned, x='tsne_1', y='tsne_2', hue='mouse',
                ax=ax, palette='tab20', s=40, alpha=0.8)
ax.set_title('t-SNE (colored by mouse ID)')
ax.legend(title='Mouse', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- Plot 4: total_MI ---
ax = axes[3]
ax.scatter(df_unassigned['tsne_1'], df_unassigned['tsne_2'], color='lightgray', s=20, alpha=0.5)
sc = ax.scatter(df_assigned['tsne_1'], df_assigned['tsne_2'], c=df_assigned['total_MI'],
                cmap='coolwarm', s=40, alpha=0.8, vmin=0, vmax=1.5)
ax.set_title('t-SNE (totalMI)')
plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.8)

for i, col in enumerate(behavior_keys):
    ax = axes[i + 4]
    ax.scatter(df_unassigned['tsne_1'], df_unassigned['tsne_2'], color='lightgray', s=15, alpha=0.5)
    sc = ax.scatter(df_assigned['tsne_1'], df_assigned['tsne_2'], c=df_assigned[col],
                    cmap='coolwarm', s=15, alpha=0.9, vmin=0, vmax=0.35)
    ax.set_title(f't-SNE (colored by {col})')
    plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.8)

# Final layout
fig.suptitle('t-SNE Embedding with Clustering, Area, Mouse ID, and Raw MI Features', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(data_dir, f'MI_cluster_all_{signal_name}_area_mouse_zscored_MIthreshold_{threshold}.png'), dpi=400, bbox_inches="tight")
plt.show()

##################################################################################
# Multiple k values
##########################################################
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define color mapping
area_colors = {
    'superficial': 'purple',
    'deep': 'gold'
}
# Set up
df_base = mi_pd_lt.copy()
z_cols = [f'z_{col}' for col in raw_mi_values]
scaler = StandardScaler()
X_base = scaler.fit_transform(df_base[z_cols].values)


k_values = [2, 3, 4, 5]

# Run t-SNE once (same across clusterings)
tsne = TSNE(n_components=2, perplexity=20, learning_rate='auto', init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_base)
df_base['tsne_1'], df_base['tsne_2'] = X_tsne[:, 0], X_tsne[:, 1]

# Run clustering for all k and store results
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_base[f'cluster_k{k}'] = kmeans.fit_predict(X_base)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    initial_clusters = kmeans.fit_predict(X_base)
    centroids = kmeans.cluster_centers_

    final_labels = np.full(X_base.shape[0], -10)  # default = unassigned

    for cid in range(k):
        cluster_indices = np.where(initial_clusters == cid)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_points = X_base[cluster_indices]
        centroid = centroids[cid]
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        threshold = np.percentile(dists, 95)
        keep_mask = dists <= threshold
        keep_indices = cluster_indices[keep_mask]
        final_labels[keep_indices] = cid

    df_base[f'cluster_k{k}'] = final_labels

# --- Plotting ---
raw_mi_cols = list(raw_mi_values.keys())
n_cluster_plots = len(k_values)
metadata_cols = ['area', 'mouse']
n_metadata_plots = len(metadata_cols)
n_feature_plots = len(raw_mi_cols)

n_cols = 4
n_rows = int(np.ceil((n_cluster_plots + n_metadata_plots + n_feature_plots) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = axes.flatten()

plot_idx = 0

# Top row: Clustering results
for k in k_values:
    sns.scatterplot(data=df_base, x='tsne_1', y='tsne_2', hue=f'cluster_k{k}',
                    ax=axes[plot_idx], palette='tab10', s=40, alpha=0.8)
    axes[plot_idx].set_title(f't-SNE (cluster k={k})')
    axes[plot_idx].legend(title=f'k={k}', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 15)
    plot_idx += 1

# Metadata: area and mouse
for meta_col in metadata_cols:
    if meta_col == 'area':
        use_palette = area_colors
    else:
        use_palette = 'tab20'
    sns.scatterplot(data=df_base, x='tsne_1', y='tsne_2', hue=meta_col,
                    ax=axes[plot_idx], palette=use_palette, s=40, alpha=0.8)
    axes[plot_idx].set_title(f't-SNE (colored by {meta_col})')
    axes[plot_idx].legend(title=meta_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plot_idx += 1

# Raw MI values
for col in raw_mi_cols:
    sc = axes[plot_idx].scatter(df_base['tsne_1'], df_base['tsne_2'], c=df_base[col],
                                cmap='coolwarm', s=15, alpha=0.9, vmin=0, vmax=0.25)
    axes[plot_idx].set_title(f't-SNE (colored by {col})')
    plt.colorbar(sc, ax=axes[plot_idx], orientation='vertical', shrink=0.8)
    plot_idx += 1

# Turn off any unused axes
for i in range(plot_idx, len(axes)):
    axes[i].axis('off')

# Final layout
fig.suptitle('t-SNE Embedding with Clusterings (k=2-5), Metadata, and Raw MI Features', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(data_dir, f'MI_cluster_all_kALL_{signal_name}_overview_threshold_{threshold}.png'), dpi=400, bbox_inches="tight")
plt.show()

##################################################################
#
#             COUNTING CELLS
#
################################################################
from scipy.stats import mannwhitneyu

clusters_name = 'cluster_k4'
# Count neurons per mouse, area, and cluster
mouse_counts = df_base.groupby(['area', 'mouse', clusters_name]).size().reset_index(name='count')
# Total neurons per mouse
mouse_totals = df_base.groupby(['area', 'mouse']).size().reset_index(name='total_neurons')

# Merge and normalize
mouse_counts = pd.merge(mouse_counts, mouse_totals, on=['area', 'mouse'])
mouse_counts['normalized'] = mouse_counts['count'] / mouse_counts['total_neurons']

# Plotting
palette = {'superficial': 'purple', 'deep': 'gold'}
plt.figure(figsize=(10, 6))

# Barplot
ax = sns.barplot(data=mouse_counts, x=clusters_name, y='normalized',
                 hue='area', palette=palette, ci='sd', edgecolor='k')

# Overlay mouse-level dots
sns.stripplot(data=mouse_counts, x=clusters_name, y='normalized',
              hue='area', dodge=True, color='black', size=5,
              jitter=False, ax=ax)

# Remove duplicate legend
handles, labels = ax.get_legend_handles_labels()
n = len(set(mouse_counts['area']))
plt.legend(handles[:n], labels[:n], title='Area', loc='upper right')

# Statistical tests: Mann-Whitney U between superficial and deep for each cluster
clusters = sorted(mouse_counts[clusters_name].unique())
y_offset = 0.02

for clust in clusters:
    group = mouse_counts[mouse_counts[clusters_name] == clust]
    sup_vals = group[group['area'] == 'superficial']['normalized']
    deep_vals = group[group['area'] == 'deep']['normalized']

    if len(sup_vals) > 0 and len(deep_vals) > 0:
        stat, p = mannwhitneyu(sup_vals, deep_vals, alternative='two-sided')
        # Annotate significance level
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''

        if sig:
            max_y = group['normalized'].max()
            ax.text(clust, max_y + y_offset, sig, ha='center', va='bottom', fontsize=14, color='black')

# Labels and save
plt.title(f'Normalized Neuron Counts per Cluster (by Area Depth: {clusters_name})')
plt.ylabel('Fraction of Neurons in Cluster')
plt.xlabel('Cluster ID')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_{clusters_name}_all_{signal_name}_area_counts_significant.png'), dpi=400,
            bbox_inches="tight")
plt.show()

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

pvals = []
cluster_ids = sorted(mouse_counts[clusters_name].unique())

for clust in cluster_ids:
    group = mouse_counts[mouse_counts[clusters_name] == clust]
    sup_vals = group[group['area'] == 'superficial']['normalized']
    deep_vals = group[group['area'] == 'deep']['normalized']
    if len(sup_vals) > 0 and len(deep_vals) > 0:
        stat, p = mannwhitneyu(sup_vals, deep_vals, alternative='two-sided')
    else:
        p = 1.0
    pvals.append(p)

print(pvals)
# FDR correction
corrected_pvals = multipletests(pvals, method='fdr_bh')[1]


from scipy.stats import chi2_contingency, fisher_exact
# Create contingency table: rows = superficial/deep, columns = cluster
contingency = pd.crosstab(df_base['area'], df_base['cluster_k3'])
chi2, p, dof, expected = chi2_contingency(contingency)

print("Chi-square Test across clusters")
print("p-value:", p)
print("Expected counts:\n", expected)


##############################################################
#
#                       HEATMAP
#
##############################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
def plot_tsne_area_proportion_heatmap(df, bins=50, sigma=1.5):
    """
    Plots a smoothed heatmap showing the proportion of superficial vs deep cells across t-SNE space
    using a custom purple-to-gold colormap.
    """
    # Coordinates and area depth
    x = df['tsne_1'].values
    y = df['tsne_2'].values
    depth = df['area'].values
    # Initialize histograms
    heatmap_sup = np.zeros((bins, bins))
    heatmap_deep = np.zeros((bins, bins))
    heatmap_total = np.zeros((bins, bins))
    xedges = np.linspace(x.min(), x.max(), bins + 1)
    yedges = np.linspace(y.min(), y.max(), bins + 1)
    # Bin cells
    for i in range(len(x)):
        xi = np.searchsorted(xedges, x[i]) - 1
        yi = np.searchsorted(yedges, y[i]) - 1
        if 0 <= xi < bins and 0 <= yi < bins:
            if depth[i] == 'superficial':
                heatmap_sup[yi, xi] += 1
            elif depth[i] == 'deep':
                heatmap_deep[yi, xi] += 1
            heatmap_total[yi, xi] += 1
    # Compute normalized difference
    with np.errstate(divide='ignore', invalid='ignore'):
        prop_diff = (heatmap_sup - heatmap_deep) / heatmap_total
        prop_diff[np.isnan(prop_diff)] = 0
    # Smooth the proportion map
    prop_diff_smooth = gaussian_filter(prop_diff, sigma=sigma)
    # Define custom purple to gold colormap
    purple_gold_cmap = LinearSegmentedColormap.from_list(
        "purple_gold", ["purple", "gold"]
    )
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(prop_diff_smooth, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               origin='lower', cmap=purple_gold_cmap, vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(label='Proportion Superficial (+1) → Deep (-1)')
    plt.title('Smoothed Spatial Distribution: Superficial vs Deep Cells')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()
plot_tsne_area_proportion_heatmap(df_base, bins=100, sigma = 1)


##############################################################
#
#            Clustering deep and sup separately
#
##############################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

k = 3
for area in ['deep','superficial']:
    df = mi_pd_lt[mi_pd_lt['area'] == area]
    # 1. Select the MI columns
    mi_cols = ['mi_pos', 'mi_posdir', 'mi_dir', 'mi_speed',
               'mi_time', 'mi_inner_trial_time', 'mi_trial_id']
    X = df[mi_cols].values
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = os.path.join(data_dir, mouse)  # mouse save dir
        msave_dir = os.path.join(save_dir, mouse)  # mouse save dir
        mi_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_mi_{signal_name}_dict.pkl")
        session_names = list(mi_dict.keys())
        session_names.sort()
        for idx, session in enumerate(session_names):
            if 'lt' in session:
                mice_df = df[df['mouse'] == mouse]
                clusters_mouse = np.array(mice_df['cluster'])
                mi_dict[session]['cluster_id'] = clusters_mouse
        lrgu.save_pickle(msave_dir, f"{mouse}_mi_cluster_{signal_name}_dict.pkl", mi_dict)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Add to DataFrame
    df['tsne_1'] = X_tsne[:, 0]
    df['tsne_2'] = X_tsne[:, 1]

    # --- Plotting Setup ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()
    # Plot 1: t-SNE colored by cluster
    sns.scatterplot(data=df, x='tsne_1', y='tsne_2', hue='cluster',
                    ax=axes[0], palette='tab10', s=40, alpha=0.8)
    axes[0].set_title('t-SNE (colored by cluster)')
    axes[0].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Plot 2: t-SNE colored by mouse name
    sns.scatterplot(data=df, x='tsne_1', y='tsne_2', hue='mouse',
                    ax=axes[1], palette='Set2', s=40, alpha=0.8)
    axes[1].set_title('t-SNE (colored by mouse)')
    axes[1].legend(title='Mouse', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Plot 3–9: t-SNE colored by each MI feature
    for i, col in enumerate(mi_cols):
        ax = axes[i+2]
        sc = ax.scatter(df['tsne_1'], df['tsne_2'], c=df[col],
                        cmap='coolwarm', s=15, alpha=0.9, vmin = 0, vmax = 0.4)
        ax.set_title(f't-SNE (colored by {col})')
        plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.8)
    # Final layout
    fig.suptitle('t-SNE Embedding with Clustering, Mouse ID, and MI Features', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(data_dir, f'MI_cluster_{area}_{signal_name}.png'), dpi=400, bbox_inches="tight")
    plt.show()

