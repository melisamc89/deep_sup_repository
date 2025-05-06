import sys, copy, os

import numpy as np
import pandas as pd
import learning_repo.general_utils as lrgu
import sys, copy, os
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

learning_dir = '/home/melma31/Documents/learning_project/'
deep_sup_dir = '/home/melma31/Documents/deepsup_project/'
data_dir = os.path.join(learning_dir, 'processed_data')
save_dir = os.path.join(learning_dir, 'mutual_info')
if not os.path.isdir(save_dir): os.makedirs(save_dir)


learning_condition = 'learners'
use_values = 'z'
reduce = 'tSNE' ### 'PCA', 'UMAP'
if learning_condition == 'learners':
    mice_list = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
    learning_name_output = learning_condition
    learners = [0, 1, 2, 3, 4]
    learners_names = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
    non_learners = []
    non_learners_names = []
else:
    mice_list = ['M2019', 'M2021', 'M2022', 'M2023', 'M2024', 'M2025', 'M2026']
    learning_name_output = learning_condition + '_non_learners'
    learners = [0,1,2,3,4]
    learners_names = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
    non_learners = [5,6]
    non_learners_names = ['M2021', 'M2022']

mice_list = learners + non_learners
mice_names = learners_names + non_learners_names
case = 'mov_same_length'
cells = 'all_cells'
signal_name = 'clean_traces'
### re-arange dictionaries for session ordering.

MIR_dict = {'session1':{},'session2':{},'session3':{},'session4':{}}
MI_dict = {'session1':{},'session2':{},'session3':{},'session4':{}}
NMI_dict = {'session1':{},'session2':{},'session3':{},'session4':{}}
MI_total_dict =  {'session1':{},'session2':{},'session3':{},'session4':{}}
# Define learners and non-learners
# Dummy behavioral labels for illustration (assuming 6)
behavior_labels = ['pos', 'posdir', 'dir', 'speed', 'time', 'trial_id_mat']
##### create joint organized dictionary
sessions_names = ['session1','session2','session3','session4']
for mouse in mice_names:
    msave_dir = os.path.join(save_dir, mouse) #mouse save dir
    mi_dict = lrgu.load_pickle(msave_dir,f"{mouse}_mi_{case}_{cells}_{signal_name}_dict.pkl")
    session_names = list(mi_dict.keys())
    session_names.sort()
    for idx, session in enumerate(session_names):
        MIR = np.vstack(mi_dict[session]['MIR']).T
        MIR_dict[sessions_names[idx]][mouse] = MIR

MI_session_dict =  {'session1':{},'session2':{},'session3':{},'session4':{}}
mouse_session_list =  {'session1':{},'session2':{},'session3':{},'session4':{}}
for session in sessions_names:
    MIR_list =[]
    mouse_session = []
    for idx,mouse in enumerate(mice_names):
        from scipy.stats import zscore
        n_cells = MIR_dict[session][mouse].shape[0]
        MIR_list.append(MIR_dict[session][mouse])
        mouse_array = np.ones((n_cells,1))*idx
        mouse_session.append(mouse_array)
    MIR = np.vstack(MIR_list)
    mouse_session_final = np.vstack(mouse_session)
    mouse_session_list[session]['mice'] = mouse_session_final
    MI_session_dict[session]['MIR']=MIR
# --- Setup ---
session = 'session4'
k = 3
unassigned_cluster_id = -10  # label for unassigned cells
# Scale MI features
mi_raw = zscore(MI_session_dict[session]['MIR'], axis=1)
mi_scaled = StandardScaler().fit_transform(mi_raw)
# Dimensionality reduction (optional visualization)
if reduce == 'PCA':
    reducer = PCA(n_components=2)
    mi_reduced = reducer.fit_transform(mi_scaled)
    reducer_name = 'PC'
elif reduce == 'tSNE':
    reducer = TSNE(n_components=2, perplexity=50, random_state=42, init='pca')
    mi_reduced = reducer.fit_transform(mi_scaled)
    reducer_name = 'tSNE'
# --- Clustering in Original Feature Space ---
kmeans = KMeans(n_clusters=k, random_state=42)
initial_clusters = kmeans.fit_predict(mi_scaled)
centroids = kmeans.cluster_centers_
# Assign only 60% of cells per cluster (closest in original space)
final_cluster_labels = np.full(mi_scaled.shape[0], unassigned_cluster_id)  # default = unassigned
for cid in range(k):
    cluster_indices = np.where(initial_clusters == cid)[0]
    if len(cluster_indices) == 0:
        continue
    cluster_points = mi_scaled[cluster_indices]
    centroid = centroids[cid]
    dists = np.linalg.norm(cluster_points - centroid, axis=1)
    threshold = np.percentile(dists, 75)
    keep_mask = dists <= threshold
    keep_indices = cluster_indices[keep_mask]
    final_cluster_labels[keep_indices] = cid

data_dir = os.path.join(deep_sup_dir, 'MIR')
save_dir = os.path.join(deep_sup_dir, 'MIR','Transfer')
if not os.path.isdir(save_dir): os.makedirs(save_dir)
signal_name = 'clean_traces'
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
threshold = 0
mi_pd['topcells'] = mi_pd['total_MI'].apply(lambda x: 'yes' if x >= threshold else 'no')

mi_pd_lt = mi_pd[mi_pd['session_type'] == 'lt']
mi_pd_lt = mi_pd_lt[mi_pd_lt['topcells'] == 'yes']

from sklearn.metrics import pairwise_distances

# Columns used for clustering
z_cols = [f'z_{key}' for key in raw_mi_values]
# Get feature matrix for mi_pd_lt
X_target = mi_pd_lt[z_cols].values
X_target_scaled = StandardScaler().fit_transform(X_target)

# Predict clusters
pred_clusters = kmeans.predict(X_target_scaled)
pred_centroids = kmeans.cluster_centers_

# Assign only top 75% closest per cluster
final_labels = np.full(X_target_scaled.shape[0], -10)  # default: unassigned

for cid in range(k):
    cluster_indices = np.where(pred_clusters == cid)[0]
    if len(cluster_indices) == 0:
        continue
    cluster_points = X_target_scaled[cluster_indices]
    centroid = pred_centroids[cid]
    dists = np.linalg.norm(cluster_points - centroid, axis=1)
    threshold = np.percentile(dists, 75)
    keep_mask = dists <= threshold
    keep_indices = cluster_indices[keep_mask]
    final_labels[keep_indices] = cid

# Assign to new column in mi_pd_lt
mi_pd_lt = mi_pd_lt.copy()
mi_pd_lt['transferred_cluster'] = final_labels

mi_pd = mi_pd_lt.copy()
# Save clusters to each session
mice_list =  ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9',
              'ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
for mouse in mice_list:
    mdata_dir = os.path.join(data_dir, mouse)
    msave_dir = os.path.join(save_dir, mouse)
    if not os.path.isdir(msave_dir): os.makedirs(msave_dir)
    mi_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_mi_{signal_name}_dict_alldir.pkl")
    for session in sorted(mi_dict):
        if 'lt' in session:
            clusters_mouse = mi_pd[mi_pd['mouse'] == mouse]['transferred_cluster'] .values
            mi_dict[session]['cluster_id'] = clusters_mouse
    lrgu.save_pickle(msave_dir, f"{mouse}_mi_transferred_cluster_{k}_all_{signal_name}_dict.pkl", mi_dict)



X_scaled = X_target_scaled.copy()
tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto', init='pca', random_state=42)
X_tsne = reducer.fit_transform(X_scaled)
# Transform new data
#X_tsne = reducer.transform(X_target_scaled)  # Same scaling as before!
mi_pd_lt['tsne_1'], mi_pd_lt['tsne_2'] = X_tsne[:, 0], X_tsne[:, 1]

# Separate assigned vs. unassigned
df_assigned = mi_pd_lt[mi_pd_lt['transferred_cluster'] != -10 ]
df_unassigned = mi_pd_lt[mi_pd_lt['transferred_cluster'] == -10]
# Plotting
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
axes = axes.flatten()
# --- Plot 1: Cluster ID ---
ax = axes[0]
# Gray background for discarded cells
ax.scatter(df_unassigned['tsne_1'], df_unassigned['tsne_2'], color='lightgray', s=20, label='Unassigned', alpha=0.5)
# Colored overlay for clustered cells
sns.scatterplot(data=df_assigned, x='tsne_1', y='tsne_2', hue='transferred_cluster',
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
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_zscored_MIthreshold_{threshold}.png'), dpi=400, bbox_inches="tight")
plt.show()

from scipy.stats import mannwhitneyu

clusters_name = 'transferred_cluster'
# Count neurons per mouse, area, and cluster
mouse_counts = mi_pd_lt.groupby(['area', 'mouse', clusters_name]).size().reset_index(name='count')
# Total neurons per mouse
mouse_totals = mi_pd_lt.groupby(['area', 'mouse']).size().reset_index(name='total_neurons')

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
y_offset = 0.005

for clust in clusters:
    group = mouse_counts[mouse_counts[clusters_name] == clust]
    sup_vals = group[group['area'] == 'superficial']['normalized']
    deep_vals = group[group['area'] == 'deep']['normalized']

    if len(sup_vals) > 0 and len(deep_vals) > 0:
        stat, p = mannwhitneyu(sup_vals, deep_vals, alternative='two-sided')
        # Annotate significance level
        sig = f"{p:.3f}"
        #if p < 0.001:
        #    sig = '***'
        #elif p < 0.01:
        #    sig = '**'
        #elif p < 0.05:
        #    sig = '*'
        print(p)
        if sig:
            max_y = group['normalized'].max()
            ax.text(clust + 1, max_y + y_offset, sig, ha='center', va='bottom', fontsize=14, color='black')

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

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

raw_mi_cols = list(raw_mi_values.keys())  # e.g., ['pos', 'posdir', 'dir', ...]

clusters = sorted(mi_pd_lt['transferred_cluster'].unique())
palette = {'superficial': 'purple', 'deep': 'gold'}

# Plot per MI feature
for info in raw_mi_cols:
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(
        data=mi_pd_lt,
        x='transferred_cluster',
        y=info,
        hue='area',
        palette=palette,
        split=True,
        inner='quartile',
        cut=0
    )
    plt.title(f'{info} across Transferred Clusters (superficial vs deep)')
    plt.xlabel('Transferred Cluster')
    plt.ylabel('Raw Mutual Information')

    # Significance testing per cluster
    y_offset = 0.0001
    max_val = mi_pd_lt[info].max()

    for clust in clusters:
        group = mi_pd_lt[mi_pd_lt['transferred_cluster'] == clust]
        sup_vals = group[group['area'] == 'superficial'][info]
        deep_vals = group[group['area'] == 'deep'][info]

        if len(sup_vals) > 0 and len(deep_vals) > 0:
            stat, p = mannwhitneyu(sup_vals, deep_vals, alternative='two-sided')
            sig = f"{p:.4f}"
            #if p < 0.001:
            #    sig = '***'
            #elif p < 0.01:
            #    sig = '**'
            #elif p < 0.05:
            #    sig = '*'

            if sig:
                ax.text(clust + 1, max_val + y_offset, sig,
                        ha='center', va='bottom', fontsize=12, color='black')

    plt.legend(title='Area', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"violin_MI_{info}_by_cluster_area.png"), dpi=300)
    plt.show()

