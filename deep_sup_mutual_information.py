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

                    pos_dir = pos[:, 0] * mov_dir
                    inner_time = lrgu.get_signal(session_pd, 'inner_trial_time')
                    trial_id = lrgu.get_signal(session_pd, 'trial_id_mat')
                    time = np.arange(0,pos.shape[0])

                    signal = lrgu.get_signal(session_pd, signal_name)
                    #noise_idx, signal_idx = filter_noisy_outliers(signal)

                    behaviours_list = [pos[:,0], pos_dir, mov_dir, speed, time, inner_time, trial_id]
                    beh_names = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID']
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
                    mi_dict[maze]['behaviour'] = behaviours_list
                    mi_dict[maze]['signal'] = signal
                    mi_dict[maze]['valid_index'] = valid_index
                    mi_dict[maze]['MIR'] = mi_all

                lrgu.save_pickle(msave_dir,f"{mouse}_mi_{signal_name}_dict.pkl", mi_dict)
                lrgu.print_time_verbose(local_time, global_time)
                sys.stdout = original
                f.close()



################################################################################################
#
#                           PLOTTING MIR                                                        #
#
################################################################################################


import pandas as pd
import matplotlib.pyplot as plt
#import ptitprince as pt

#from statsmodels.formula.api import ols
#import statsmodels.api as sm
from scipy import stats
import seaborn as sns

data_dir = os.path.join(base_dir, 'MIR')

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'

mouse_name_list = []
mi_pos_list= []
mi_posdir_list = []
mi_speed_list = []
mi_dir_list = []
mi_time_list = []
mi_inner_time_list = []
mi_trial_id_list = []
area_list = []
session_list = []

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = os.path.join(data_dir, mouse)  # mouse save dir
        mi_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_mi_{signal_name}_dict.pkl")
        session_names = list(mi_dict.keys())
        session_names.sort()
        for idx, session in enumerate(session_names):
            if 'lt' in session:
                session_type = 'lt'
            if 'rot' in session:
                session_type = 'rot'
            for neuron in range(len(mi_dict[session]['MIR'][0])):
                mouse_name_list.append(mouse)
                area_list.append(area)
                session_list.append(session_type)
                mi_pos_list.append(mi_dict[session]['MIR'][0][neuron])
                mi_posdir_list.append(mi_dict[session]['MIR'][1][neuron])
                mi_dir_list.append(mi_dict[session]['MIR'][2][neuron])
                mi_speed_list.append(mi_dict[session]['MIR'][3][neuron])
                mi_time_list.append(mi_dict[session]['MIR'][4][neuron])
                mi_inner_time_list.append(mi_dict[session]['MIR'][5][neuron])
                mi_trial_id_list.append(mi_dict[session]['MIR'][6][neuron])

mi_pd = pd.DataFrame(data={ 'mouse': mouse_name_list,
                            'area': area_list,
                            'session_type': session_list,
                            'mi_pos': mi_pos_list,
                            'mi_posdir': mi_posdir_list,
                            'mi_dir': mi_dir_list,
                            'mi_speed': mi_speed_list,
                            'mi_time': mi_time_list,
                            'mi_inner_trial_time': mi_inner_time_list,
                            'mi_trial_id': mi_trial_id_list
                           })


mi_pd_lt = mi_pd[mi_pd['session_type'] == 'lt']
palette = ['purple', 'yellow']  # Define your own list of colors

fig, axes = plt.subplots(1,7, figsize = (21,3))

sns.violinplot(mi_pd_lt, hue="area", y="mi_pos", ax=axes[0], palette = palette, legend= False)
#sns.stripplot(mi_pd_lt,hue="area", y="mi_pos",palette = 'dark:k', dodge=True, size=5, jitter=True, ax = axes[0],legend=False)

sns.violinplot(mi_pd_lt, hue="area", y="mi_posdir", ax=axes[1], palette = palette, legend= False)
#sns.stripplot(mi_pd_lt,hue="area", y="mi_posdir",palette = 'dark:k', dodge=True, size=5, jitter=True, ax = axes[1],legend=False)

sns.violinplot(mi_pd_lt, hue="area", y="mi_dir", ax=axes[2], palette = palette, legend= False)
#sns.stripplot(mi_pd_lt,hue="area", y="mi_dir",palette = 'dark:k', dodge=True, size=5, jitter=True, ax = axes[2],legend=False)

sns.violinplot(mi_pd_lt, hue="area", y="mi_speed", ax=axes[3], palette = palette, legend= False)
#sns.stripplot(mi_pd_lt,hue="area", y="mi_speed",palette = 'dark:k', dodge=True, size=5, jitter=True, ax = axes[3],legend=False)

sns.violinplot(mi_pd_lt, hue="area", y="mi_time", ax=axes[4], palette = palette, legend= False)
#sns.stripplot(mi_pd_lt,hue="area", y="mi_time",palette = 'dark:k', dodge=True, size=5, jitter=True, ax = axes[4],legend=False)

sns.violinplot(mi_pd_lt, hue="area", y="mi_inner_trial_time", ax=axes[5], palette = palette, legend= False)
#sns.stripplot(mi_pd_lt,hue="area", y="mi_inner_trial_time",palette = 'dark:k', dodge=True, size=5, jitter=True, ax = axes[5],legend=False)

sns.violinplot(mi_pd_lt, hue="area", y="mi_trial_id", ax=axes[6], palette = palette, legend= False)
#sns.stripplot(mi_pd_lt,hue="area", y="mi_trial_id",palette = 'dark:k', dodge=True, size=5, jitter=True, ax = axes[6],legend=False)

fig.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_{signal_name}.png'), dpi=400, bbox_inches="tight")
###################################################################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

layer = 'superficial'
df = mi_pd_lt[mi_pd_lt['area'] == layer]

# 1. Select the MI columns
mi_cols = ['mi_pos', 'mi_posdir', 'mi_dir', 'mi_speed',
           'mi_time', 'mi_inner_trial_time', 'mi_trial_id']

X = df[mi_cols].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

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
# Add mouse names as annotations (centered)
#for mouse_name in df['mouse'].unique():
#    x_mean = df[df['mouse'] == mouse_name]['tsne_1'].mean()
#    y_mean = df[df['mouse'] == mouse_name]['tsne_2'].mean()
#    axes[1].text(x_mean, y_mean, mouse_name, fontsize=9, weight='bold', ha='center', va='center')

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
plt.savefig(os.path.join(data_dir, f'MI_cluster_{layer}_{signal_name}.png'), dpi=400, bbox_inches="tight")
plt.show()