import os, copy
from datetime import datetime
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.ndimage import gaussian_filter1d

import umap
from sklearn.decomposition import PCA

import learning_repo.general_utils as lrgu
import learning_repo.geometric_utils as lrgeo
from statistics import mode
import matplotlib.pyplot as plt
from datetime import datetime
from structure_index import compute_structure_index

from scipy.signal import convolve

def gaussian_kernel(size, sigma):
    """ Create a Gaussian kernel given a size and sigma (standard deviation). """
    # Create a vector from -size to size
    x = np.arange(-size, size + 1)
    # Gaussian formula
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    # Normalize the kernel
    return kernel / kernel.sum()


def apply_gaussian_filter_to_matrix(data, kernel_size=5, sigma=1):
    """ Apply a Gaussian filter to each column of a matrix. """
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    # Initialize an array to hold the smoothed data
    smoothed_data = np.zeros_like(data)
    # Loop through each column in the data matrix
    for i in range(data.shape[1]):
        # Apply convolution to smooth the data for each column
        smoothed_data[:, i] = convolve(data[:, i], kernel, mode='same')
    return smoothed_data

from sklearn.metrics import pairwise_distances

def filter_noisy_outliers(data):
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,1), axis=1) - 1
    noiseIdx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    signalIdx = np.where(nn_dist >= np.percentile(nn_dist, 20))[0]
    return noiseIdx, signalIdx

base_dir =  '/home/melma31/Documents/deepsup_project/'

data_dir = os.path.join(base_dir, 'data')
save_dir = os.path.join(base_dir, 'results')
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

models = ['Calb', 'Chrna7', 'Thy1']

num_neigh = 120
dim = 3
min_dist = 0.1


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
                print('Mouse: ' + mouse)
                dynamics_dict = {}
                mdata_dir = os.path.join(data_dir,coloring,paradigm_,'processed_data',model ,mouse) #mouse data dir
                msave_dir = os.path.join(save_dir, mouse) #mouse save dir
                if not os.path.isdir(msave_dir): os.makedirs(msave_dir)

                #print(f"\t#### {mouse}: {case} | {cells} | {signal_name} ####")
                #print(f'\t{datetime.now():%Y-%m-%d %H:%M}\n')
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
                    mice_maze = mouse_dict[maze]

                    params = {'columns_to_rename': {'Fs': 'sf', 'pos': 'position', 'vel': 'speed'}}
                    session_pd = copy.deepcopy(mice_maze)
                    for old, new in params['columns_to_rename'].items():
                        if old in session_pd.columns: session_pd.rename(columns={old: new}, inplace=True)
                    session_pd = lrgu.add_mov_direction_mat_field(session_pd)
                    session_pd = lrgu.add_inner_trial_time_field(session_pd)
                    session_pd = lrgu.add_trial_id_mat_field(session_pd)

                    signal = lrgu.get_signal(session_pd, 'clean_traces')
                    pos = lrgu.get_signal(session_pd, 'position')
                    speed = lrgu.get_signal(session_pd, 'speed')
                    trial_id = lrgu.get_signal(session_pd, 'trial_id_mat')
                    mov_dir = lrgu.get_signal(session_pd, 'mov_direction')


                    #clean mov dir to avoid gaps in ring
                    temp_mov = np.diff(pos,axis=0)*20
                    temp_mov = np.concatenate([temp_mov[0,:].reshape(-1,2), temp_mov], axis= 0)[:,0]
                    temp_mov = gaussian_filter1d(temp_mov, sigma = 5, axis = 0)
                    temp_mov[temp_mov<0] = -1
                    temp_mov = np.sign(temp_mov)
                    mov_dir[mov_dir==0] = temp_mov[np.where(mov_dir==0)[0]]

                    #compute umap
                    umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist,                                                              random_state=42)
                    umap_model.fit(signal)
                    umap_emb = umap_model.transform(signal)
                    #mean center umap to facilitate future steps
                    umap_emb -= umap_emb.mean(axis=0)

                    mov_dir_color = lrgu.get_dir_color(mov_dir)
                    row = 3
                    col = 6
                    fig = plt.figure(figsize=(25, 9))
                    pos = pos[:,0]
                    ax = fig.add_subplot(row, col, 1, projection='3d')
                    b = ax.scatter(*umap_emb[:, :3].T, color=mov_dir_color, s=1, alpha=0.5)
                    ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
                    ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
                    ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
                    ax.set_title('UMAP Direction')

                    ax = fig.add_subplot(row, col, 7, projection='3d')
                    b = ax.scatter(*umap_emb[:, :3].T, c=pos, s=1, alpha=0.5)
                    ax.set_title('UMAP Position')

                    ax = fig.add_subplot(row, col, 13, projection='3d')
                    b = ax.scatter(*umap_emb[:, :3].T, c=np.arange(umap_emb.shape[0]), s=1, alpha=0.5)
                    ax.set_title('UMAP Time')
                    si_vector = np.zeros((6,))
                    si_vector_pos = np.zeros((6,))
                    si_vector_dir = np.zeros((6,))

                    time = np.arange(0,len(signal))
                    noise_idx, signal_idx = filter_noisy_outliers(signal)
                    fake_time = time[signal_idx]
                    print(f"Working on session {maze}:", end='')
                    si, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   fake_time)
                    si_vector[0] = si
                    si, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   pos[signal_idx])
                    si_vector_pos[0] = si
                    si, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   mov_dir[signal_idx])
                    si_vector_dir[0] = si


                    for index , filter_size in enumerate([5,10,15,20,25]):
                        rates = apply_gaussian_filter_to_matrix(signal, kernel_size=filter_size, sigma=20)
                        pca = PCA(n_components=10)
                        rates_pca = pca.fit_transform(rates)
                        vel_rates_pca = np.diff(rates_pca, axis=0)
                        rates_pca = rates_pca[:-1, :]  # skip last
                        #compute umap
                        umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist,                                                              random_state=42)
                        umap_model.fit(rates)
                        umap_emb = umap_model.transform(rates)
                        #mean center umap to facilitate future steps
                        umap_emb -= umap_emb.mean(axis=0)

                        si, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   fake_time)
                        si_vector[index+1] = si

                        si, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   pos[signal_idx])
                        si_vector_pos[index+1] = si
                        si, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   mov_dir[signal_idx])
                        si_vector_dir[index+1] = si

                        ax = fig.add_subplot(row, col, index + 2 , projection='3d')
                        b = ax.scatter(*umap_emb[:, :3].T, color=mov_dir_color, s=1, alpha=0.5)
                        ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
                        ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
                        ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
                        ax.set_title('UMAP Direction')

                        ax = fig.add_subplot(row, col, index + 8, projection='3d')
                        b = ax.scatter(*umap_emb[:, :3].T, c=pos, s=1, alpha=0.5)
                        ax.set_title('UMAP Position')

                        ax = fig.add_subplot(row, col, index + 14, projection='3d')
                        b = ax.scatter(*umap_emb[:, :3].T, c=np.arange(umap_emb.shape[0]), s=1, alpha=0.5)
                        ax.set_title('UMAP Time')

                    dynamics_dict[maze] = {
                        'signal': signal,
                        'pos': pos,
                        'speed': speed,
                        'trial_id': trial_id,
                        'mov_dir': mov_dir,
                        'si': si_vector,
                        'si_pos': si_vector_pos,
                        'si_dir': si_vector_dir,

                    }
                    fig.suptitle(f'{mouse}')
                    figure_name = f"{mouse}_UMAP_filter_"
                    msave_dir = os.path.join(base_dir, 'figures')  # mouse save dir
                    plt.savefig(os.path.join(msave_dir, figure_name + ".svg"), dpi=400, bbox_inches="tight")
                    plt.savefig(os.path.join(msave_dir, figure_name + ".png"), dpi=400,
                                bbox_inches="tight")

                pickle_name = f"{mouse}_si_filters.pkl"
                lrgu.save_pickle(msave_dir,pickle_name, dynamics_dict)



###########################################################################

                    #PLOT SI for different filters

##########################################################################



import pandas as pd
import matplotlib.pyplot as plt
#import ptitprince as pt

#from statsmodels.formula.api import ols
#import statsmodels.api as sm
from scipy import stats
import seaborn as sns


data_dir = os.path.join(base_dir, 'figures')

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'

mouse_name_list = []
si_umap = []
si_umap1 = []
si_umap2 = []
si_umap3 = []
si_umap4 = []
si_umap5 = []

area_list = []
session_list = []

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = data_dir
        si_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_si_filters.pkl")
        session_names = list(si_dict.keys())
        session_names.sort()
        for idx, session in enumerate(session_names):
            area_list.append(area)
            if 'lt' in session:
                session_type = 'lt'
            if 'rot' in session:
                session_type = 'rot'

            mouse_name_list.append(mouse)
            session_list.append(session_type)
            si_umap.append(si_dict[session]['si_pos'][0])
            si_umap1.append(si_dict[session]['si_pos'][1])
            si_umap2.append(si_dict[session]['si_pos'][2])
            si_umap3.append(si_dict[session]['si_pos'][3])
            si_umap4.append(si_dict[session]['si_pos'][4])
            si_umap5.append(si_dict[session]['si_pos'][5])


si_pd = pd.DataFrame(data={'mouse': mouse_name_list,
                           'area': area_list,
                     'session_type': session_list,
                     'si': si_umap,
                           'si1': si_umap1,
                           'si2': si_umap2,
                           'si3': si_umap3,
                           'si4': si_umap4,
                           'si5': si_umap5,
                           })

si_pd_lt = si_pd[si_pd['session_type'] == 'lt']
palette = ['purple', 'yellow']  # Define your own list of colors
fig, axes = plt.subplots(1,6, figsize = (30,5))
sns.barplot(si_pd_lt, hue="area", y="si", ax=axes[0], palette = palette, legend = False)
sns.stripplot(si_pd_lt,hue="area", y="si",palette = 'dark:k', dodge=True, size=5, jitter=True, ax=axes[0],legend=False)

sns.barplot(si_pd_lt, hue="area", y="si1", ax=axes[1], palette = palette, legend = False)
sns.stripplot(si_pd_lt,hue="area", y="si1",palette = 'dark:k', dodge=True, size=5, jitter=True, ax=axes[1],legend=False)

sns.barplot(si_pd_lt, hue="area", y="si2", ax=axes[2], palette = palette, legend = False)
sns.stripplot(si_pd_lt,hue="area", y="si2",palette = 'dark:k', dodge=True, size=5, jitter=True, ax=axes[2],legend=False)

sns.barplot(si_pd_lt, hue="area", y="si3", ax=axes[3], palette = palette, legend = False)
sns.stripplot(si_pd_lt,hue="area", y="si3",palette = 'dark:k', dodge=True, size=5, jitter=True, ax=axes[3],legend=False)

sns.barplot(si_pd_lt, hue="area", y="si4", ax=axes[4], palette = palette, legend = False)
sns.stripplot(si_pd_lt,hue="area", y="si4",palette = 'dark:k', dodge=True, size=5, jitter=True, ax=axes[4],legend=False)

sns.barplot(si_pd_lt, hue="area", y="si5", ax=axes[5], palette = palette, legend = False)
sns.stripplot(si_pd_lt,hue="area", y="si5",palette = 'dark:k', dodge=True, size=5, jitter=True, ax=axes[5],legend=False)

for  i in range(6):
    axes[i].set_ylim([0,1])

plt.savefig(os.path.join(data_dir, f'SI_pos_filter_{signal_name}.png'), dpi=400, bbox_inches="tight")



########################

import pandas as pd
import matplotlib.pyplot as plt
#import ptitprince as pt

#from statsmodels.formula.api import ols
#import statsmodels.api as sm
from scipy import stats
import seaborn as sns


data_dir = os.path.join(base_dir, 'figures')

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'

mouse_name_list = []
si_umap_time = []
si_umap_pos = []
si_umap_dir = []

area_list = []
session_list = []

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = data_dir
        si_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_si_filters.pkl")
        session_names = list(si_dict.keys())
        session_names.sort()
        for idx, session in enumerate(session_names):
            area_list.append(area)
            if 'lt' in session:
                session_type = 'lt'
            if 'rot' in session:
                session_type = 'rot'

            mouse_name_list.append(mouse)
            session_list.append(session_type)
            si_umap_time.append(si_dict[session]['si'])
            si_umap_pos.append(si_dict[session]['si_pos'])
            si_umap_dir.append(si_dict[session]['si_dir'])



si_pd = pd.DataFrame(data={'mouse': mouse_name_list,
                           'area': area_list,
                     'session_type': session_list,
                     'si': si_umap_time,
                           'si_pos': si_umap_pos,
                           'si_dir': si_umap_dir,
                           })

si_pd_lt = si_pd[si_pd['session_type'] == 'lt']
palette = ['purple', 'yellow']  # Define your own list of colors
fig, axes = plt.subplots(3,1, figsize = (8,5))
si_pd_sup = si_pd_lt[si_pd_lt['area'] == 'superficial']
si_pd_deep = si_pd_lt[si_pd_lt['area'] == 'deep']

for i in range(len(si_pd_sup)):
    axes[0].scatter(filters, si_pd_sup['si'].iloc[i], c = 'purple')
    axes[0].plot(filters, si_pd_sup['si'].iloc[i], c = 'purple')
for i in range(len(si_pd_deep)):
    axes[0].scatter(filters, si_pd_deep['si'].iloc[i], c = 'yellow')
    axes[0].plot(filters, si_pd_deep['si'].iloc[i], c = 'yellow')

for i in range(len(si_pd_sup)):
    axes[1].scatter(filters, si_pd_sup['si_pos'].iloc[i], c = 'purple')
    axes[1].plot(filters, si_pd_sup['si_pos'].iloc[i], c = 'purple')
for i in range(len(si_pd_deep)):
    axes[1].scatter(filters, si_pd_deep['si_pos'].iloc[i], c = 'yellow')
    axes[1].plot(filters, si_pd_deep['si_pos'].iloc[i], c = 'yellow')

for i in range(len(si_pd_sup)):
    axes[2].scatter(filters, si_pd_sup['si_dir'].iloc[i], c = 'purple')
    axes[2].plot(filters, si_pd_sup['si_dir'].iloc[i], c = 'purple')
for i in range(len(si_pd_deep)):
    axes[2].scatter(filters, si_pd_deep['si_dir'].iloc[i], c = 'yellow')
    axes[2].plot(filters, si_pd_deep['si_dir'].iloc[i], c = 'yellow')

for  i in range(3):
    axes[i].set_ylim([0,1])

plt.savefig(os.path.join(data_dir, f'SI_pos_filter_individual_{signal_name}.png'), dpi=400, bbox_inches="tight")
