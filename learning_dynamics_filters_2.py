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
import elephant

from scipy.signal import convolve

from elephant.kernels import GaussianKernel
from quantities import ms
import quantities as pq

import numpy as np
from elephant.kernels import GaussianKernel
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
                    #umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist,                                                              random_state=42)
                    #umap_model.fit(signal)
                    #umap_emb = umap_model.transform(signal)
                    #mean center umap to facilitate future steps
                    #umap_emb -= umap_emb.mean(axis=0)

                    mov_dir_color = lrgu.get_dir_color(mov_dir)
                    row = 3
                    col = 11
                    #fig = plt.figure(figsize=(25, 9))
                    pos = pos[:,0]
                    #ax = fig.add_subplot(row, col, 1, projection='3d')
                    #b = ax.scatter(*umap_emb[:, :3].T, color=mov_dir_color, s=1, alpha=0.5)
                    #ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
                    #ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
                    #ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
                    #ax.set_title('UMAP Direction')

                    #ax = fig.add_subplot(row, col, 12, projection='3d')
                    #b = ax.scatter(*umap_emb[:, :3].T, c=pos, s=1, alpha=0.5)
                    #ax.set_title('UMAP Position')

                    #ax = fig.add_subplot(row, col, 23, projection='3d')
                    #b = ax.scatter(*umap_emb[:, :3].T, c=np.arange(umap_emb.shape[0]), s=1, alpha=0.5)
                    #ax.set_title('UMAP Time')

                    si_vector = np.zeros((11,))
                    si_vector_pos = np.zeros((11,))
                    si_vector_dir = np.zeros((11,))

                    time = np.arange(0,len(signal))
                    noise_idx, signal_idx = filter_noisy_outliers(signal)
                    fake_time = time[signal_idx]
                    #signal_idx = np.arange(umap_emb.shape[0])
                    #fake_time = signal_idx
                    print(f"Working on session {maze}:", end='')
                    si, process_info, overlap_mat, _ = compute_structure_index(signal[signal_idx],
                                                                                   fake_time)
                    si_vector[0] = si
                    si, process_info, overlap_mat, _ = compute_structure_index(signal[signal_idx],
                                                                                   pos[signal_idx])
                    si_vector_pos[0] = si
                    si, process_info, overlap_mat, _ = compute_structure_index(signal[signal_idx],
                                                                                   mov_dir[signal_idx])
                    si_vector_dir[0] = si

                    kernels = [1,2,4,6,8,10,12,14,16,18]
                    for index , filter_size in enumerate(kernels):
                        rates = smooth_calcium_signals(signal, filter_size)
                        #pca = PCA(n_components=10)
                        #rates_pca = pca.fit_transform(rates)
                        #vel_rates_pca = np.diff(rates_pca, axis=0)
                        #rates_pca = rates_pca[:-1, :]  # skip last
                        #compute umap
                        #umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist,                                                              random_state=42)
                        #umap_model.fit(rates)
                        #umap_emb = umap_model.transform(rates)
                        #mean center umap to facilitate future steps
                        #umap_emb -= umap_emb.mean(axis=0)
                        umap_emb = rates.copy()
                        si, process_info, overlap_mat, _ = compute_structure_index(rates[signal_idx],
                                                                                   fake_time)
                        si_vector[index+1] = si

                        si, process_info, overlap_mat, _ = compute_structure_index(rates[signal_idx],
                                                                                   pos[signal_idx])
                        si_vector_pos[index+1] = si
                        si, process_info, overlap_mat, _ = compute_structure_index(rates[signal_idx],
                                                                                   mov_dir[signal_idx])
                        si_vector_dir[index+1] = si

                        #ax = fig.add_subplot(row, col, index + 2 , projection='3d')
                        #b = ax.scatter(*umap_emb[:, :3].T, color=mov_dir_color, s=1, alpha=0.5)
                        #ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
                        #ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
                        #ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
                        #ax.set_title('UMAP Direction')

                        #ax = fig.add_subplot(row, col, index + 13, projection='3d')
                        #b = ax.scatter(*umap_emb[:, :3].T, c=pos, s=1, alpha=0.5)
                        #ax.set_title('UMAP Position')

                        #ax = fig.add_subplot(row, col, index + 24, projection='3d')
                        #b = ax.scatter(*umap_emb[:, :3].T, c=np.arange(umap_emb.shape[0]), s=1, alpha=0.5)
                        #ax.set_title('UMAP Time')

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
                    #fig.suptitle(f'{mouse}')
                    #figure_name = f"{mouse}_UMAP_filter_"
                    #msave_dir = os.path.join(base_dir, 'figures')  # mouse save dir
                    #plt.savefig(os.path.join(msave_dir, figure_name + ".svg"), dpi=400, bbox_inches="tight")
                    #plt.savefig(os.path.join(msave_dir, figure_name + ".png"), dpi=400,
                    #            bbox_inches="tight")

                pickle_name = f"{mouse}_si_complete_signal_filters_0-18.pkl"
                lrgu.save_pickle(msave_dir,pickle_name, dynamics_dict)

##############################################################################################################
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
                    pca = PCA(n_components=10)
                    rates_pca = pca.fit_transform(signal)
                    umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist,                                                              random_state=42)
                    umap_model.fit(rates_pca)
                    umap_emb = umap_model.transform(rates_pca)
                    #mean center umap to facilitate future steps
                    #umap_emb -= umap_emb.mean(axis=0)

                    mov_dir_color = lrgu.get_dir_color(mov_dir)
                    row = 3
                    col = 11
                    #fig = plt.figure(figsize=(25, 9))
                    pos = pos[:,0]
                    #ax = fig.add_subplot(row, col, 1, projection='3d')
                    #b = ax.scatter(*umap_emb[:, :3].T, color=mov_dir_color, s=1, alpha=0.5)
                    #ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
                    #ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
                    #ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
                    #ax.set_title('UMAP Direction')

                    #ax = fig.add_subplot(row, col, 12, projection='3d')
                    #b = ax.scatter(*umap_emb[:, :3].T, c=pos, s=1, alpha=0.5)
                    #ax.set_title('UMAP Position')

                    #ax = fig.add_subplot(row, col, 23, projection='3d')
                    #b = ax.scatter(*umap_emb[:, :3].T, c=np.arange(umap_emb.shape[0]), s=1, alpha=0.5)
                    #ax.set_title('UMAP Time')

                    si_vector = np.zeros((11,))
                    si_vector_pos = np.zeros((11,))
                    si_vector_dir = np.zeros((11,))

                    time = np.arange(0,len(signal))
                    noise_idx, signal_idx = filter_noisy_outliers(signal)
                    fake_time = time[signal_idx]
                    #signal_idx = np.arange(umap_emb.shape[0])
                    #fake_time = signal_idx
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

                    kernels = [1,2,4,6,8,10,12,14,16,18]
                    for index , filter_size in enumerate(kernels):
                        rates = smooth_calcium_signals(signal, filter_size)
                        pca = PCA(n_components=10)
                        rates_pca = pca.fit_transform(rates)
                        #vel_rates_pca = np.diff(rates_pca, axis=0)
                        #rates_pca = rates_pca[:-1, :]  # skip last
                        #compute umap
                        umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist,                                                              random_state=42)
                        umap_model.fit(rates_pca)
                        umap_emb = umap_model.transform(rates_pca)
                        #mean center umap to facilitate future steps
                        #umap_emb -= umap_emb.mean(axis=0)
                        #umap_emb = rates.copy()
                        si, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   fake_time)
                        si_vector[index+1] = si

                        si, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   pos[signal_idx])
                        si_vector_pos[index+1] = si
                        si, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   mov_dir[signal_idx])
                        si_vector_dir[index+1] = si

                        #ax = fig.add_subplot(row, col, index + 2 , projection='3d')
                        #b = ax.scatter(*umap_emb[:, :3].T, color=mov_dir_color, s=1, alpha=0.5)
                        #ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
                        #ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
                        #ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
                        #ax.set_title('UMAP Direction')

                        #ax = fig.add_subplot(row, col, index + 13, projection='3d')
                        #b = ax.scatter(*umap_emb[:, :3].T, c=pos, s=1, alpha=0.5)
                        #ax.set_title('UMAP Position')

                        #ax = fig.add_subplot(row, col, index + 24, projection='3d')
                        #b = ax.scatter(*umap_emb[:, :3].T, c=np.arange(umap_emb.shape[0]), s=1, alpha=0.5)
                        #ax.set_title('UMAP Time')

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
                    #fig.suptitle(f'{mouse}')
                    #figure_name = f"{mouse}_UMAP_filter_"
                    #msave_dir = os.path.join(base_dir, 'figures')  # mouse save dir
                    #plt.savefig(os.path.join(msave_dir, figure_name + ".svg"), dpi=400, bbox_inches="tight")
                    #plt.savefig(os.path.join(msave_dir, figure_name + ".png"), dpi=400,
                    #            bbox_inches="tight")

                pickle_name = f"{mouse}_si_umap10pcs_filters_0-18.pkl"
                lrgu.save_pickle(msave_dir,pickle_name, dynamics_dict)



#############################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
#import ptitprince as pt
import numpy as np
import os
#from statsmodels.formula.api import ols
#import statsmodels.api as sm
from scipy import stats
import seaborn as sns
#[0,1,2,5,7,10,15,20,25,30,50]
filters = np.array([0,1,2,4,6,8,10,12,14,16,18])*50
data_dir = os.path.join(base_dir, 'figures')

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'


#pickle_name = f"{mouse}_si_umap_10pcs_filters_0-18.pkl"
#lrgu.save_pickle(msave_dir, pickle_name, dynamics_dict)

file_names = ['_si_complete_signal_filters_','_si_filters_','_si_umap_10pcs_filters_','_si_umap10pcs_filters_']
si_pd_list = []

for type in file_names :
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
            si_dict = lrgu.load_pickle(mdata_dir, f"{mouse}"+type+ "0-18.pkl")
            ##CZ3_si_filters_0-18.pkl
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

    si_pd_list.append(si_pd)

mark = ['.','*','>','<']
fig, axes = plt.subplots(3, 1, figsize=(8, 5))

for type in range(len(si_pd_list)):
    si_pd = si_pd_list[type]
    si_pd_lt = si_pd[si_pd['session_type'] == 'lt']
    palette = ['purple', 'yellow']  # Define your own list of colors
    si_pd_sup = si_pd_lt[si_pd_lt['area'] == 'superficial']
    si_pd_deep = si_pd_lt[si_pd_lt['area'] == 'deep']

    for i in range(len(si_pd_sup)):
        axes[0].scatter(filters, si_pd_sup['si'].iloc[i], c = 'purple',marker=mark[type], alpha = 0.5)
        axes[0].plot(filters, si_pd_sup['si'].iloc[i], c = 'purple')
    for i in range(len(si_pd_deep)):
        axes[0].scatter(filters, si_pd_deep['si'].iloc[i], c = 'yellow',marker=mark[type], alpha = 0.5)
        axes[0].plot(filters, si_pd_deep['si'].iloc[i], c = 'yellow')

    for i in range(len(si_pd_sup)):
        axes[1].scatter(filters, si_pd_sup['si_pos'].iloc[i], c = 'purple',marker=mark[type], alpha = 0.5)
        axes[1].plot(filters, si_pd_sup['si_pos'].iloc[i], c = 'purple')
    for i in range(len(si_pd_deep)):
        axes[1].scatter(filters, si_pd_deep['si_pos'].iloc[i], c = 'yellow',marker=mark[type], alpha = 0.5)
        axes[1].plot(filters, si_pd_deep['si_pos'].iloc[i], c = 'yellow')

    for i in range(len(si_pd_sup)):
        axes[2].scatter(filters, si_pd_sup['si_dir'].iloc[i], c = 'purple',marker=mark[type], alpha = 0.5)
        axes[2].plot(filters, si_pd_sup['si_dir'].iloc[i], c = 'purple')
    for i in range(len(si_pd_deep)):
        axes[2].scatter(filters, si_pd_deep['si_dir'].iloc[i], c = 'yellow',marker=mark[type], alpha = 0.5)
        axes[2].plot(filters, si_pd_deep['si_dir'].iloc[i], c = 'yellow')

    for  i in range(3):
        axes[i].set_ylim([-0.1,1.1])
        axes[i].set_xlabel('Kernel size (ms)')

    axes[0].set_ylabel('SI Time')
    axes[1].set_ylabel('SI Pos')
    axes[2].set_ylabel('SI Dir')

plt.savefig(os.path.join(data_dir, f'SI_0-18_individual_{signal_name}.png'), dpi=400, bbox_inches="tight")


figures, axes = plt.subplots(3,1, figsize = (10,8))

superficial_colors = ['indigo','purple','violet','darkviolet']
deep_colors = ['gold','yellow','orange','olive']

for type in range(len(si_pd_list)):
    si_pd = si_pd_list[type]
    si_pd_lt = si_pd[si_pd['session_type'] == 'lt']
    palette = ['purple', 'yellow']  # Define your own list of colors
    si_pd_sup = si_pd_lt[si_pd_lt['area'] == 'superficial']
    si_pd_deep = si_pd_lt[si_pd_lt['area'] == 'deep']

    axes[0].errorbar(filters,np.mean(si_pd_sup['si']),np.std(np.array(si_pd_sup['si'])), c = superficial_colors[type],marker=mark[type], alpha = 1)
    axes[0].errorbar(filters,np.mean(si_pd_deep['si']),np.std(np.array(si_pd_sup['si'])), c = deep_colors[type],marker=mark[type], alpha = 1)

    axes[1].errorbar(filters,np.mean(si_pd_sup['si_pos']),np.std(np.array(si_pd_sup['si_pos'])), c = superficial_colors[type],marker=mark[type], alpha = 0.5)
    axes[1].errorbar(filters,np.mean(si_pd_deep['si_pos']),np.std(np.array(si_pd_sup['si_pos'])), c =deep_colors[type],marker=mark[type], alpha = 0.5)

    axes[2].errorbar(filters,np.mean(si_pd_sup['si_dir']),np.std(np.array(si_pd_sup['si_dir'])), c =superficial_colors[type],marker=mark[type], alpha = 0.5)
    axes[2].errorbar(filters,np.mean(si_pd_deep['si_dir']),np.std(np.array(si_pd_sup['si_dir'])), c =deep_colors[type],marker=mark[type], alpha = 0.5)

axes[2].set_xlabel('Filter size (ms)')

#axes[2].legend(['Superficial','Deep'])
axes[0].set_ylabel('SI Time')
axes[1].set_ylabel('SI Pos')
axes[2].set_ylabel('SI Dir')
axes[2].legend(['Superficial Complete Signal','Deep Complete Signal', 'Superficial UMAP','Deep UMAP',
                'Superficial 10PCS','Deep 10PCS',
                'Superficial UMAP10PCs', 'Deep UMAP10PCs',])


for  i in range(3):
    axes[i].set_ylim([-0.1,1.1])
plt.savefig(os.path.join(data_dir, f'SI_0_18_{signal_name}.png'), dpi=400, bbox_inches="tight")

########################################################################################

#############################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
#import ptitprince as pt
import numpy as np
import os
#from statsmodels.formula.api import ols
#import statsmodels.api as sm
from scipy import stats
import seaborn as sns
#[0,1,2,5,7,10,15,20,25,30,50]
filters = np.array([0,1,2,4,6,8,10,12,14,16,18])*50
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

#pickle_name = f"{mouse}_si_umap_10pcs_filters_0-18.pkl"
#lrgu.save_pickle(msave_dir, pickle_name, dynamics_dict)

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = data_dir
        si_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_si_complete_signal_filters_0-18.pkl")
        ##CZ3_si_filters_0-18.pkl
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
    axes[i].set_ylim([-0.1,1.1])
    axes[i].set_xlabel('Kernel size (ms)')

axes[0].set_ylabel('SI Time')
axes[1].set_ylabel('SI Pos')
axes[2].set_ylabel('SI Dir')

plt.savefig(os.path.join(data_dir, f'SI_filter_complete_signal_0-18_individual_{signal_name}.png'), dpi=400, bbox_inches="tight")


figures, axes = plt.subplots(3,1)
axes[0].errorbar(filters,np.mean(si_pd_sup['si']),np.std(np.array(si_pd_sup['si'])), c = 'purple')
axes[0].errorbar(filters,np.mean(si_pd_deep['si']),np.std(np.array(si_pd_sup['si'])), c = 'yellow')

axes[1].errorbar(filters,np.mean(si_pd_sup['si_pos']),np.std(np.array(si_pd_sup['si_pos'])), c = 'purple')
axes[1].errorbar(filters,np.mean(si_pd_deep['si_pos']),np.std(np.array(si_pd_sup['si_pos'])), c = 'yellow')

axes[2].errorbar(filters,np.mean(si_pd_sup['si_dir']),np.std(np.array(si_pd_sup['si_dir'])), c = 'purple')
axes[2].errorbar(filters,np.mean(si_pd_deep['si_dir']),np.std(np.array(si_pd_sup['si_dir'])), c = 'yellow')

axes[2].set_xlabel('Filter size (ms)')

axes[2].legend(['Superficial','Deep'])
axes[0].set_ylabel('SI Time')
axes[1].set_ylabel('SI Pos')
axes[2].set_ylabel('SI Dir')

for  i in range(3):
    axes[i].set_ylim([-0.1,1.1])
plt.savefig(os.path.join(data_dir, f'SI_complete_signal_mean_filter_0_18_{signal_name}.png'), dpi=400, bbox_inches="tight")

########################################################################################


