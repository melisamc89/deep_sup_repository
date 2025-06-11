import os
import sys
import copy

import time
import timeit
import os, copy

import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

def return_dir_color(dir_mat):
    dir_color = np.zeros((dir_mat.shape[0],3))
    for point in range(dir_mat.shape[0]):
        if dir_mat[point]==0:
            dir_color[point] = [14/255,14/255,143/255]
        elif dir_mat[point]==1:
            dir_color[point] = [12/255,136/255,249/255]
        else:
            dir_color[point] = [17/255,219/255,224/255]
    return dir_color


base_dir =  '/home/melma31/Documents/deepsup_project/'

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'

data_dir = os.path.join(base_dir, 'MIR')
save_dir = os.path.join(base_dir, 'SI_Filters')
if not os.path.isdir(save_dir): os.makedirs(save_dir)

dim = 3
si_neigh = 50

init_view1 = 30
init_view2 = 30
mice_dict = {'deep':['GC2']}
mice_area = list(mice_dict.keys())

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
                mov_dir_color = return_dir_color(mov_dir)
                signal = mi_dict[session]['signal']
                signal = signal[valid_index, :]
                noise_idx, signal_idx = filter_noisy_outliers(signal)

                row = 3
                col = 5
                fig = plt.figure(figsize=(25, 9))

                # compute umap
                umap_model = umap.UMAP(n_neighbors=120, n_components=dim, min_dist=0.1, random_state=42)
                umap_model.fit(signal[valid_index, :])
                umap_emb = umap_model.transform(signal[valid_index, :])

                ax = fig.add_subplot(row, col, 1, projection='3d')
                b = ax.scatter(*umap_emb[:, :3].T, color=mov_dir_color, s=1, alpha=0.5)
                ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
                ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
                ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
                ax.set_title('UMAP Direction')
                ax.view_init(0, 0)  # Set the view angle

                ax = fig.add_subplot(row, col, 6, projection='3d')
                b = ax.scatter(*umap_emb[:, :3].T, c=beh_variables['pos'], s=1, alpha=0.5, cmap = 'magma')
                ax.set_title('UMAP Position')
                ax.view_init(0, 0)  # Set the view angle

                ax = fig.add_subplot(row, col, 11, projection='3d')
                b = ax.scatter(*umap_emb[:, :3].T, c=beh_variables['time'], s=1, alpha=0.5,cmap = 'Greens')
                ax.set_title('UMAP Time')
                ax.view_init(0, 0)  # Set the view angle

                kernels = [4, 8, 14,20]
                for index, filter_size in enumerate(kernels):
                    rates = smooth_calcium_signals(signal, filter_size)
                    umap_model = umap.UMAP(n_neighbors=120, n_components=dim, min_dist=0.1, random_state=42)
                    umap_model.fit(rates)
                    umap_emb = umap_model.transform(rates)
                    ax = fig.add_subplot(row, col, index + 2 , projection='3d')
                    b = ax.scatter(*umap_emb[:, :3].T, color=mov_dir_color, s=1, alpha=0.5)
                    ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
                    ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
                    ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
                    ax.set_title('UMAP Direction')
                    ax.view_init(init_view1, init_view2)  # Set the view angle

                    ax = fig.add_subplot(row, col, index + 7, projection='3d')
                    b = ax.scatter(*umap_emb[:, :3].T, c=beh_variables['pos'], s=1, alpha=0.5,cmap = 'magma')
                    ax.set_title('UMAP Position')
                    ax.view_init(init_view1, init_view2)  # Set the view angle

                    ax = fig.add_subplot(row, col, index + 12, projection='3d')
                    b = ax.scatter(*umap_emb[:, :3].T, c=beh_variables['time'], s=1, alpha=0.5, cmap = 'Greens')
                    ax.set_title('UMAP Time')
                    ax.view_init(init_view1, init_view2)  # Set the view angle

                fig.tight_layout()

                fig.savefig(os.path.join(msave_dir, f"{mouse}_umap_filters_{signal_name}_{session}_{init_view1}_{init_view2}.png"), dpi=400,
                            bbox_inches="tight")