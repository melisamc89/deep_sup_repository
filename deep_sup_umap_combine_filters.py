import os
import sys
import copy
import time
import timeit
import umap
import numpy as np
import matplotlib.pyplot as plt
import learning_repo.general_utils as lrgu
import learning_repo.geometric_utils as lrgeo
from structure_index import compute_structure_index
from sklearn.metrics import pairwise_distances
from scipy.ndimage import gaussian_filter1d

def smooth_calcium_signals(calcium_data, sigma=4):
    smoothed_signals = np.zeros_like(calcium_data)
    for i in range(calcium_data.shape[1]):
        smoothed_signals[:, i] = gaussian_filter1d(calcium_data[:, i], sigma=sigma)
    return smoothed_signals

def filter_noisy_outliers(data):
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D, 1), axis=1) - 1
    noiseIdx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    signalIdx = np.where(nn_dist >= np.percentile(nn_dist, 20))[0]
    return noiseIdx, signalIdx

def return_dir_color(dir_mat):
    dir_color = np.zeros((dir_mat.shape[0], 3))
    for point in range(dir_mat.shape[0]):
        if dir_mat[point] == 0:
            dir_color[point] = [14/255, 14/255, 143/255]
        elif dir_mat[point] == 1:
            dir_color[point] = [12/255, 136/255, 249/255]
        else:
            dir_color[point] = [17/255, 219/255, 224/255]
    return dir_color

base_dir = '/home/melma31/Documents/deepsup_project/'
mice_dict = { 'deep': ['GC2']}
signal_name = 'clean_traces'
data_dir = os.path.join(base_dir, 'MIR')
save_dir = os.path.join(base_dir, 'transfer_Filters')
os.makedirs(save_dir, exist_ok=True)

dim = 3
init_view1 = 30
init_view2 = 30

for area in mice_dict:
    for mouse in mice_dict[area]:
        print(mouse)
        global_time = timeit.default_timer()
        mdata_dir = os.path.join(data_dir, mouse)
        msave_dir = os.path.join(save_dir, mouse)
        os.makedirs(msave_dir, exist_ok=True)
        f = open(os.path.join(msave_dir, f"{mouse}_linearity_{signal_name}_logFile.txt"), 'w')
        sys.stdout = lrgu.Tee(sys.stdout, f)

        mi_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_mi_cluster_all_{signal_name}_dict.pkl")
        session_names = sorted(mi_dict.keys())

        for idx, session in enumerate(session_names):
            if 'lt' not in session:
                continue

            print(f"Working on session {session} ({idx + 1}/{len(session_names)}): ", sep='', end='')
            beh = mi_dict[session]['behaviour']
            valid_index = np.arange(0, beh['mov_dir'].shape[0])
            beh_variables = {
                'pos': beh['position'][valid_index].copy(),
                'time': beh['time'][valid_index].copy()
            }

            signal = mi_dict[session]['signal'][valid_index, :]
            kernels = [4, 20]

            smoothed_signals_dict = {}
            for filt in kernels:
                if filt == 0:
                    smoothed_signals_dict[filt] = signal.copy()
                else:
                    smoothed_signals_dict[filt] = smooth_calcium_signals(signal, filt)

            # ---- Concatenate all filters ----
            concatenated_signal = np.vstack([
                smoothed_signals_dict[filt] for filt in kernels
            ])
            color_data_concat = np.concatenate([
                beh_variables[cmap_type] for _ in kernels
            ])

            umap_model = umap.UMAP(
                n_neighbors=120,
                n_components=dim,
                min_dist=0.1,
                random_state=42
            )
            umap_embedding = umap_model.fit_transform(concatenated_signal)

            # ---- 3D plot ----
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                umap_embedding[:, 0],
                umap_embedding[:, 1],
                umap_embedding[:, 2],
                c=color_data_concat,
                cmap=cmap,
                s=1,
                alpha=0.5
            )
            ax.set_title(f"UMAP on Concatenated Filters - Colored by {label}")
            ax.view_init(init_view1, init_view2)
            fig.savefig(
                os.path.join(msave_dir, f"{mouse}_umap_concatfilters_{signal_name}_{session}_{cmap_type}.png"),
                dpi=400
            )
            plt.close(fig)
        sys.stdout = sys.__stdout__
        f.close()

