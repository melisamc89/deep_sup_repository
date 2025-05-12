import os
import sys
import copy
import umap

import time
import timeit
from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt
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
save_dir = os.path.join(base_dir, 'time_sequences')

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
                    valid_index = np.arange(0,mov_dir.shape[0])
                    trial_id = lrgu.get_signal(session_pd, 'trial_id_mat')
                    time = np.arange(0,pos.shape[0])

                    signal = lrgu.get_signal(session_pd, signal_name)
    # --- Correlation-based sorting ---
                    corr_matrix = np.corrcoef(signal.T)  # shape: (n_cells, n_cells)
                    i, j = np.unravel_index(np.argmax(corr_matrix - np.eye(corr_matrix.shape[0])), corr_matrix.shape)
                    seed_cell = i  # choose one of the two
                    corr_with_seed = corr_matrix[seed_cell]
                    corr_order = np.argsort(corr_with_seed)[::-1]  # descending order


                    # --- PCA-based sorting ---
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(signal.T)  # shape: (n_cells, 2)
                    angles = np.arctan2(pca_result[:, 1], pca_result[:, 0])  # angle in radians
                    pca_order = np.argsort(angles)[::-1]

                    # Time vector in seconds
                    time_vector = np.arange(0, signal.shape[0]) / 20  # assuming 20 Hz sampling
                    # Get trial transition times (in seconds)
                    # Flatten trial IDs and find transitions
                    trial_id = trial_id.flatten()
                    trial_transitions = np.where(np.diff(trial_id) != 0)[0]

                    # Keep every third transition (assuming right → stop1 → left → right...)
                    cycle_starts = trial_transitions[::3]
                    cycle_times = cycle_starts / 20  # convert to seconds

                    # --- PCA Sorted Plot ---
                    plt.figure(figsize=(12, 6))
                    plt.imshow(signal[:, pca_order].T, aspect='auto', cmap='binary',
                               extent=[time_vector[0], time_vector[-1], 0, signal.shape[1]])

                    ## Vertical lines at cycle start times
                    #for t in cycle_times:
                    #    plt.axvline(x=t, color='red', linestyle='--', linewidth=0.5)

                    plt.title(f"{mouse} - PCA Sorted Activity")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Neuron (sorted by PCA)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(msave_dir, f"{mouse}_{idx}_pca_sorted_heatmap.png"))
                    plt.show()

                    # --- Corr Sorted Plot ---
                    plt.figure(figsize=(12, 6))
                    plt.imshow(signal[:, corr_order].T, aspect='auto', cmap='binary',
                               extent=[time_vector[0], time_vector[-1], 0, signal.shape[1]])

                    for t in cycle_times:
                        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.5)

                    plt.title(f"{mouse} - Corr Sorted Activity")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Neuron (sorted by Corr)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(msave_dir, f"{mouse}_{idx}_corr_sorted_heatmap.png"))
                    plt.show()
