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
import matplotlib.pyplot as plt

base_dir =  '/home/melma31/Documents/deepsup_project/'
#__________________________________________________________________________
#|                                                                        |#
#|                           Signal Examples                              |#
#|________________________________________________________________________|#
data_dir = os.path.join(base_dir, 'data')
save_dir = os.path.join(base_dir, 'heatmap')
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

general_mice_list = {'lt_and_rot':{'Calb': [],
              'Chrna7': [],
              'Thy1':['GC2']},
              'only_lt':{
                  'Calb':[],
                  'Chrna7': [],
                   'Thy1':[]}
              }


signal_name = 'clean_traces'
models = ['Calb', 'Chrna7', 'Thy1']
R_summary = {}  # <- Store R values per mouse and segment

for coloring in coloring_condition:
    if coloring == 'single_color':
        paradigm =['lt_and_rot','only_lt']
        #paradigm =['lt_and_rot']

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





            from scipy.signal import find_peaks
            import matplotlib.pyplot as plt
            import numpy as np
            random_neuron = 10
            # Define the trial_ids we want to display
            trial_ids_to_show = [10, 11, 12,13,14,15,16,17,18,19,20]

            # Get the indices where trial_id is in the desired range
            trial_mask = np.isin(trial_id, trial_ids_to_show)

            # Apply the mask to filter position, speed, direction, and other signals
            position_filtered = pos[trial_mask,0]
            time_filtered = time[trial_mask]
            speed_filtered = speed[trial_mask]
            direction_filtered = mov_dir[trial_mask]
            spikes_matrix_filtered = signal[trial_mask, :]

            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.signal import find_peaks
            from scipy.ndimage import gaussian_filter1d

            fig, axs = plt.subplots(4, 1, figsize=(10, 20))
            time_filtered = np.arange(0, len(position_filtered))
            # Plot position
            axs[0].plot(time_filtered, position_filtered, label='Position', color='k')
            # Filtered signal with sigma=20
            filtered_signal_20 = gaussian_filter1d(spikes_matrix_filtered[:, random_neuron], sigma=20)
            norm_filtered_20 = (filtered_signal_20 - np.min(filtered_signal_20)) / (
                        np.max(filtered_signal_20) - np.min(filtered_signal_20) + 1e-9)
            axs[0].scatter(
                time_filtered,
                position_filtered,
                s=50 + 1000 * norm_filtered_20,
                c='yellow',
                alpha=norm_filtered_20,
                label='Filtered σ=20'
            )
            # Filtered signal with sigma=10
            filtered_signal_10 = gaussian_filter1d(spikes_matrix_filtered[:, random_neuron], sigma=10)
            norm_filtered_10 = (filtered_signal_10 - np.min(filtered_signal_10)) / (
                        np.max(filtered_signal_10) - np.min(filtered_signal_10) + 1e-9)
            axs[0].scatter(
                time_filtered,
                position_filtered,
                s=50 + 500 * norm_filtered_10,
                c='green',
                alpha=norm_filtered_10,
                label='Filtered σ=10'
            )
            # Filtered signal with sigma=10
            filtered_signal_10 = gaussian_filter1d(spikes_matrix_filtered[:, random_neuron], sigma=5)
            norm_filtered_10 = (filtered_signal_10 - np.min(filtered_signal_10)) / (
                        np.max(filtered_signal_10) - np.min(filtered_signal_10) + 1e-9)
            axs[0].scatter(
                time_filtered,
                position_filtered,
                s=50 + 250 * norm_filtered_10,
                c='blue',
                alpha=norm_filtered_10,
                label='Filtered σ=5'
            )
            # Filtered signal with sigma=1
            peaks, _ = find_peaks(spikes_matrix_filtered[:, random_neuron])
            axs[0].scatter(
                time_filtered[peaks],
                position_filtered[peaks],
                s=100,
                c='k',
                alpha=1,
                label='Peaks'
            )
            axs[0].set_title('Position Over Time (Trial ID = 10, 11, 12)')
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Position (m)')
            axs[0].set_xlim([0, len(position_filtered)])
            axs[0].legend()
            # Heatmaps
            filtered_spikes_1 = gaussian_filter1d(spikes_matrix_filtered.T, sigma=1, axis=1)
            cax1 = axs[1].imshow(filtered_spikes_1, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=5)
            axs[1].set_title('Heatmap (Filter = 1 ms)')
            axs[1].set_xlabel('Time (ms)')
            axs[1].set_ylabel('Neurons')
            axs[1].set_xticks([0, len(time_filtered) // 2, len(time_filtered) - 1])
            axs[1].set_xticklabels([0, len(time_filtered) // 2, len(time_filtered) - 1])
            filtered_spikes_10 = gaussian_filter1d(spikes_matrix_filtered.T, sigma=10, axis=1)
            cax2 = axs[2].imshow(filtered_spikes_10, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=5)
            axs[2].set_title('Heatmap (Filter = 10 ms)')
            axs[2].set_xlabel('Time (ms)')
            axs[2].set_ylabel('Neurons')
            axs[2].set_xticks([0, len(time_filtered) // 2, len(time_filtered) - 1])
            axs[2].set_xticklabels([0, len(time_filtered) // 2, len(time_filtered) - 1])
            filtered_spikes_20 = gaussian_filter1d(spikes_matrix_filtered.T, sigma=20, axis=1)
            cax3 = axs[3].imshow(filtered_spikes_20, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=5)
            axs[3].set_title('Heatmap (Filter = 20 ms)')
            axs[3].set_xlabel('Time (ms)')
            axs[3].set_ylabel('Neurons')
            axs[3].set_xticks([0, len(time_filtered) // 2, len(time_filtered) - 1])
            axs[3].set_xticklabels([0, len(time_filtered) // 2, len(time_filtered) - 1])
            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory + 'heatmap_example.svg'))
            plt.show()


            fig, axs = plt.subplots(4, 1, figsize=(10, 20))
            time_filtered = np.arange(0, len(position_filtered))
            # Plot position
            axs[0].plot(time_filtered, position_filtered, label='Position', color='k')
            # Filtered signal with sigma=20
            filtered_signal_20 = gaussian_filter1d(spikes_matrix_filtered[:, random_neuron], sigma=20)
            norm_filtered_20 = (filtered_signal_20 - np.min(filtered_signal_20)) / (
                        np.max(filtered_signal_20) - np.min(filtered_signal_20) + 1e-9)
            axs[0].scatter(
                time_filtered,
                position_filtered,
                s=50 + 1000 * norm_filtered_20,
                c='yellow',
                alpha=norm_filtered_20,
                label='Filtered σ=20'
            )
            # Filtered signal with sigma=10
            filtered_signal_10 = gaussian_filter1d(spikes_matrix_filtered[:, random_neuron], sigma=10)
            norm_filtered_10 = (filtered_signal_10 - np.min(filtered_signal_10)) / (
                        np.max(filtered_signal_10) - np.min(filtered_signal_10) + 1e-9)
            axs[0].scatter(
                time_filtered,
                position_filtered,
                s=50 + 500 * norm_filtered_10,
                c='green',
                alpha=norm_filtered_10,
                label='Filtered σ=10'
            )
            # Filtered signal with sigma=10
            filtered_signal_10 = gaussian_filter1d(spikes_matrix_filtered[:, random_neuron], sigma=5)
            norm_filtered_10 = (filtered_signal_10 - np.min(filtered_signal_10)) / (
                        np.max(filtered_signal_10) - np.min(filtered_signal_10) + 1e-9)
            axs[0].scatter(
                time_filtered,
                position_filtered,
                s=50 + 250 * norm_filtered_10,
                c='blue',
                alpha=norm_filtered_10,
                label='Filtered σ=5'
            )
            # Filtered signal with sigma=1
            peaks, _ = find_peaks(spikes_matrix_filtered[:, random_neuron])
            axs[0].scatter(
                time_filtered[peaks],
                position_filtered[peaks],
                s=100,
                c='k',
                alpha=1,
                label='Peaks'
            )
            axs[0].set_title('Position Over Time (Trial ID = 10, 11, 12)')
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Position (m)')
            axs[0].set_xlim([0, len(position_filtered)])
            axs[0].legend()
            # Heatmaps
            filtered_spikes_1 = gaussian_filter1d(spikes_matrix_filtered.T, sigma=1, axis=1)
            cax1 = axs[1].imshow(filtered_spikes_1, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=5)
            axs[1].set_title('Heatmap (Filter = 1 ms)')
            axs[1].set_xlabel('Time (ms)')
            axs[1].set_ylabel('Neurons')
            axs[1].set_xticks([0, len(time_filtered) // 2, len(time_filtered) - 1])
            axs[1].set_xticklabels([0, len(time_filtered) // 2, len(time_filtered) - 1])
            filtered_spikes_10 = gaussian_filter1d(spikes_matrix_filtered.T, sigma=10, axis=1)
            cax2 = axs[2].imshow(filtered_spikes_10, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=5)
            axs[2].set_title('Heatmap (Filter = 10 ms)')
            axs[2].set_xlabel('Time (ms)')
            axs[2].set_ylabel('Neurons')
            axs[2].set_xticks([0, len(time_filtered) // 2, len(time_filtered) - 1])
            axs[2].set_xticklabels([0, len(time_filtered) // 2, len(time_filtered) - 1])
            filtered_spikes_20 = gaussian_filter1d(spikes_matrix_filtered.T, sigma=20, axis=1)
            cax3 = axs[3].imshow(filtered_spikes_20, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=5)
            axs[3].set_title('Heatmap (Filter = 20 ms)')
            axs[3].set_xlabel('Time (ms)')
            axs[3].set_ylabel('Neurons')
            axs[3].set_xticks([0, len(time_filtered) // 2, len(time_filtered) - 1])
            axs[3].set_xticklabels([0, len(time_filtered) // 2, len(time_filtered) - 1])
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir + 'heatmap_example.svg'))
            plt.show()
