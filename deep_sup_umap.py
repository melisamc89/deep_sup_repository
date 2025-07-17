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
#__________________________________________________________________________
#|                                                                        |#
#|                           Mutual Information (MIR)                     |#
#|________________________________________________________________________|#
data_dir = os.path.join(base_dir, 'data')
save_dir = os.path.join(base_dir, 'umap_embeddings')

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
dim = 3

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

                umap_dict = dict()

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
                    umap_dict[maze] = dict()
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

                    behaviours_list = [pos[:,0], pos_dir, mov_dir, speed, time, inner_time, trial_id]
                    beh_names = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID']
                    behaviour_dict = {
                        'position':behaviours_list[0][valid_index],
                        '(pos,dir)':behaviours_list[1][valid_index],
                        'mov_dir':behaviours_list[2][valid_index],
                        'speed':behaviours_list[3][valid_index],
                        'time':behaviours_list[4][valid_index],
                        'inner_time':behaviours_list[5][valid_index],
                        'trial_id':behaviours_list[6][valid_index]
                    }
                    signal = lrgu.get_signal(session_pd, signal_name)
                    noise_idx, signal_idx = filter_noisy_outliers(signal)

                    #compute umap
                    umap_model = umap.UMAP(n_neighbors=120, n_components=dim, min_dist=0.1,                                                              random_state=42)
                    umap_model.fit(signal[valid_index,:])
                    umap_emb = umap_model.transform(signal[valid_index,:])

                    umap_dict[maze] = {
                        'beh_variables': copy.deepcopy(behaviour_dict),
                        'signal': signal.copy(),
                        'umap': umap_emb
                    }
                lrgu.save_pickle(msave_dir, f"{mouse}_umap_{dim}_{signal_name}_dict.pkl", umap_dict)
                lrgu.print_time_verbose(local_time, global_time)
                sys.stdout = original


##############################################
# Plot umaps#
##############################################

import matplotlib.pyplot as plt

data_dir = os.path.join(base_dir, 'SI')
mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'

init_view1 = 30
init_view2 = 60

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        # Load your previously saved si_dict
        msave_dir = os.path.join(save_dir, mouse)  # mouse save dir
        if not os.path.isdir(msave_dir): os.makedirs(msave_dir)

        umap_dict = lrgu.load_pickle(os.path.join(save_dir, mouse),
                                   f"{mouse}_umap_{dim}_{signal_name}_dict.pkl")

        session_names = list(umap_dict.keys())
        for session in session_names:
            if 'lt' in session :
                beh_variables = umap_dict[session]['beh_variables']
                umap_session = umap_dict[session]['umap']

                pos = beh_variables['position']
                speed = beh_variables['speed']
                trial_id_mat = beh_variables['trial_id']
                mov_dir = beh_variables['mov_dir']
                time = beh_variables['time']
                mov_dir_color = return_dir_color(mov_dir)

                ### define figure size
                row = 1
                col = 5
                fig = plt.figure(figsize=(15, 5))
                ax = fig.add_subplot(row, col, 1, projection='3d')
                umap_emb = umap_dict[session]['umap']
                ax.set_title('Position')
                ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=pos, s=0.5, alpha=0.5, cmap='magma')
                ax.grid(False)
                ax.view_init(init_view1, init_view2)  # Set the view angle

                ax = fig.add_subplot(row, col, 2, projection='3d')
                ax.set_title('Direction')
                ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], color=mov_dir_color, s=0.5, alpha=0.5)
                ax.grid(False)
                ax.view_init(init_view1, init_view2)  # Set the view angle

                #ax = fig.add_subplot(row, col, 3 , projection='3d')
                #ax.set_title('Speed')
                #ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=speed, s=0.5, alpha=0.05, cmap='Reds')
                #ax.grid(False)
                #ax.view_init(init_view1, init_view2)  # Set the view angle

                ax = fig.add_subplot(row, col, 3 , projection='3d')
                ax.set_title('Time')
                ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=np.arange(0, umap_emb.shape[0]), s=0.5,
                                       alpha=0.5, cmap='YlGn_r')
                ax.grid(False)
                ax.view_init(init_view1, init_view2)  # Set the view angle

                #ax = fig.add_subplot(row, col, 5 , projection='3d')
                #ax.set_title('Trial ID ')
                #ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=trial_id_mat, s=0.05, alpha=0.5, cmap='viridis')
                #ax.grid(False)
                #ax.view_init(init_view1, init_view2)  # Set the view angle
                fig.tight_layout()

                fig.savefig(os.path.join(msave_dir, f"{mouse}_umap_{signal_name}_{session}_{init_view1}_{init_view2}.png"), dpi=400,
                        bbox_inches="tight")
                fig.savefig(os.path.join(msave_dir, f"{mouse}_umap_{signal_name}_{session}_{init_view1}_{init_view2}.svg"), dpi=400,
                        bbox_inches="tight")


