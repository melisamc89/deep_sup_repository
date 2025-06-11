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


base_dir =  '/home/melma31/Documents/deepsup_project/'

#__________________________________________________________________________
#|                                                                        |#
#|                           STRUCTURE INDEX                              |#
#|________________________________________________________________________|#

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

params1 = {
    'n_bins': 10,
    'discrete_label': False,
    'continuity_kernel': None,
    'perc_neigh': 1,
    'num_shuffles': 0,
    'verbose': False
}

si_beh_params = {}
for beh in ['time']:
    si_beh_params[beh] = copy.deepcopy(params1)

signal_name = 'clean_traces'
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

                params = {
                    'signal_name': signal_name,
                    'n_neighbors': 50,
                }

                si_dict = dict()

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

                    si_dict[maze] = dict()
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

                    #clean mov dir to avoid gaps in ring
                    temp_mov = np.diff(pos,axis=0)*20
                    temp_mov = np.concatenate([temp_mov[0,:].reshape(-1,2), temp_mov], axis= 0)[:,0]
                    temp_mov = gaussian_filter1d(temp_mov, sigma = 5, axis = 0)
                    temp_mov[temp_mov<0] = -1
                    temp_mov = np.sign(temp_mov)
                    mov_dir[mov_dir==0] = temp_mov[np.where(mov_dir==0)[0]]

                    valid_index = mov_dir!=0
                    beh_variables = {
                    }
                    signal = lrgu.get_signal(session_pd, signal_name)[valid_index,:]
                    #compute umap
                    umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist,                                                              random_state=42)
                    umap_model.fit(signal)
                    umap_emb = umap_model.transform(signal)

                    time = np.arange(0,len(signal))
                    noise_idx, signal_idx = filter_noisy_outliers(signal)
                    fake_time = time[signal_idx]
                    beh_variables['time'] = fake_time
                    print(f"Working on session {maze}:", end='')
                    for beh_name, beh_val in beh_variables.items():
                        si_beh_params[beh_name]['n_neighbors'] = 50
                        si, process_info, overlap_mat, _ = compute_structure_index(signal[signal_idx],
                                                                                   beh_val,
                                                                                   **si_beh_params[beh_name])

                        si_umap, process_info, overlap_mat, _ = compute_structure_index(umap_emb[signal_idx],
                                                                                   beh_val,
                                                                                   **si_beh_params[beh_name])

                        si_dict[maze][beh_name] = {
                        'si': copy.deepcopy(si),
                        'process_info': copy.deepcopy(process_info),
                        'overlap_mat': copy.deepcopy(overlap_mat),
                        'beh_params': copy.deepcopy(si_beh_params[beh_name]),
                        'valid_idx': valid_index.copy(),
                        'signal_idx': signal_idx.copy(),
                        'signal': signal.copy(),
                        'si_umap': copy.deepcopy(si),
                        'umap_signal': umap_emb.copy(),
                        }

                        print(f" {beh_name}={si:.4f} |", end='', sep = '', flush='True')

                lrgu.save_pickle(msave_dir,f"{mouse}_si_{signal_name}_dict.pkl", si_dict)
                lrgu.print_time_verbose(local_time, global_time)
                sys.stdout = original
                f.close()



#__________________________________________________________________________
#|                                                                        |#
#|                                PLOT SI                                 |#
#|________________________________________________________________________|#

import pandas as pd
import matplotlib.pyplot as plt
#import ptitprince as pt

#from statsmodels.formula.api import ols
#import statsmodels.api as sm
from scipy import stats
import seaborn as sns


data_dir = os.path.join(base_dir, 'results')

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'


mouse_name_list = []
si_time_list = []
si_umap_time_list = []
area_list = []
session_list = []

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = data_dir
        si_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_si_{signal_name}_dict.pkl")
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
            si_time_list.append(si_dict[session]['time']['si'])
            si_umap_time_list.append(si_dict[session]['time']['si_umap'])

si_pd = pd.DataFrame(data={'mouse': mouse_name_list,
                           'area': area_list,
                     'session_type': session_list,
                     'si_time': si_time_list,
                     'si_time_umap': si_umap_time_list})

si_pd_lt = si_pd[si_pd['session_type'] == 'lt']
palette = ['purple', 'yellow']  # Define your own list of colors
fig, axes = plt.subplots(1,2, figsize = (10,5))
sns.barplot(si_pd_lt, hue="area", y="si_time", ax=axes[0], palette = palette, legend= False)
sns.stripplot(si_pd_lt,hue="area", y="si_time",palette = 'dark:k', dodge=True, size=5, jitter=True, ax = axes[0],legend=False)
sns.barplot(si_pd_lt, hue="area", y="si_time_umap", ax=axes[1], palette = palette, legend = False)
sns.stripplot(si_pd_lt,hue="area", y="si_time_umap",palette = 'dark:k', dodge=True, size=5, jitter=True, ax=axes[1],legend=False)

plt.savefig(os.path.join(data_dir, f'SI_{signal_name}.png'), dpi=400, bbox_inches="tight")
