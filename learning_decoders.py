import os
import sys
import copy

import time
import timeit 
from datetime import datetime

import numpy as np 

import learning_repo.general_utils as lrgu
from learning_repo.decoders import decoders_1D
from structure_index import compute_structure_index

base_dir =  '/home/julio/Documents/learning_project/'

#__________________________________________________________________________
#|                                                                        |#
#|                               DECODERS                                 |#
#|________________________________________________________________________|#


data_dir = os.path.join(base_dir, 'processed_data')
save_dir = os.path.join(base_dir, 'decoders')
if not os.path.isdir(save_dir): os.makedirs(save_dir)

mice_list = ['M2019', 'M2021', 'M2022', 'M2023', 'M2024', 'M2025', 'M2026']

for (case, cells,signal_name) in [(x,y,z) for x in ['mov', 'mov_same_length', 'all', 'all_same_length']
                                     for y in ['same_cells', 'all_cells'] 
                                     for z  in ['clean_traces']]:

    for mouse in mice_list:
        print()

        global_time = timeit.default_timer()     #initialize time
        local_time = timeit.default_timer()

        mdata_dir = os.path.join(data_dir, mouse) #mouse data dir
        msave_dir = os.path.join(save_dir, mouse) #mouse save dir
        if not os.path.isdir(msave_dir): os.makedirs(msave_dir)

        # f = open(os.path.join(msave_dir, f"{mouse}_linearity_{case}_{cells}_{signal_name}_logFile.txt"), 'w')
        # original = sys.stdout
        # sys.stdout = lrgu.Tee(sys.stdout, f)

        print(f"\t#### {mouse}: {case} | {cells} | {signal_name} ####")
        print(f'\t{datetime.now():%Y-%m-%d %H:%M}\n')

        params = {
            'signal_name': signal_name,
            'case': case,
            'cells': cells,
        }

        dec_dict = dict()

        file_name =  mouse+'_'+case+'_df_dict.pkl'
        mouse_dict = lrgu.load_pickle(mdata_dir,file_name)

        session_names = list(mouse_dict.keys())
        session_names.sort()

        for idx, session in enumerate(session_names):
            print(f"Working on session {session} ({idx+1}/{len(session_names)}):", end='')

            dec_dict[session] = dict()
            mov_dir = lrgu.get_signal(mouse_dict[session], 'mov_direction').copy()
            valid_index = mov_dir!=0


            beh_variables = {
                'posx': lrgu.get_signal(mouse_dict[session], 'position')[valid_index,0].copy(),
                'speed': lrgu.get_signal(mouse_dict[session], 'speed')[valid_index].copy(),
                'trial_id_mat': lrgu.get_signal(mouse_dict[session], 'trial_id_mat')[valid_index].copy(),
                'dir': lrgu.get_signal(mouse_dict[session], 'mov_direction')[valid_index].copy()
            }

            trial_id = lrgu.get_signal(mouse_dict[session], 'trial_id_mat')[valid_index].copy()

            signal = lrgu.get_signal(mouse_dict[session], signal_name)[valid_index,:]
            if params['cells']=='same_cells':         #keep only same cells across days
                signal = signal[:, mouse_dict[session]['global_id'][0]]

            for beh_name, beh_val  in beh_variables.items():

                if beh_name in ['posx', 'speed','trial_id_mat']: 
                    decoder_list = ["wf", "xgb"]
                    metric = "median_absolute_error"
                else:
                    decoder_list = ["svc"]
                    metric = "f1_score"

                R2s, pred = decoders_1D(signal, beh_val, decoder_list=decoder_list, trial_signal = trial_id, metric = metric, n_splits=10)

                dec_dict[session][beh_name] = {
                    'R2s': copy.deepcopy(R2s),
                    'pred': copy.deepcopy(pred),
                    'beh_val': copy.deepcopy(beh_val),
                    'signal': copy.deepcopy(signal),
                    'params': copy.deepcopy(params),
                }


                print(f" {beh_name}={np.nanmean(R2s['base_signal'][decoder_list[0]]):.4f} |", end='', sep = '', flush='True')
            print()
        lrgu.save_pickle(msave_dir,f"{mouse}_dec_{case}_{cells}_{signal_name}_dict.pkl", dec_dict)
        lrgu.print_time_verbose(local_time, global_time)
        sys.stdout = original
        f.close()


#__________________________________________________________________________
#|                                                                        |#
#|                             PLOT  DECODERS                             |#
#|________________________________________________________________________|#


import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt

from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats
import seaborn as sns

mouse_name_list = []
time_diff_list = []
session_idx_list = []

dec_posx_list= []
dec_speed_list = []
dec_dir_list = []
dec_time_list = []


mice_list = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
# mice_list = ['M2021', 'M2022']
save_dir = os.path.join(base_dir, 'decoders')

case = 'mov_same_length'
cells = 'same_cells'
signal_name = 'clean_traces'
for mouse in mice_list:

    msave_dir = os.path.join(save_dir, mouse) #mouse save dir
    dec_dict = lrgu.load_pickle(msave_dir,f"{mouse}_dec_{case}_{cells}_{signal_name}_dict.pkl")

    session_names = list(dec_dict.keys())
    session_names.sort()

    d0 = datetime.strptime(session_names[0][:8], "%Y%m%d")
    for idx, session in enumerate(session_names):
        new_date = datetime.strptime(session[:8], "%Y%m%d")
        days_diff = (new_date-d0).days
        if idx==2:
            days_diff += 0.5

        mouse_name_list.append(mouse)
        time_diff_list.append(days_diff)
        session_idx_list.append(idx)

        dec_posx_list.append(np.nanmean(dec_dict[session]['posx']['R2s']['base_signal']['xgb'][:,0,0]))
        dec_dir_list.append(np.nanmean(dec_dict[session]['dir']['R2s']['base_signal']['svc'][:,0,0]))
        dec_speed_list.append(np.nanmean(dec_dict[session]['speed']['R2s']['base_signal']['xgb'][:,0,0]))
        dec_time_list.append(np.nanmean(dec_dict[session]['trial_id_mat']['R2s']['base_signal']['xgb'][:,0,0]))


dec_pd = pd.DataFrame(data={'mouse': mouse_name_list,
                     'session_day': time_diff_list,
                     'session_idx': session_idx_list,
                     'dec_posx': dec_posx_list,
                     'dec_dir': dec_dir_list,
                     'dec_speed': dec_speed_list,
                     'dec_time': dec_time_list})   


fig = plt.figure(figsize=(10,6))

limits = {
    'dec_posx': [0, 14],
    'dec_speed': [0, 8],
    'dec_time': [0, 22],
    'dec_dir': [0.65, 1]
}
for idx, ylabel in enumerate(['dec_posx', 'dec_speed', 'dec_time', 'dec_dir']):

    ax = plt.subplot(1,4, idx+1)
    sns.barplot(x = 'session_day', y = ylabel, data = dec_pd, ax = ax)
    x_pos = {
        0: 0,
        1: 1,
        1.5: 2,
        3: 3,
        6: 4
    }

    for mouse in dec_pd['mouse'].unique():
        mouse_pd = dec_pd.loc[dec_pd['mouse']==mouse]
        x = [x_pos[i] for i in mouse_pd['session_day'].to_list()]
        y = mouse_pd[ylabel].to_list()
        ax.plot(x, y, color=[.3,.3,.3], alpha = 0.3, linewidth = 0.5)
        ax.scatter(x,y, color=[0,0,0], s=10)
    ax.set_ylim(limits[ylabel])
fig.suptitle(f'dec_{case}_{cells}_{signal_name}')
plt.savefig(os.path.join(save_dir, f'dec_day_{case}_{cells}_{signal_name}.png'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir, f'dec_day_{case}_{cells}_{signal_name}.svg'), dpi = 400,bbox_inches="tight")


fig = plt.figure(figsize=(10,6))

limits = {
    'dec_posx': [0, 14],
    'dec_speed': [0, 8],
    'dec_time': [0, 22],
    'dec_dir': [0.65, 1]
}
for idx, ylabel in enumerate(['dec_posx', 'dec_speed', 'dec_time', 'dec_dir']):

    ax = plt.subplot(1,4, idx+1)
    sns.barplot(x = 'session_idx', y = ylabel, data = dec_pd, 
                ci = 'sd', ax = ax)

    for mouse in dec_pd['mouse'].unique():
        mouse_pd = dec_pd.loc[dec_pd['mouse']==mouse]
        x = mouse_pd['session_idx'].to_list()
        y = mouse_pd[ylabel].to_list()
        ax.plot(x, y, color=[.3,.3,.3], alpha = 0.3, linewidth = 0.5)
        ax.scatter(x,y, color=[0,0,0], s=10)

    ax.set_ylim(limits[ylabel])

    for d0 in range(4):
        for d1 in range(d0+1,4):

            v0 = dec_pd.loc[dec_pd['session_idx']==d0][ylabel].to_list()
            v1 = dec_pd.loc[dec_pd['session_idx']==d1][ylabel].to_list()

            if (stats.shapiro(v0).pvalue<=0.5) or (stats.shapiro(v1).pvalue<=0.5):
                print(f'{ylabel} {d0}|{d1} ks_2samp p={stats.ks_2samp(v0, v1)[1]:.4f}')
            else:
                print(f'{ylabel} {d0}|{d1}ttest_rel p={stats.ttest_rel(v0, v1)[1]:.4f}')


fig.suptitle(f'dec_{case}_{cells}_{signal_name}')
plt.savefig(os.path.join(save_dir, f'dec_idx_{case}_{cells}_{signal_name}.png'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_dir, f'dec_idx_{case}_{cells}_{signal_name}.svg'), dpi = 400,bbox_inches="tight")





#__________________________________________________________________________
#|                                                                        |#
#|                               DECODERS                                 |#
#|________________________________________________________________________|#


data_dir = os.path.join(base_dir, 'processed_data')
save_dir = os.path.join(base_dir, 'decoders')
if not os.path.isdir(save_dir): os.makedirs(save_dir)

mice_list = ['M2019', 'M2021', 'M2022', 'M2023', 'M2024', 'M2025', 'M2026']

for (case, cells,signal_name) in [(x,y,z) for x in ['mov', 'mov_same_length', 'all', 'all_same_length']
                                     for y in ['same_cells', 'all_cells'] 
                                     for z  in ['clean_traces']]:
    case = 'mov_same_length'
    cells = 'same_cells'
    signal_name = 'clean_traces'
    for mouse in mice_list:
        print()

        global_time = timeit.default_timer()     #initialize time
        local_time = timeit.default_timer()

        mdata_dir = os.path.join(data_dir, mouse) #mouse data dir
        msave_dir = os.path.join(save_dir, mouse) #mouse save dir
        if not os.path.isdir(msave_dir): os.makedirs(msave_dir)

        # f = open(os.path.join(msave_dir, f"{mouse}_linearity_{case}_{cells}_{signal_name}_logFile.txt"), 'w')
        # original = sys.stdout
        # sys.stdout = lrgu.Tee(sys.stdout, f)

        print(f"\t#### {mouse}: {case} | {cells} | {signal_name} ####")
        print(f'\t{datetime.now():%Y-%m-%d %H:%M}\n')

        params = {
            'signal_name': signal_name,
            'case': case,
            'cells': cells,
        }

        dec_dict = dict()

        file_name =  mouse+'_'+case+'_df_dict.pkl'
        mouse_dict = lrgu.load_pickle(mdata_dir,file_name)

        session_names = list(mouse_dict.keys())
        session_names.sort()
        R2s_single_cells = {}
        for beh_name in list(beh_variables.keys()):
            R2s_single_cells[beh_name] = np.zeros((len(session_names), 10, signal.shape[1]))

        import itertools
        # Define your list of numbers
        num_cells = len(mouse_dict[session_names[0]]['global_id'][0])
        numbers = list(range(num_cells))  # This will generate numbers from 0 to 9
        # Generate all unique 3-number combinations
        combinations = list(itertools.combinations(numbers, 3))

        keep_10000 = np.arange(len(combinations))
        np.random.shuffle(keep_10000)
        keep_10000 = keep_10000[:10000]
        combinations = [x for idx,x in enumerate(combinations) if idx in keep_10000]
        R2s_triple_cells = {}
        for beh_name in list(beh_variables.keys()):
            R2s_triple_cells[beh_name] = np.zeros((len(session_names), 10, len(combinations)))

        for idx, session in enumerate(session_names):
            print(f"Working on session {session} ({idx+1}/{len(session_names)}):", end='', flush=True)

            dec_dict[session] = dict()
            mov_dir = lrgu.get_signal(mouse_dict[session], 'mov_direction').copy()
            valid_index = mov_dir!=0


            beh_variables = {
                'posx': lrgu.get_signal(mouse_dict[session], 'position')[valid_index,0].copy(),
                'speed': lrgu.get_signal(mouse_dict[session], 'speed')[valid_index].copy(),
                'trial_id_mat': lrgu.get_signal(mouse_dict[session], 'trial_id_mat')[valid_index].copy(),
                'dir': lrgu.get_signal(mouse_dict[session], 'mov_direction')[valid_index].copy()
            }

            trial_id = lrgu.get_signal(mouse_dict[session], 'trial_id_mat')[valid_index].copy()

            signal = lrgu.get_signal(mouse_dict[session], signal_name)[valid_index,:]
            if params['cells']=='same_cells':         #keep only same cells across days
                signal = signal[:, mouse_dict[session]['global_id'][0]]

            for beh_name, beh_val  in beh_variables.items():

                if beh_name in ['posx', 'speed','trial_id_mat']: 
                    decoder_list = ["xgb"]
                    metric = "median_absolute_error"
                else:
                    decoder_list = ["svc"]
                    metric = "f1_score"

                # #single cells
                # for c in range(signal.shape[1]):
                #     R2s, pred = decoders_1D(signal[:,c].reshape(-1,1), beh_val, decoder_list=decoder_list, trial_signal = trial_id, metric = metric, n_splits=10)
                #     R2s_single_cells[beh_name][idx, :,c] = R2s['base_signal'][decoder_list[0]][:,0,0]

                #tripple cells
                for i, (a,b,c) in enumerate(combinations):
                    R2s, pred = decoders_1D(signal[:,[a,b,c]], beh_val, decoder_list=decoder_list, trial_signal = trial_id, metric = metric, n_splits=10)
                    R2s_triple_cells[beh_name][idx, :,i] = R2s['base_signal'][decoder_list[0]][:,0,0]
                    

                dec_dict[session][beh_name] = {
                    'R2s': copy.deepcopy(R2s),
                    'pred': copy.deepcopy(pred),
                    'beh_val': copy.deepcopy(beh_val),
                    'signal': copy.deepcopy(signal),
                    'params': copy.deepcopy(params),
                }

                print(f" {beh_name}={np.nanmean(R2s['base_signal'][decoder_list[0]]):.4f} |", end='', sep = '', flush='True')
            print()

        lrgu.save_pickle(msave_dir,f"{mouse}_dec_{case}_{cells}_{signal_name}_dict.pkl", dec_dict)
        lrgu.print_time_verbose(local_time, global_time)
        sys.stdout = original
        f.close()



import itertools
# Define your list of numbers
numbers = list(range(60))  # This will generate numbers from 0 to 9
# Generate all unique 3-number combinations
combinations = list(itertools.combinations(numbers, 3))
# Print the combinations
for combo in combinations:
    print(combo)
print(combo, information, 'mouse2019')



