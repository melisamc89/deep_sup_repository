import os, copy
import learning_repo.general_utils as lrgu


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


##############################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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


si_pd_all = pd.DataFrame(data={'mouse': mouse_name_list,
                           'area': area_list,
                     'session_type': session_list,
                     'si': si_umap_time,
                           'si_pos': si_umap_pos,
                           'si_dir': si_umap_dir,
                           })

si_pd_lt = si_pd_all[si_pd_all['session_type'] == 'lt']
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

plt.savefig(os.path.join(data_dir, f'SI_filter_umap_10pcs_0-18_individual_{signal_name}.png'), dpi=400, bbox_inches="tight")


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
plt.savefig(os.path.join(data_dir, f'SI_umap_10pcs_mean_filter_0_18_{signal_name}.png'), dpi=400, bbox_inches="tight")

########################################################################################

