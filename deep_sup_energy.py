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

number_of_minimum =75
percentage = 10
percentage_text = '50'

base_dir =  '/home/melma31/Documents/deepsup_project/'
#__________________________________________________________________________
#|                                                                        |#
#|                           Energy calculation                     |#
#|________________________________________________________________________|#
data_dir = os.path.join(base_dir, 'data')
save_dir = os.path.join(base_dir, 'Energy')
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

                    import numpy as np
                    import matplotlib.pyplot as plt
                    from scipy.stats import linregress
                    from scipy.signal import argrelextrema

                    def compute_hopfield_energy(X, pattern_indices, eval_indices):
                        patterns = X[pattern_indices]
                        W = np.sum([np.outer(p, p) for p in patterns], axis=0)
                        np.fill_diagonal(W, 0)
                        energy = -0.5 * np.einsum('ij,ij->i', X[eval_indices] @ W, X[eval_indices])
                        return energy, eval_indices  # Also return the evaluation x-axis


                    def get_energy_segments(signal):
                        X = signal.copy()
                        T = X.shape[0]
                        ten_percent = int(percentage/100 * T)
                        segments = {
                            'First_'+percentage_text: {
                                'train': np.arange(0, ten_percent),
                                'eval': np.arange(ten_percent, T)
                            },
                            'Last_'+percentage_text: {
                                'train': np.arange(T - ten_percent, T),
                                'eval': np.arange(0, T - ten_percent)
                            }
                        }

                        energies = {}
                        for label, idxs in segments.items():
                            energy, eval_idx = compute_hopfield_energy(X, pattern_indices=idxs['train'],
                                                                       eval_indices=idxs['eval'])
                            energies[label] = {'energy': energy, 'eval_idx': eval_idx}
                        return energies


                    def plot_energy_with_minima_fit(energies, top_k=30, fig_name='.png', mouse='', model=''):
                        selected_labels = list(energies.keys())
                        vertical_shifts = [i * 100 for i in range(len(selected_labels))]
                        plt.figure(figsize=(12, 5))

                        # Initialize nested dict with model and mouse
                        if model not in R_summary:
                            R_summary[model] = {}
                        R_summary[model][mouse] = {}

                        for i, label in enumerate(selected_labels):
                            energy = energies[label]['energy']
                            x = energies[label]['eval_idx']
                            x_rel = np.arange(len(energy))  # relative to evaluation start

                            # Find local minima
                            minima_idx = argrelextrema(energy, np.less, order=1)[0]
                            minima_values = energy[minima_idx]

                            if len(minima_idx) > top_k:
                                top_indices = np.argsort(minima_values)[:top_k]
                                minima_idx = minima_idx[top_indices]
                                minima_values = minima_values[top_indices]

                            minima_x = x_rel[minima_idx]
                            minima_y = minima_values

                            slope, intercept, r_value, _, _ = linregress(minima_x, minima_y)
                            R_summary[model][mouse][label] = r_value  # <<< Save R-value per model and mouse

                            regression_line = slope * x_rel + intercept
                            shift = vertical_shifts[i]
                            shifted_energy = energy + shift
                            shifted_regression = regression_line + shift
                            shifted_minima_y = minima_y + shift

                            plt.plot(x[0] + x_rel, shifted_energy, label=f"{label} Energy", linewidth=1.2)
                            plt.scatter(x[0] + minima_x, shifted_minima_y, color='black', s=10, label=f"{label} Minima")
                            plt.plot(x[0] + x_rel, shifted_regression, linestyle='--', linewidth=2,
                                     label=f"{label} Fit (slope={slope:.4f}, R={r_value:.3f})")

                        plt.xlabel('Time')
                        plt.ylabel('Shifted Energy')
                        plt.title(f'Energy Fit — {mouse}', fontsize=20)
                        plt.legend()
                        plt.grid(False)
                        plt.tight_layout()
                        plt.savefig(fig_name+'.png', dpi=400, bbox_inches="tight")
                        plt.savefig(fig_name+'.svg', dpi=400, bbox_inches="tight")

                        plt.show()


                    # === MAIN EXECUTION ===
                    # signal = lrgu.get_signal(session_pd, signal_name)
                    # Assuming you already have: signal (2D array: time x features)
                    energies = get_energy_segments(signal)
                    name = os.path.join(save_dir,f'energy_{mouse}_min_{number_of_minimum}_segment_{percentage}')
                    plot_energy_with_minima_fit(energies, top_k=number_of_minimum, fig_name=name, mouse=mouse, model=model)


import pandas as pd
import pickle
# Save as pickle
with open(os.path.join(save_dir, 'R_summary.pkl'), 'wb') as f:
    pickle.dump(R_summary, f)

# Save as CSV
records = []
for mouse, segs in R_summary.items():
    for segment, r in segs.items():
        records.append({'Mouse': mouse, 'Segment': segment, 'R_value': r})
df_r = pd.DataFrame(records)
df_r.to_csv(os.path.join(save_dir, 'R_summary.csv'), index=False)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator  # pip install statannotations

# ---- Step 1: Categorize mice by depth ----
mice_dict = {
    'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
    'deep': ['ChZ4','ChZ8','ChZ7','GC2','GC3','GC7','GC5_nvista','TGrin1']
}

# ---- Step 2: Flatten R_summary into a DataFrame ----
rows = []
for model, mice_data in R_summary.items():
    for mouse, segments in mice_data.items():
        for segment, r_value in segments.items():  # 'First 10%' and 'Last 10%'
            depth = 'superficial' if mouse in mice_dict['superficial'] else 'deep'
            rows.append({
                'Mouse': mouse,
                'Model': model,
                'Depth': depth,
                'Segment': segment,
                'R_value': r_value
            })

df = pd.DataFrame(rows)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

# ---- Plot with dots ----
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=df, x='Depth', y='R_value', hue='Segment', ci='sd', capsize=.1, dodge=True)

# Add dots
sns.stripplot(
    data=df, x='Depth', y='R_value', hue='Segment',
    dodge=True, alpha=0.6, size=6, jitter=True,
    linewidth=0.5, edgecolor='gray', ax=ax
)

# Fix legend (to avoid duplicate entries)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title='Segment')

# Statistical annotation comparing Depth within each Segment
pairs = [
    (("superficial", "First_"+percentage_text), ("deep", "First_"+percentage_text)),
    (("superficial", "Last_"+percentage_text), ("deep", "Last_"+percentage_text)),
]

annotator = Annotator(ax, pairs, data=df, x='Depth', y='R_value', hue='Segment')
annotator.configure(test='t-test_ind', text_format='star', loc='inside')
annotator.apply_and_annotate()

ax.set_title('Energy Linearity (R) by Depth')
ax.set_ylabel('R Value')
ax.set_xlabel('Depth')
fig_name = os.path.join(save_dir, 'R_minima'+str(number_of_minimum)+'_segment_'+percentage_text+'.png')
plt.savefig(fig_name, dpi=400, bbox_inches="tight")
plt.tight_layout()
plt.show()


# ---- Step 4: Plot comparison by Model ----
plt.figure(figsize=(8,6))
ax = sns.barplot(data=df, x='Model', y='R_value', hue='Segment', ci='sd', capsize=.1)
# Add dots
sns.stripplot(
    data=df, x='Model', y='R_value', hue='Segment',
    dodge=True, alpha=0.6, size=6, jitter=True,
    linewidth=0.5, edgecolor='gray', ax=ax
)
ax.set_title('Energy Linearity (R) by Model')
ax.set_ylabel('R Value')
ax.set_xlabel('Model')

# Statistical annotation
pairs = [
    (("Calb", "First_"+percentage_text), ("Calb", "Last_"+percentage_text)),
    (("Chrna7", "First_"+percentage_text), ("Chrna7", "Last_"+percentage_text)),
    (("Thy1", "First_"+percentage_text), ("Thy1", "Last_"+percentage_text)),
]
annotator = Annotator(ax, pairs, data=df, x='Model', y='R_value', hue='Segment')
annotator.configure(test='t-test_paired', text_format='star', loc='inside')
annotator.apply_and_annotate()
fig_name = os.path.join(save_dir, 'R_minima'+str(number_of_minimum)+'_segment_'+percentage_text+'_model.png')
plt.savefig(fig_name, dpi=400, bbox_inches="tight")
plt.tight_layout()
plt.show()

###################load SI #############################################

data_dir = os.path.join(base_dir, 'results')
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

# ---- Step 1: Flatten R_summary into a DataFrame ----
r_data = []
for model, mouse_dict in R_summary.items():
    for mouse, segments in mouse_dict.items():
        for segment, r_val in segments.items():
            r_data.append({
                'mouse': mouse,
                'segment': 'First_'+percentage_text if 'First' in segment else 'Last_'+percentage_text,
                'R_value': r_val
            })
r_df = pd.DataFrame(r_data)
# ---- Step 2: Filter si_pd for 'lt' sessions only ----
si_lt = si_pd[si_pd['session_type'] == 'lt'].drop_duplicates(subset='mouse')
# ---- Step 3: Merge SI with R values ----
merged_df = pd.merge(r_df, si_lt[['mouse', 'si_time']], on='mouse', how='inner')
# ---- Step 4: Add Depth info ----
depth_dict = {
    'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
    'deep': ['ChZ4','ChZ8','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
}
depth_map = {}
for depth, mice in depth_dict.items():
    for m in mice:
        depth_map[m] = depth
merged_df['Depth'] = merged_df['mouse'].map(depth_map)
# ---- Step 5: Custom colors ----
custom_palette = {
    'superficial': '#9900ff',  # RGB: 153, 0, 255
    'deep': '#cc9900',         # RGB: 204, 153, 0
}
# ---- Step 6: Plot ----
sns.set(style="white", context="talk")
for segment in ['First_'+percentage_text, 'Last_'+percentage_text]:
    plt.figure(figsize=(7, 7))  # Square aspect ratio
    seg_df = merged_df[merged_df['segment'] == segment]
    # Scatter points
    ax = sns.scatterplot(
        data=seg_df, x='si_time', y='R_value', hue='Depth',
        palette=custom_palette, s=100, edgecolor='black'
    )
    # Overall regression fit
    if len(seg_df) >= 2:
        slope, intercept, r_all, _, _ = linregress(seg_df['si_time'], seg_df['R_value'])
        x_vals = np.linspace(seg_df['si_time'].min(), seg_df['si_time'].max(), 100)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, color='gray', label=f'Overall fit (R={r_all:.2f})', linestyle='--')
    # Per-group regression fits
    for depth_key, color in custom_palette.items():
        group_df = seg_df[seg_df['Depth'] == depth_key]
        if len(group_df) >= 2:
            slope, intercept, r_val, _, _ = linregress(group_df['si_time'], group_df['R_value'])
            x_vals = np.linspace(group_df['si_time'].min(), group_df['si_time'].max(), 100)
            y_vals = slope * x_vals + intercept
            label = f'{depth_key.capitalize()} fit (R={r_val:.2f})'
            ax.plot(x_vals, y_vals, color=color, label=label)
    plt.title(f'Structure Index vs Energy R — {segment}')
    plt.xlabel('Structure Index (Time)')
    plt.ylabel('Energy R-Value')
    plt.legend(title='Group / Fit', fontsize='x-small', loc='lower right')
    ax.grid(False)
    plt.tight_layout()
    fig_name = os.path.join(save_dir,'R_vs_E_'+segment+'_minimum_' + str(number_of_minimum) + '_segment_' + percentage_text + '.png')
    plt.savefig(fig_name, dpi=400, bbox_inches="tight")
    plt.show()
