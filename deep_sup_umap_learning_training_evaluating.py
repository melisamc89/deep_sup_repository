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















from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

# Define segments
n_train_start = int(T * 0.15)
n_train_end = int(T * 0.85)
X_train = X[n_train_start:n_train_end]

X_test_early = X[:int(T * 0.15)]
X_test_late = X[-int(T * 0.15):]

# Time values (for coloring)
time_test_early = np.linspace(0, 0.10, len(X_test_early))
time_train = np.linspace(0.15, 0.85, len(X_train))
time_test_late = np.linspace(0.90, 1.00, len(X_test_late))

# UMAP embedding
umap_model = UMAP(n_neighbors=120, n_components=3, min_dist=0.1, random_state=42)
X_train_umap = umap_model.fit_transform(X_train)
X_test_early_umap = umap_model.transform(X_test_early)
X_test_late_umap = umap_model.transform(X_test_late)

# Get limits for uniform axis scaling
combined = np.vstack([X_test_early_umap, X_train_umap, X_test_late_umap])
lims = {
    'x': (combined[:, 0].min(), combined[:, 0].max()),
    'y': (combined[:, 1].min(), combined[:, 1].max()),
    'z': (combined[:, 2].min(), combined[:, 2].max())
}

# View angle
elev, azim = 30, 90
# Plotting
fig = plt.figure(figsize=(18, 6))
# Subplot 1: Early test
ax1 = fig.add_subplot(141, projection='3d')
ax1.scatter(X_test_early_umap[:, 0], X_test_early_umap[:, 1], X_test_early_umap[:, 2],
            c=time_test_early, cmap='viridis', s=10, vmin = 0, vmax = 1)
ax1.set_title("Test: First 15%")
ax1.set_xlim(lims['x'])
ax1.set_ylim(lims['y'])
ax1.set_zlim(lims['z'])
ax1.view_init(elev=elev, azim=azim)
ax1.grid(False)
# Subplot 2: Training
ax2 = fig.add_subplot(142, projection='3d')
ax2.scatter(X_train_umap[:, 0], X_train_umap[:, 1], X_train_umap[:, 2],c = np.arange(0,X_train_umap.shape[0])/T,
            cmap='viridis', s=10, vmin = 0, vmax = 1)
ax2.set_title("Train: Middle 70%")
ax2.set_xlim(lims['x'])
ax2.set_ylim(lims['y'])
ax2.set_zlim(lims['z'])
ax2.view_init(elev=elev, azim=azim)
ax2.grid(False)
# Subplot 3: Late test
ax3 = fig.add_subplot(143, projection='3d')
ax3.scatter(X_test_late_umap[:, 0], X_test_late_umap[:, 1], X_test_late_umap[:, 2],
            c=time_test_late, cmap='viridis', s=10, vmin= 0, vmax = 1)
ax3.set_title("Test: Last 15%")
ax3.set_xlim(lims['x'])
ax3.set_ylim(lims['y'])
ax3.set_zlim(lims['z'])
ax3.view_init(elev=elev, azim=azim)
ax3.grid(False)
# Subplot 4: All combined
ax4 = fig.add_subplot(144, projection='3d')
ax4.scatter(X_train_umap[:, 0], X_train_umap[:, 1], X_train_umap[:, 2],
            c='gray', s=10, label='Train', alpha = 0.01)
ax4.scatter(X_test_early_umap[:, 0], X_test_early_umap[:, 1], X_test_early_umap[:, 2],
            c=plt.cm.viridis(mcolors.Normalize(vmin=0.0, vmax=1.0)(time_test_early)), s=10, label='Early Test')
ax4.scatter(X_test_late_umap[:, 0], X_test_late_umap[:, 1], X_test_late_umap[:, 2],
            c=plt.cm.viridis(mcolors.Normalize(vmin=0.0, vmax=1.0)(time_test_late)), s=10, label='Late Test')
ax4.set_title("Overlayed")
ax4.set_xlim(lims['x'])
ax4.set_ylim(lims['y'])
ax4.set_zlim(lims['z'])
ax4.view_init(elev=elev, azim=azim)
ax4.legend(loc='upper right')
ax4.grid(False)
plt.tight_layout()
plt.show()
