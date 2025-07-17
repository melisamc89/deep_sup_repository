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
            kernels = [0, 4, 8, 14,20]

            smoothed_signals_dict = {}
            for filt in kernels:
                if filt == 0:
                    smoothed_signals_dict[filt] = signal.copy()
                else:
                    smoothed_signals_dict[filt] = smooth_calcium_signals(signal, filt)

            for cmap_type, cmap, label in [('pos', 'plasma', 'Position'), ('time', 'summer', 'Time')]:
                fig = plt.figure(figsize=(20, 20))
                for i, train_filt in enumerate(kernels):
                    umap_model = umap.UMAP(n_neighbors=120, n_components=dim, min_dist=0.1, random_state=42)
                    umap_model.fit(smoothed_signals_dict[train_filt])

                    for j, test_filt in enumerate(kernels):
                        transformed = umap_model.transform(smoothed_signals_dict[test_filt])
                        color_data = beh_variables[cmap_type]

                        ax = fig.add_subplot(len(kernels), len(kernels), i * len(kernels) + j + 1, projection='3d')
                        ax.scatter(*transformed[:, :3].T, c=color_data, cmap=cmap, s=1, alpha=0.5)
                        ax.set_title(f"Train: {train_filt}, Test: {test_filt}")
                        ax.view_init(init_view1, init_view2)

                fig.suptitle(f'UMAP: Train/Test Filters - Colored by {label}', fontsize=16)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(os.path.join(msave_dir, f"{mouse}_umap_crossfilter_{signal_name}_{session}_{cmap_type}.png"), dpi=400)

        sys.stdout = sys.__stdout__
        f.close()










#################3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
X = signal
# UMAP embedding
umap_model = umap.UMAP(n_neighbors=120, n_components=10, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X)

# KDE in UMAP space
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(X_umap)
log_density = kde.score_samples(X_umap)
energy = -log_density

# Plot energy over time and UMAP
fig = plt.figure(figsize=(12, 10))

# Top: Energy over time
ax1 = fig.add_subplot(211)
ax1.plot(energy, color='darkorange')
ax1.set_title('Energy over Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('Energy (−log density)')
ax1.grid(True)

# Bottom: UMAP 3D colored by energy
ax2 = fig.add_subplot(212, projection='3d')
sc = ax2.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2],
                 c=energy, cmap='viridis', s=10)
ax2.set_title('UMAP Embedding Colored by Energy')
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.set_zlabel('UMAP 3')
fig.colorbar(sc, ax=ax2, label='Energy (−log density)', shrink=0.6, pad=0.1)

plt.tight_layout()
plt.show()

#####

# Step 1: Define weight matrix W for Hopfield-like energy
# Use Hebbian rule based on a few "pattern" states (e.g., selected timepoints)
pattern_indices = [200, 500, 800, 1000, 3000,6000,7000,3200]  # arbitrary example patterns
patterns = X[pattern_indices]
W = np.sum([np.outer(p, p) for p in patterns], axis=0)
# Ensure W is symmetric and zero diagonal (optional)
np.fill_diagonal(W, 0)
# Step 2: Compute energy over time
# E(x_t) = -0.5 * x_t^T W x_t
energy_hopfield = -0.5 * np.einsum('ij,ij->i', X @ W, X)
# Step 3: Plot energy over time
plt.figure(figsize=(10, 4))
plt.plot(energy_hopfield, color='crimson', label='Hopfield-Like Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Hopfield-Like Quadratic Energy over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Smooth the Hopfield energy signal
from scipy.signal import find_peaks

energy_hopfield_smooth = gaussian_filter1d(energy_hopfield, sigma=3)

# Find local minima
minima_indices_hopfield, _ = find_peaks(-energy_hopfield_smooth, distance=50)

# Create label array for UMAP coloring: 1 = local minima, 0 = otherwise

colors_hopfield = np.zeros_like(energy_hopfield, dtype=int)
colors_hopfield[minima_indices_hopfield] = 1

# Plot energy over time and UMAP highlighting local minima
fig = plt.figure(figsize=(12, 10))

# Top: Energy over time with minima marked
ax1 = fig.add_subplot(211)
ax1.plot(energy_hopfield, color='darkorange', label='Hopfield Energy')
ax1.plot(minima_indices_hopfield, energy_hopfield[minima_indices_hopfield],
         'bo', label='Local Minima')
ax1.set_title('Hopfield-Like Energy over Time with Minima')
ax1.set_xlabel('Time')
ax1.set_ylabel('Energy')
ax1.legend()
ax1.grid(True)

# Bottom: UMAP embedding colored by local minima
ax2 = fig.add_subplot(212, projection='3d')
#sc = ax2.scatter(X_umap[:, 1], X_umap[:, 2], X_umap[:, 0],
#                 c=dir_color, s=10, alpha = 0.05)
ax2.set_title('UMAP Embedding: Local Minima Highlighted')
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.set_zlabel('UMAP 3')
fig.colorbar(sc, ax=ax2, ticks=[0, 1], label='0 = Not Minima, 1 = Minima', shrink=0.6, pad=0.1)
ax2.scatter(X_umap[minima_indices_hopfield, 1], X_umap[minima_indices_hopfield, 2], X_umap[minima_indices_hopfield, 0],
                 c = beh['time'][minima_indices_hopfield],cmap='YlGn', s=50)
plt.tight_layout()
plt.show()
####################3

# Optimized Lyapunov estimation by computing nearest neighbors only once

# Re-import necessary packages after reset
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Simulate calcium traces (T x N)
T, N =X.shape
# Step 1: Hopfield energy
patterns = X[pattern_indices]
W = np.sum([np.outer(p, p) for p in patterns], axis=0)
np.fill_diagonal(W, 0)
energy = -0.5 * np.einsum('ij,ij->i', X @ W, X)

# Step 2: Smooth and find local minima
energy_smooth = gaussian_filter1d(energy, sigma=3)
minima_indices, _ = find_peaks(-energy_smooth, distance=100)
is_minima = np.zeros_like(energy, dtype=int)
is_minima[minima_indices] = 1

# Step 3: Estimate local Lyapunov exponents (optimized)
window = 10
nbrs = NearestNeighbors(n_neighbors=2).fit(X)
dists, idxs = nbrs.kneighbors(X)

lyapunov_exponents = np.full(T, np.nan)
for t in range(T - window):
    nn_idx = idxs[t, 1] if idxs[t, 0] == t else idxs[t, 0]
    deltas = []
    for tau in range(window):
        if t + tau >= T or nn_idx + tau >= T:
            break
        d = np.linalg.norm(X[t + tau] - X[nn_idx + tau])
        deltas.append(d)

    deltas = np.array(deltas)
    if np.all(deltas > 0):
        log_deltas = np.log(deltas)
        reg = LinearRegression().fit(np.arange(len(log_deltas)).reshape(-1, 1), log_deltas)
        lyapunov_exponents[t] = reg.coef_[0]

# Step 4: Estimate curvature (second derivative of smoothed energy)
curvature = np.gradient(np.gradient(energy_smooth))

# Create dataframe
df = pd.DataFrame({
    'Energy': energy,
    'Energy_smooth': energy_smooth,
    'Is_Minima': is_minima,
    'Curvature': curvature,
    'Lyapunov': lyapunov_exponents
})

# Plot energy over time and UMAP highlighting local minima
fig = plt.figure(figsize=(12, 10))
# Top: Energy over time with minima marked
ax1 = fig.add_subplot(211)
ax1.plot(curvature, color='darkorange', label='Hopfield Energy')
ax1.plot(minima_indices_hopfield, energy_hopfield[minima_indices_hopfield],
         'bo', label='Local Minima')
ax1.set_title('Hopfield-Like Energy GRADIENTE over Time with Minima')
ax1.set_xlabel('Time')
ax1.set_ylabel('Energy GRADIENT')
ax1.legend()
ax1.grid(True)
# Bottom: UMAP embedding colored by local minima
ax2 = fig.add_subplot(212, projection='3d')
sc = ax2.scatter(X_umap[:, 1], X_umap[:, 2], X_umap[:, 0],
                 c=curvature, s=10, alpha = 0.5, cmap = 'viridis', vmin = -10, vmax = 10)
ax2.set_title('UMAP Embedding: ENERGY GRADIENT colored')
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.set_zlabel('UMAP 3')
fig.colorbar(sc, ax=ax2)
#ax2.scatter(X_umap[minima_indices_hopfield, 1], X_umap[minima_indices_hopfield, 2], X_umap[minima_indices_hopfield, 0],
#                 c = beh['time'][minima_indices_hopfield],cmap='YlGn', s=50)
plt.tight_layout()
plt.show()