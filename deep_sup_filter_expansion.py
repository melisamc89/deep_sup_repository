import os
import sys
import copy

import time
import timeit
import os, copy

import umap
from sklearn.decomposition import PCA

import learning_repo.general_utils as lrgu
import learning_repo.geometric_utils as lrgeo
from structure_index import compute_structure_index
import elephant
from scipy.signal import convolve
from elephant.kernels import GaussianKernel
from quantities import ms
import quantities as pq
import numpy as np
from elephant.statistics import instantaneous_rate
import quantities as pq
import neo
from scipy.ndimage import gaussian_filter1d

def smooth_calcium_signals(calcium_data, sigma=4):
    """
    Apply Gaussian smoothing to each row of the calcium imaging data.
    Each row represents the calcium signal of one neuron.

    Parameters:
    - calcium_data: 2D numpy array, where each row is a neuron's time series.
    - sigma: Standard deviation for Gaussian kernel. The value is in terms of index units.

    Returns:
    - smoothed_signals: 2D numpy array of smoothed signals.
    """
    # Initialize an array to store the smoothed signals
    smoothed_signals = np.zeros_like(calcium_data)

    # Apply Gaussian smoothing to each row (neuron)
    for i in range(calcium_data.shape[1]):
        smoothed_signals[:,i] = gaussian_filter1d(calcium_data[:,i], sigma=sigma)

    return smoothed_signals

from sklearn.metrics import pairwise_distances

def filter_noisy_outliers(data):
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,1), axis=1) - 1
    noiseIdx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    signalIdx = np.where(nn_dist >= np.percentile(nn_dist, 20))[0]
    return noiseIdx, signalIdx


import numpy as np


def sum_consecutive_trial_distances(rates, trial_id, order="sorted", metric="euclidean"):
    """
    rates: array-like, shape (T, D) or (T,) — signal at each timepoint (can be multidimensional).
    trial_id: array-like, shape (T,) — trial id for each timepoint.
    order: "sorted" (by increasing trial_id) or "appearance" (first occurrence order).
    metric: currently supports "euclidean".

    Returns:
        total_distance (float), per_pair (list of (trial_i, trial_j, distance))
    """
    rates = np.asarray(rates)
    trial_id = np.asarray(trial_id)

    if rates.ndim == 1:
        rates = rates[:, None]  # make it (T, 1)

    # choose trial order
    uniq = np.unique(trial_id) if order == "sorted" else np.array(list(dict.fromkeys(trial_id.tolist())))

    # compute centers (mean per trial_id)
    centers = []
    valid_trials = []
    for tid in uniq:
        mask = (trial_id == tid)
        if not np.any(mask):
            continue
        # mean over timepoints in the trial; ignore NaNs
        center = np.nanmean(rates[mask], axis=0)
        # skip if all NaN
        if np.isnan(center).all():
            continue
        centers.append(center)
        valid_trials.append(tid)
    centers = np.vstack(centers)  # shape (Ntrials, D)

    # distances between consecutive trials
    if metric != "euclidean":
        raise ValueError("Only 'euclidean' metric is implemented.")
    #diffs = centers[1:] - centers[:-1]
    #dists = np.linalg.norm(diffs, axis=1)

    diffs = np.diff(centers)
    dists = np.sum(np.abs(diffs))

    #total_distance = float(np.nansum(dists))
    #per_pair = [(valid_trials[i], valid_trials[i + 1], float(dists[i])) for i in range(len(dists))]
    return dists


def compute_trial_id(pos, high_thr=50, low_thr=10):
    """
    Compute trial IDs from a 1D position signal.
    A trial is defined as:
      - animal goes above `high_thr`
      - then comes back below `low_thr`
    After that, trial_id increments by 1.

    Parameters
    ----------
    pos : array-like, shape (T,)
        Position signal over time.
    high_thr : float
        Threshold for "forward" excursion.
    low_thr : float
        Threshold for "return" excursion.

    Returns
    -------
    trial_id : np.ndarray, shape (T,)
        Trial ID for each time point (starts at 0).
    """

    pos = np.asarray(pos).ravel()
    trial_id = np.zeros_like(pos, dtype=int)

    in_high = False   # flag: we crossed above high_thr
    current_id = 0

    for t in range(1, len(pos)):
        if not in_high and pos[t] > high_thr:
            in_high = True
        elif in_high and pos[t] < low_thr:
            current_id += 1
            in_high = False
        trial_id[t] = current_id

    return trial_id
# ---------- Exact Minimum Enclosing Ball (Welzl) ----------
def _ball_from(points):
    P = np.asarray(points, dtype=float)
    if len(P) == 0:
        return np.zeros(1), 0.0
    if len(P) == 1:
        return P[0], 0.0
    if len(P) == 2:
        c = (P[0] + P[1]) / 2.0
        r = np.linalg.norm(P[0] - c)
        return c, r
    A = 2 * (P[1:] - P[0])                    # (k-1, d)
    b = np.sum(P[1:]**2 - P[0]**2, axis=1)    # (k-1,)
    try:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        c = P.mean(axis=0)
    r = np.max(np.linalg.norm(P - c, axis=1))
    return c, r

def _is_in_ball(p, c, r, eps=1e-10):
    return np.linalg.norm(p - c) <= r + 1e-10 + eps

def _welzl(P, R, d):
    if len(R) == d + 1 or len(P) == 0:
        c, r = _ball_from(R)
        return c, r
    p = P.pop()
    c, r = _welzl(P, R, d)
    if _is_in_ball(p, c, r):
        P.append(p)
        return c, r
    R.append(p)
    c2, r2 = _welzl(P, R, d)
    R.pop()
    P.append(p)
    return c2, r2

def minimum_enclosing_ball(X):
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.zeros((X.shape[1] if X.ndim==2 else 1,)), 0.0
    n, d = X.shape
    P = [x for x in X]
    rng = np.random.default_rng(123)
    rng.shuffle(P)
    c, r = _welzl(P, [], d)
    return c, float(r)

# ---------- Trial centroids helper ----------
def trial_centroids(rates, trial_id, order="sorted"):
    """
    rates: (T, D), trial_id: (T,)
    Returns: centers (Ntrials, D) sorted by 'order', and the corresponding trial IDs.
    """
    rates = np.asarray(rates)
    tids  = np.asarray(trial_id).ravel()
    if rates.ndim == 1:
        rates = rates[:, None]

    uniq = np.unique(tids) if order == "sorted" else np.array(list(dict.fromkeys(tids.tolist())))
    centers, keep_tids = [], []
    for tid in uniq:
        m = (tids == tid)
        if not np.any(m):
            continue
        c = np.nanmean(rates[m], axis=0)
        if not np.isnan(c).all():
            centers.append(c)
            keep_tids.append(tid)
    if len(centers) == 0:
        return np.empty((0, rates.shape[1])), np.array([], dtype=int)
    return np.vstack(centers), np.asarray(keep_tids)


base_dir =  '/home/melma31/Documents/deepsup_project/'

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'

data_dir = os.path.join(base_dir, 'MIR')
save_dir = os.path.join(base_dir, 'Filters_expansion')
if not os.path.isdir(save_dir): os.makedirs(save_dir)

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        print(mouse)

        global_time = timeit.default_timer()  # initialize time
        local_time = timeit.default_timer()
        mdata_dir = os.path.join(data_dir, mouse)  # mouse save dir
        msave_dir = os.path.join(save_dir, mouse)  # mouse save dir
        if not os.path.isdir(msave_dir): os.makedirs(msave_dir)
        f = open(os.path.join(msave_dir, f"{mouse}_linearity_{signal_name}_logFile.txt"), 'w')
        original = sys.stdout
        sys.stdout = lrgu.Tee(sys.stdout, f)

        mi_dict = lrgu.load_pickle(mdata_dir, f"{mouse}_mi_cluster_all_{signal_name}_dict.pkl")
        session_names = list(mi_dict.keys())
        session_names.sort()
        si_dict = dict()
        for idx, session in enumerate(session_names):
            if 'lt' in session:
                si_dict[session] = dict()
                print(f"Working on session {session} ({idx + 1}/{len(session_names)}): ", sep='', end='')
                mov_dir = mi_dict[session]['behaviour']['mov_dir']
                valid_index =  mi_dict[session]['valid_index']

                valid_index = np.arange(0,mov_dir.shape[0])
                beh_variables = {
                    'pos': mi_dict[session]['behaviour']['position'][valid_index].copy(),
                    '(pos,dir)': mi_dict[session]['behaviour']['(pos,dir)'][valid_index].copy(),
                    'speed': mi_dict[session]['behaviour']['speed'][valid_index].copy(),
                    'trial_id_mat': mi_dict[session]['behaviour']['trial_id'][valid_index].copy(),
                    'dir':  mi_dict[session]['behaviour']['mov_dir'][valid_index].copy(),
                    'time': mi_dict[session]['behaviour']['time'][valid_index].copy()
                    }

                position = beh_variables['pos']
                trial_id = compute_trial_id(position)
                signal = mi_dict[session]['signal']
                signal = signal[valid_index, :]
                #selected_indices = np.random.choice(signal.shape[1], 20, replace=False)
                #signal = signal[:, selected_indices]

                noise_idx, signal_idx = filter_noisy_outliers(signal)
                print(f"Working on session {session} ({idx + 1}/{len(session_names)}):",end='')

                kernels = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
                for index , filter_size in enumerate(kernels):
                    rates = smooth_calcium_signals(signal, filter_size)

                    #distance = np.linalg.norm(rates[0,:] - rates[-1,:])
                    #distance = np.sum(np.abs(np.diff(rates,axis=0)))
                    #distance  = sum_consecutive_trial_distances(rates, trial_id, order="sorted", metric="euclidean")
                    # Compute trial centroids in original (filtered) space, then MEB radius
                    centers, _ = trial_centroids(rates, trial_id, order="sorted")

                    # MEB (same definition as your dispersion code)
                    if centers.shape[0] > 0:
                        _, meb_radius = minimum_enclosing_ball(centers)
                        D = centers.shape[1]
                        meb_radius_norm = meb_radius / np.sqrt(D) if D > 0 else 0.0
                    else:
                        meb_radius = 0.0
                        meb_radius_norm = 0.0

                    si_dict[session][str(filter_size)] = {
                            'expansion': copy.deepcopy(meb_radius_norm),
                            'valid_idx': valid_index.copy(),
                            'signal_idx': signal_idx.copy(),
                            'signal': rates.copy()
                    }

                    print(f"{filter_size}={meb_radius_norm:.4f} |", end='', sep='', flush='True')
                    print()

        lrgu.save_pickle(msave_dir, f"{mouse}_distance_expansion_MEB_{signal_name}_dict.pkl", si_dict)
        lrgu.print_time_verbose(local_time, global_time)
        sys.stdout = original

######################### plot distnace #########################################


import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats

data_dir = os.path.join(base_dir, 'SI')
mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'
mov_cond = 'all_mov'

df_rows = []
for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        si_dict = lrgu.load_pickle(os.path.join(save_dir, mouse),
                                   f"{mouse}_distance_expansion_{signal_name}_dict.pkl")
        for session, filter_data in si_dict.items():
                for filter_id,data in filter_data.items():
                    row = {
                        'area': area,
                        'session': session,
                        'mouse': mouse,
                        'session': session,
                        'filter': int(filter_id),
                        'distance': data['expansion']
                    }
                    df_rows.append(row)
# Create DataFrame
df = pd.DataFrame(df_rows)
# Optional: sort it for readability
df.sort_values(by=['area','mouse', 'session', 'filter'], inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
# Set up style and palette
palette = {'superficial': '#9900ff', 'deep': '#cc9900'}
# Convert filter size to time in seconds (sampling rate = 20 Hz)
sampling_rate = 20  # Hz
df['filter_time'] = df['filter'] / sampling_rate
# Get unique behavioral labels
# Create subplots: one row per behavior
fig, axes = plt.subplots(1, 1, figsize=(8, 2))
# If only one behavior, keep axes as list
# Plot each behavior separately
sns.lineplot(data=df,
                 x='filter_time', y='distance', hue='area',
                 errorbar='sd', ax=axes, palette=palette, marker='o')
# Labels and formatting
axes.set_title(f'Distance vs Filter Time')
axes.set_ylabel('Distance')
axes.set_xlabel('Filter Size (seconds)')
axes.legend(title='Area', loc='upper right')
axes.grid(False)
# Final layout
plt.tight_layout()
plt.show()
# Save figure
fig.savefig(os.path.join(save_dir, f"distance_road_filter_{signal_name}.png"), dpi=400,
            bbox_inches="tight")
fig.savefig(os.path.join(save_dir, f"distance_road_filter_{signal_name}.svg"), dpi=400,
            bbox_inches="tight")