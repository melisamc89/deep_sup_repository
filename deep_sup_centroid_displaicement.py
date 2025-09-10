import os, copy
from datetime import datetime
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.ndimage import gaussian_filter1d

import umap
from sklearn.decomposition import PCA

import learning_repo.general_utils as lrgu
import learning_repo.geometric_utils as lrgeo
from statistics import mode
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np

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


base_dir =  '/home/melma31/Documents/deepsup_project/'
#__________________________________________________________________________
#|                                                                        |#
#|                               DYNAMICS                                 |#
#|________________________________________________________________________|#

data_dir = os.path.join(base_dir, 'data')
save_dir = os.path.join(base_dir, 'dynamical_dynamics')
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

#general_mice_list = {'lt_and_rot':{
#              'Thy1':['GC2']}}
models = ['Calb', 'Chrna7', 'Thy1']
num_neigh = 120
dim = 3
min_dist = 0.1

for coloring in ['single_color']:
    if coloring == 'single_color':
        paradigm =['lt_and_rot','only_lt']
        #paradigm =['only_lt']
    else:
        paradigm =['only_lt']
    for paradigm_ in paradigm:
        mice_list = general_mice_list[paradigm_]
        for model in models:
            mice_model_list = general_mice_list[paradigm_][model]
            for mouse in mice_model_list:
                dynamics_dict = {}
                mdata_dir = os.path.join(data_dir,coloring,paradigm_,'processed_data',model ,mouse) #mouse data dir
                msave_dir = os.path.join(save_dir, mouse) #mouse save dir
                if not os.path.isdir(msave_dir): os.makedirs(msave_dir)

                #print(f"\t#### {mouse}: {case} | {cells} | {signal_name} ####")
                #print(f'\t{datetime.now():%Y-%m-%d %H:%M}\n')
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
                    mice_maze = mouse_dict[maze]
                    dynamics_dict[maze] = {}
                    params = {'columns_to_rename': {'Fs': 'sf', 'pos': 'position', 'vel': 'speed'}}
                    session_pd = copy.deepcopy(mice_maze)
                    for old, new in params['columns_to_rename'].items():
                        if old in session_pd.columns: session_pd.rename(columns={old: new}, inplace=True)
                    session_pd = lrgu.add_mov_direction_mat_field(session_pd)
                    session_pd = lrgu.add_inner_trial_time_field(session_pd)
                    session_pd = lrgu.add_trial_id_mat_field(session_pd)

                    signal = lrgu.get_signal(session_pd, 'clean_traces')
                    pos = lrgu.get_signal(session_pd, 'position')
                    speed = lrgu.get_signal(session_pd, 'speed')
                    trial_id = lrgu.get_signal(session_pd, 'trial_id_mat')
                    mov_dir = lrgu.get_signal(session_pd, 'mov_direction')
                    trial_id = compute_trial_id(pos[:,0])

                    #clean mov dir to avoid gaps in ring
                    temp_mov = np.diff(pos,axis=0)*20
                    temp_mov = np.concatenate([temp_mov[0,:].reshape(-1,2), temp_mov], axis= 0)[:,0]
                    temp_mov = gaussian_filter1d(temp_mov, sigma = 5, axis = 0)
                    temp_mov[temp_mov<0] = -1
                    temp_mov = np.sign(temp_mov)
                    mov_dir[mov_dir==0] = temp_mov[np.where(mov_dir==0)[0]]

                    #valid_index = np.where(mov_dir!=0)[0]
                    #signal = signal[valid_index]
                    # --- Drop NaNs / invalids (same as before)
                    valid_mask = np.isfinite(signal).all(axis=1) & np.isfinite(trial_id)
                    signal_valid = signal[valid_mask]
                    trial_id_valid = trial_id[valid_mask].astype(int).ravel()

                    # --- Compute one centroid per trial ID (sorted by trial ID)
                    unique_tids = np.unique(trial_id_valid)  # e.g., [0,1,2,...]
                    centroids_orig = np.vstack([
                        signal_valid[trial_id_valid == tid].mean(axis=0)
                        for tid in unique_tids
                    ]) if unique_tids.size else np.empty((0, signal.shape[1]))

                    # Labels/metadata (optional but handy)
                    centroid_labels = [str(int(tid)) for tid in unique_tids]  # ["0","1","2",...]
                    centroid_trial_ranges = [(int(tid), int(tid)) for tid in unique_tids]  # [(0,0),(1,1),...]
                    centroid_trials_lists = [[int(tid)] for tid in unique_tids]  # [[0],[1],...]

                    # --- Fit UMAP on the full valid signal and project both signal and centroids
                    umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist, random_state=42)
                    umap_model.fit(signal_valid)
                    umap_emb = umap_model.transform(signal_valid)
                    umap_centroids = umap_model.transform(centroids_orig) if len(centroids_orig) else np.empty((0, dim))


                    dynamics_dict[maze]["umap"] = {
                        "params": {"n_neighbors": num_neigh, "n_components": dim, "min_dist": min_dist,
                                   "random_state": 42},
                        "embedding_valid_only": umap_emb,  # (num_valid_timepoints, dim)
                        "trial_id": trial_id_valid,  # length = umap_emb.shape[0]
                    }

                    dynamics_dict[maze]["centroids"] = {
                        "labels": centroid_labels,
                        "trial_ranges": centroid_trial_ranges,
                        "trials_per_centroid": centroid_trials_lists,
                        "orig_space": centroids_orig,
                        "umap_space": umap_centroids,
                        "trial_ids": unique_tids.astype(int),
                        "counts_per_trial": np.array([np.sum(trial_id_valid == tid) for tid in unique_tids], dtype=int),
                    }

                    # Save per mouse so you collect them for all animals
                    lrgu.save_pickle(msave_dir, f"{mouse}_dynamics_centroids.pkl", dynamics_dict)


##############################################################################################
######################PLOT UMAP and CENTROIDS ################################################
##############################################################################################

import os, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
import learning_repo.general_utils as lrgu

# ------------ CONFIG ------------
data_root = os.path.join(base_dir, 'dynamical_dynamics')
out_dir   = os.path.join(data_root, 'group_flow_evolution')
os.makedirs(out_dir, exist_ok=True)

mice_dict = {
    'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
    'deep':        ['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
}

cmap_name = 'viridis'   # or 'YlGn_r'
bg_alpha  = 0.25        # background embedding transparency
bg_size   = 3           # background point size
ctr_size  = 30          # centroid point size

# ------------ HELPERS ------------
def load_mouse_umap_and_centroids(mouse: str):
    """
    Returns:
      - bg_points: (N, 3) background UMAP points (full embedding across all mazes), NaN rows removed
      - ctr_points: (K, 3) centroid UMAP points concatenated across mazes (ordered by encounter)
      - ctr_order: (K,) integer order indices 0..K-1
    """
    mdir = os.path.join(data_root, mouse)
    fname = f"{mouse}_dynamics_centroids.pkl"
    fpath = os.path.join(mdir, fname)
    if not os.path.isfile(fpath):
        return None, None, None

    d = lrgu.load_pickle(mdir, fname)

    # Collect full embeddings across mazes
    bg_list = []
    ctr_list = []
    ord_list = []
    k_accum = 0

    for maze, md in d.items():
        # --- Background embedding: prefer full; fallback to valid_only
        emb = None
        if "umap" in md:
            emb_full = md["umap"].get("embedding_full", None)
            emb_valid = md["umap"].get("embedding_valid_only", None)
            if emb_full is not None:
                emb = np.asarray(emb_full)
            elif emb_valid is not None:
                emb = np.asarray(emb_valid)
        if emb is not None and emb.shape[1] >= 3:
            # drop rows with NaNs
            m = np.isfinite(emb).all(axis=1)
            if m.any():
                bg_list.append(emb[m, :3])

        # --- Centroids
        if "centroids" in md:
            C = md["centroids"].get("umap_space", None)
            tids = md["centroids"].get("trial_ids", None)
            if C is not None and np.size(C) and np.asarray(C).shape[1] >= 3 and tids is not None:
                C = np.asarray(C)
                tids = np.asarray(tids)
                order = np.argsort(tids)
                C = C[order, :3]
                k = C.shape[0]
                ctr_list.append(C)
                ord_list.append(np.arange(k_accum, k_accum + k))
                k_accum += k

    bg_points  = np.vstack(bg_list) if len(bg_list) else None
    ctr_points = np.vstack(ctr_list) if len(ctr_list) else None
    ctr_order  = np.concatenate(ord_list) if len(ord_list) else None

    return bg_points, ctr_points, ctr_order

def make_grid(n, max_cols=4):
    cols = min(max_cols, max(1, n))
    rows = int(np.ceil(n / cols))
    return rows, cols

def get_group_axis_limits(group_mice):
    xs, ys, zs = [], [], []
    for mouse in group_mice:
        bg, ctr, _ = load_mouse_umap_and_centroids(mouse)
        for arr in (bg, ctr):
            if arr is not None and arr.size:
                xs.append(arr[:, 0]); ys.append(arr[:, 1]); zs.append(arr[:, 2])
    if not xs:  # default box
        return (-1, 1), (-1, 1), (-1, 1)
    xs = np.concatenate(xs); ys = np.concatenate(ys); zs = np.concatenate(zs)
    def pad(lo, hi):
        p = 0.05 * (hi - lo + 1e-9)
        return lo - p, hi + p
    return pad(xs.min(), xs.max()), pad(ys.min(), ys.max()), pad(zs.min(), zs.max())

# ------------ PLOTTING ------------
for group_name, group_mice in mice_dict.items():
    n = len(group_mice)
    rows, cols = make_grid(n, max_cols=4)

    # consistent axes across the group's subplots
    xlim, ylim, zlim = get_group_axis_limits(group_mice)

    fig = plt.figure(figsize=(5*cols, 5*rows))
    fig.suptitle(f"{group_name.capitalize()}: UMAP (full) + centroid projections", fontsize=16)

    global_min_idx, global_max_idx = np.inf, -np.inf

    for i, mouse in enumerate(group_mice):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        ax.set_title(mouse, fontsize=11)
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.set_zlabel('UMAP-3')
        #ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)

        bg_points, ctr_points, ctr_order = load_mouse_umap_and_centroids(mouse)
        if (bg_points is None or not bg_points.size) and (ctr_points is None or not ctr_points.size):
            ax.text2D(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes, fontsize=10)
            continue

        # --- Background: full UMAP embedding (gray)
        if bg_points is not None and bg_points.size:
            ax.scatter(bg_points[:, 0], bg_points[:, 1], bg_points[:, 2],
                       s=bg_size, alpha=bg_alpha, c='0.6', edgecolors='none')

        # --- Centroids: colored by order; also connect to show flow
        if ctr_points is not None and ctr_points.size:
            # update global color limits
            global_min_idx = min(global_min_idx, np.min(ctr_order))
            global_max_idx = max(global_max_idx, np.max(ctr_order))

            sc = ax.scatter(ctr_points[:, 0], ctr_points[:, 1], ctr_points[:, 2],
                            c=ctr_order, cmap=cmap_name, s=ctr_size, alpha=0.95, edgecolors='none')
            # subtle trajectory line
            ax.plot(ctr_points[:, 0], ctr_points[:, 1], ctr_points[:, 2], lw=0.7, alpha=0.6, color='k')

    # Shared colorbar for centroid order
    if np.isfinite(global_min_idx) and global_max_idx >= global_min_idx:
        norm = plt.Normalize(vmin=global_min_idx, vmax=global_max_idx)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_name)
        sm.set_array([])
        #cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.015, pad=0.04)
        #cbar.set_label('Centroid order (oldest → newest)')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, f"{group_name}_umap_full_plus_centroids_3D.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, f"{group_name}_umap_full_plus_centroids_3D.pdf"))
    plt.close(fig)

print(f"Saved 3D figures to: {out_dir}")




##############################################################################################
######################PLOT UMAP PER TRIALS FOR ONE EXAMPLE ####################################
##############################################################################################


import os, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
import learning_repo.general_utils as lrgu

# ---------- CONFIG ----------
mouse = "GC2"  # change as needed
data_root = os.path.join(base_dir, "dynamical_dynamics")
dyn_path  = os.path.join(data_root, mouse, f"{mouse}_dynamics_centroids.pkl")
out_dir   = os.path.join(data_root, "per_trial_figures")
os.makedirs(out_dir, exist_ok=True)

cmap_name = "viridis"
bg_color, bg_alpha, bg_size = "0.75", 0.25, 2
pt_size = 6
ctr_saved_size = 70     # centroid from saved transform(orig) -> UMAP
ctr_umapmean_size = 60  # centroid computed as mean of UMAP trial points
ctr_pca_marker = "^"    # marker for original-space centroids (PCA-3D)
ctr_umap_marker = "o"   # marker for UMAP-space centroids

# ---------- LOAD ----------
if not os.path.isfile(dyn_path):
    raise FileNotFoundError(f"Missing file: {dyn_path}")
dyn = lrgu.load_pickle(os.path.dirname(dyn_path), os.path.basename(dyn_path))

def make_grid(n, extras=0, max_cols=6):
    """
    Return (rows, cols) for n trial subplots plus 'extras' final subplots.
    """
    total = n + extras
    cols = min(max_cols, max(1, total))
    rows = int(np.ceil(total / cols))
    return rows, cols

# ---------- PLOTTING PER MAZE ----------
for maze, md in dyn.items():
    umap_info = md.get("umap", {})
    ctr_info  = md.get("centroids", {})

    emb = umap_info.get("embedding_valid_only", None)
    tids_per_point = umap_info.get("trial_id", None)
    ctr_umap_saved = ctr_info.get("umap_space", None)  # UMAP of orig centroids (saved)
    ctr_orig       = ctr_info.get("orig_space", None)  # centroids in original feature space
    ctr_tids       = ctr_info.get("trial_ids", None)

    if any(x is None for x in (emb, tids_per_point, ctr_umap_saved, ctr_orig, ctr_tids)):
        print(f"[{mouse} | {maze}] Missing required keys; skipping.")
        continue

    emb = np.asarray(emb)
    if emb.shape[1] < 3:
        print(f"[{mouse} | {maze}] UMAP is not 3D (shape {emb.shape}); skipping.")
        continue

    tids_per_point = np.asarray(tids_per_point).astype(int).ravel()
    ctr_umap_saved = np.asarray(ctr_umap_saved)
    ctr_orig       = np.asarray(ctr_orig)
    ctr_tids       = np.asarray(ctr_tids).astype(int).ravel()

    # Trials to show (sorted)
    unique_trials = np.sort(np.unique(ctr_tids))

    # Build a mapping from trial -> centroid rows (there should be 1 per trial)
    # and sort centroids by trial order once
    order = np.argsort(ctr_tids)
    ctr_tids_sorted = ctr_tids[order]
    ctr_umap_saved_sorted = ctr_umap_saved[order, :3]
    ctr_orig_sorted = ctr_orig[order]  # keep full-D; PCA will handle later

    # Precompute UMAP-mean centroids per trial directly from UMAP embedding points
    emb3 = emb[:, :3]
    umap_mean_centroids = []
    for tid in unique_trials:
        mask = (tids_per_point == tid)
        if np.any(mask):
            umap_mean_centroids.append(np.nanmean(emb3[mask], axis=0))
        else:
            umap_mean_centroids.append(np.array([np.nan, np.nan, np.nan]))
    umap_mean_centroids = np.vstack(umap_mean_centroids)

    # Shared axis limits across all subplots in this figure
    # include background, saved UMAP centroids, and UMAP-mean centroids
    all_x = np.concatenate([emb3[:,0], ctr_umap_saved_sorted[:,0], umap_mean_centroids[:,0]])
    all_y = np.concatenate([emb3[:,1], ctr_umap_saved_sorted[:,1], umap_mean_centroids[:,1]])
    all_z = np.concatenate([emb3[:,2], ctr_umap_saved_sorted[:,2], umap_mean_centroids[:,2]])
    def pad(lo, hi):
        p = 0.05 * (hi - lo + 1e-9)
        return lo - p, hi + p
    xlim, ylim, zlim = pad(np.nanmin(all_x), np.nanmax(all_x)), pad(np.nanmin(all_y), np.nanmax(all_y)), pad(np.nanmin(all_z), np.nanmax(all_z))

    # Figure: N trials + 2 extra (all UMAP centroids; all ORIGINAL centroids via PCA-3D)
    n_trials = len(unique_trials)
    rows, cols = make_grid(n_trials, extras=2, max_cols=6)
    fig = plt.figure(figsize=(4.6*cols, 4.6*rows))
    fig.suptitle(f"{mouse} | {maze} — UMAP per trial (points + centroids)\n"
                 f"Final: all centroids in UMAP (o) and ORIGINAL (PCA-3D, ^)", fontsize=14)

    # Color per trial (consistent across subplots)
    cmap = plt.get_cmap(cmap_name)
    trial_colors = {tid: cmap(i) for tid, i in zip(unique_trials, np.linspace(0.1, 0.9, n_trials))}

    # ---- Per-trial subplots ----
    for i, tid in enumerate(unique_trials):
        ax = fig.add_subplot(rows, cols, i+1, projection="3d")
        ax.set_title(f"Trial {int(tid)}", fontsize=10)

        # background UMAP (all valid points)
        ax.scatter(emb3[:,0], emb3[:,1], emb3[:,2],
                   s=bg_size, c=bg_color, alpha=bg_alpha, edgecolors="none")

        # points for this trial
        mask = (tids_per_point == tid)
        pts = emb3[mask]
        if pts.size:
            ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                       s=pt_size, alpha=0.95, edgecolors="none", c=trial_colors[tid])

        # centroid from SAVED transform(orig -> UMAP)
        # (lookup its row in the sorted arrays)
        idx = np.where(ctr_tids_sorted == tid)[0]
        if idx.size:
            c_saved = ctr_umap_saved_sorted[idx[0], :3]
            ax.scatter([c_saved[0]],[c_saved[1]],[c_saved[2]],
                       s=ctr_saved_size, facecolors="none", edgecolors="k", linewidths=1.4, marker=ctr_umap_marker,
                       label="centroid (saved UMAP)")

        # centroid computed as MEAN of UMAP trial points
        c_mean = umap_mean_centroids[i, :]
        if np.isfinite(c_mean).all():
            ax.scatter([c_mean[0]],[c_mean[1]],[c_mean[2]],
                       s=ctr_umapmean_size, facecolors=trial_colors[tid], edgecolors="k", linewidths=0.8, marker="X",
                       label="centroid (UMAP mean)")

        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")

    # ---- Final subplot: all UMAP centroids (saved) ----
    ax_u = fig.add_subplot(rows, cols, n_trials+1, projection="3d")
    ax_u.set_title("All centroids — UMAP space (saved)", fontsize=10)
    for tid in unique_trials:
        idx = np.where(ctr_tids_sorted == tid)[0]
        if idx.size:
            cc = ctr_umap_saved_sorted[idx[0], :3]
            ax_u.scatter([cc[0]],[cc[1]],[cc[2]],
                         s=ctr_saved_size, facecolors="none", edgecolors="k", linewidths=1.4,
                         marker=ctr_umap_marker, label=f"{tid}" if tid==unique_trials[0] else None,
                         c=[trial_colors[tid]])
    ax_u.set_xlim(*xlim); ax_u.set_ylim(*ylim); ax_u.set_zlim(*zlim)
    ax_u.set_xlabel("UMAP-1"); ax_u.set_ylabel("UMAP-2"); ax_u.set_zlabel("UMAP-3")

    # ---- Final subplot: all ORIGINAL-space centroids (PCA-3D) ----
    ax_o = fig.add_subplot(rows, cols, n_trials+2, projection="3d")
    ax_o.set_title("All centroids — ORIGINAL space (PCA-3D)", fontsize=10)
    # quick PCA to 3D (only for plotting)
    if ctr_orig_sorted.shape[0] >= 2 and ctr_orig_sorted.shape[1] >= 3:
        pca3 = PCA(n_components=3).fit(ctr_orig_sorted)
        ctr_orig_pca3 = pca3.transform(ctr_orig_sorted)
        for tid, point in zip(ctr_tids_sorted, ctr_orig_pca3):
            ax_o.scatter([point[0]],[point[1]],[point[2]],
                         s=ctr_saved_size, marker=ctr_pca_marker,
                         facecolors="none", edgecolors="k", linewidths=1.2,
                         c=[trial_colors[tid]], label=f"{tid}" if tid==unique_trials[0] else None)
        # set symmetric-ish limits for aesthetics
        x_o, y_o, z_o = ctr_orig_pca3[:,0], ctr_orig_pca3[:,1], ctr_orig_pca3[:,2]
        xo = pad(np.min(x_o), np.max(x_o)); yo = pad(np.min(y_o), np.max(y_o)); zo = pad(np.min(z_o), np.max(z_o))
        ax_o.set_xlim(*xo); ax_o.set_ylim(*yo); ax_o.set_zlim(*zo)
    else:
        ax_o.text2D(0.5, 0.5, "Not enough dims for PCA-3D", ha="center", va="center", transform=ax_o.transAxes)

    ax_o.set_xlabel("PC1"); ax_o.set_ylabel("PC2"); ax_o.set_zlabel("PC3")

    # (Optional) a single legend
    # For cleanliness, we won’t flood with legends in every subplot.

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fout = os.path.join(out_dir, f"{mouse}_{maze}_UMAP_per_trial_plus_centroids3D.png")
    fig.savefig(fout, dpi=300)
    fig.savefig(fout.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {fout}")
##############################################################################################
######################COMPUTE DISPERSION METRICS #######################################################
##############################################################################################


import os, numpy as np, pandas as pd
from itertools import combinations
from scipy.spatial import ConvexHull
import learning_repo.general_utils as lrgu

# ---------------- CONFIG ----------------
data_root = os.path.join(base_dir, 'dynamical_dynamics')
out_dir   = os.path.join(data_root, 'dispersion_metrics')
os.makedirs(out_dir, exist_ok=True)

# Optionally define groups to aggregate later
mice_dict = {
    'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
    'deep':        ['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
}

# ---------------- MEB (Minimum Enclosing Ball) ----------------
# Exact smallest enclosing ball via Welzl's algorithm (any dimension)
def _ball_from(points):
    """Return (center, radius) for the unique ball defined by up to d+1 affinely independent points."""
    P = np.asarray(points, dtype=float)
    if len(P) == 0:
        return np.zeros(1), 0.0
    if len(P) == 1:
        return P[0], 0.0
    if len(P) == 2:
        c = (P[0] + P[1]) / 2.0
        r = np.linalg.norm(P[0] - c)
        return c, r
    # For k >= 3: solve circumsphere (least-squares for numerical stability)
    # Sphere: ||x - c||^2 = r^2; subtract first point to linearize.
    A = 2 * (P[1:] - P[0])                    # (k-1, d)
    b = np.sum(P[1:]**2 - P[0]**2, axis=1)    # (k-1,)
    try:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        # fallback: use centroid as center
        c = P.mean(axis=0)
    r = np.max(np.linalg.norm(P - c, axis=1))
    return c, r

def _is_in_ball(p, c, r, eps=1e-10):
    return np.linalg.norm(p - c) <= r + 1e-10 + eps

def _welzl(P, R, d):
    """
    P: list of points (mutable, pops used)
    R: list of support points on boundary (<= d+1)
    d: dimension
    """
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
    """
    Exact smallest enclosing ball for points X (n x d).
    Returns center (d,), radius (float).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n == 0:
        return np.zeros(d), 0.0
    P = [x for x in X]
    rng = np.random.default_rng(123)
    rng.shuffle(P)
    c, r = _welzl(P, [], d)
    return c, float(r)


def dispersion_metrics(points, space_name="origD", normalize_dim=False):
    pts = np.asarray(points, float)
    n, d = pts.shape
    out = {"n": n, "dim": d}
    if n == 0: ...

    mu = pts.mean(axis=0)
    dists = np.linalg.norm(pts - mu, axis=1)
    if normalize_dim and d > 0:
        dists = dists / np.sqrt(d)
    out["rms_radius"] = float(np.sqrt(np.mean(dists ** 2)))
    out["median_radius"] = float(np.median(dists))
    ...
    # For pairwise distances:
    dij = np.linalg.norm(pts[i] - pts[j])
    if normalize_dim and d > 0:
        dij /= np.sqrt(d)


# ---------------- Runner: per mouse, per maze ----------------
def compute_mouse_dispersion(mouse):
    fpath = os.path.join(data_root, mouse, f"{mouse}_dynamics_centroids.pkl")
    if not os.path.isfile(fpath):
        print(f"[{mouse}] no dynamics file; skipping")
        return None
    d = lrgu.load_pickle(os.path.dirname(fpath), os.path.basename(fpath))

    rows = []
    for maze, md in d.items():
        if "centroids" not in md:
            continue
        C_umap = md["centroids"].get("umap_space", None)
        C_orig = md["centroids"].get("orig_space", None)
        if C_umap is None or C_orig is None:
            continue
        C_umap = np.asarray(C_umap)
        C_orig = np.asarray(C_orig)
        if C_umap.ndim != 2 or C_umap.shape[1] < 3 or C_orig.ndim != 2:
            continue

        # UMAP 3D (use first 3 dims)
        M_umap = dispersion_metrics(C_umap[:, :3], space_name="umap3d")
        # Original space (D dims)
        M_orig = dispersion_metrics(C_orig, space_name="origD")

        row = {
            "mouse": mouse, "maze": maze,
            # UMAP metrics
            "umap_n": M_umap["n"], "umap_meb_radius": M_umap["meb_radius"],
            "umap_rms_radius": M_umap["rms_radius"], "umap_median_radius": M_umap["median_radius"],
            "umap_p95_radius": M_umap["p95_radius"], "umap_total_variance": M_umap["total_variance"],
            "umap_generalized_variance": M_umap["generalized_variance"],
            "umap_avg_pairwise_dist": M_umap["avg_pairwise_dist"], "umap_diameter": M_umap["diameter"],
            "umap_hull_volume": M_umap["hull_volume"],
            # Original metrics
            "orig_n": M_orig["n"], "orig_meb_radius": M_orig["meb_radius"],
            "orig_rms_radius": M_orig["rms_radius"], "orig_median_radius": M_orig["median_radius"],
            "orig_p95_radius": M_orig["p95_radius"], "orig_total_variance": M_orig["total_variance"],
            "orig_generalized_variance": M_orig["generalized_variance"],
            "orig_avg_pairwise_dist": M_orig["avg_pairwise_dist"], "orig_diameter": M_orig["diameter"],
        }
        rows.append(row)

    if not rows:
        print(f"[{mouse}] no centroid sets; nothing computed.")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{mouse}_dispersion_metrics.csv"), index=False)
    print(f"[{mouse}] saved -> {os.path.join(out_dir, f'{mouse}_dispersion_metrics.csv')}")
    return df

# ---- Run for all mice in your groups (or call compute_mouse_dispersion('GC2') directly) ----
all_rows = []
for grp, mice in mice_dict.items():
    for m in mice:
        dfm = compute_mouse_dispersion(m)
        if dfm is not None:
            dfm["group"] = grp
            all_rows.append(dfm)

if all_rows:
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, "all_mice_dispersion_metrics.csv"), index=False)
    print(f"Saved all-mice table -> {os.path.join(out_dir, 'all_mice_dispersion_metrics.csv')}")
else:
    print("No dispersion tables were generated.")


##############################################################################################
######################PLOT DISPERSSION #######################################################
##############################################################################################


# === Plot dispersion metrics (deep vs superficial), as before ===
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
# --------- PATHS ---------
data_root     = os.path.join(base_dir, 'dynamical_dynamics')
in_dir        = os.path.join(data_root, 'dispersion_metrics')
out_dir_plots = os.path.join(in_dir, 'plots')
os.makedirs(out_dir_plots, exist_ok=True)
csv_path = os.path.join(in_dir, 'all_mice_dispersion_metrics.csv')
df = pd.read_csv(csv_path)
# keep only rows with group
df = df.dropna(subset=['group']).copy()
# --------- METRICS ---------
umap_metrics = [
    'umap_meb_radius', 'umap_rms_radius', 'umap_median_radius',
    'umap_p95_radius', 'umap_total_variance', 'umap_generalized_variance',
    'umap_avg_pairwise_dist', 'umap_diameter', 'umap_hull_volume'
]
# Normalized-by-dimension original-space metrics
orig_norm_metrics = [
    'orig_meb_radius_norm', 'orig_rms_radius_norm', 'orig_median_radius_norm',
    'orig_p95_radius_norm', 'orig_total_variance_per_neuron',
    'orig_gmean_variance', 'orig_avg_pairwise_dist_norm', 'orig_diameter_norm'
]
pretty = {
    'umap_meb_radius': 'MEB radius',
    'umap_rms_radius': 'RMS radius',
    'umap_median_radius': 'Median radius',
    'umap_p95_radius': '95th % radius',
    'umap_total_variance': 'Total variance (trace Σ)',
    'umap_generalized_variance': 'Generalized variance (det Σ)',
    'umap_avg_pairwise_dist': 'Avg pairwise dist',
    'umap_diameter': 'Diameter',
    'umap_hull_volume': 'Convex hull volume',
    'orig_meb_radius_norm': 'MEB radius (per √D)',
    'orig_rms_radius_norm': 'RMS radius (per √D)',
    'orig_median_radius_norm': 'Median radius (per √D)',
    'orig_p95_radius_norm': '95th % radius (per √D)',
    'orig_total_variance_per_neuron': 'Avg variance per neuron',
    'orig_gmean_variance': 'Geom. mean variance',
    'orig_avg_pairwise_dist_norm': 'Avg pairwise (per √D)',
    'orig_diameter_norm': 'Diameter (per √D)',
}
def melt_long(df_in, metrics):
    m = df_in[['mouse','maze','group'] + metrics].melt(
        id_vars=['mouse','maze','group'],
        value_vars=metrics,
        var_name='metric',
        value_name='value'
    )
    return m[np.isfinite(m['value'])]
def welch_t(a, b):
    if len(a) > 1 and len(b) > 1:
        t, p = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        return float(t), float(p)
    return np.nan, np.nan
def plot_metric_grid(long_df, title, fname_prefix, ncols=3):
    metrics = list(long_df['metric'].unique())
    n = len(metrics)
    ncols = min(ncols, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), squeeze=False)
    fig.suptitle(title, fontsize=16)
    for i, metric in enumerate(metrics):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sub = long_df[long_df['metric'] == metric]
        sns.boxplot(data=sub, x='group', y='value', whis=1.5, fliersize=0, ax=ax)
        sns.swarmplot(data=sub, x='group', y='value', size=5, alpha=0.8, ax=ax)
        ax.set_title(pretty.get(metric, metric))
        ax.set_xlabel(''); ax.set_ylabel('')
        sup_vals  = sub.loc[sub['group']=='superficial', 'value'].values
        deep_vals = sub.loc[sub['group']=='deep',        'value'].values
        t, p = welch_t(sup_vals, deep_vals)
        if np.isfinite(p):
            ax.text(0.5, 0.95, f"t={t:.2f}, p={p:.3g}", transform=ax.transAxes,
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, linewidth=0.5))
    # hide empties
    for j in range(n, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir_plots, f"{fname_prefix}.png"), dpi=300)
    fig.savefig(os.path.join(out_dir_plots, f"{fname_prefix}.pdf"))
    plt.close(fig)
# --------- Build long DFs and plot ---------
umap_long = melt_long(df, umap_metrics)
orig_norm_long = melt_long(df, orig_norm_metrics)
plot_metric_grid(umap_long,
                 "Dispersion metrics — UMAP (3D) — superficial vs deep",
                 "dispersion_umap3d_by_group", ncols=3)
plot_metric_grid(orig_norm_long,
                 "Dispersion metrics — Original (normalized by dimension) — superficial vs deep",
                 "dispersion_original_norm_by_group", ncols=3)
print(f"Saved plots to: {out_dir_plots}")
