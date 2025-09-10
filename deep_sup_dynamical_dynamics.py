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


                    #clean mov dir to avoid gaps in ring
                    temp_mov = np.diff(pos,axis=0)*20
                    temp_mov = np.concatenate([temp_mov[0,:].reshape(-1,2), temp_mov], axis= 0)[:,0]
                    temp_mov = gaussian_filter1d(temp_mov, sigma = 5, axis = 0)
                    temp_mov[temp_mov<0] = -1
                    temp_mov = np.sign(temp_mov)
                    mov_dir[mov_dir==0] = temp_mov[np.where(mov_dir==0)[0]]

                    #compute umap
                    umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist,                                                              random_state=42)
                    umap_model.fit(signal)
                    umap_emb = umap_model.transform(signal)

                    #mean center umap to facilitate future steps
                    umap_emb -= umap_emb.mean(axis=0)

                    # aling ring to 3D: make it horizontal


                    centroids, centroids_pos = lrgeo.get_centroids(umap_emb, pos[:,0], mov_dir, n_centroids=80, n_dimensions= 3)
                    umap_norm_vector, cloud_center = lrgeo.parametrize_plane(umap_emb)

                    horizontal_angle, horizontal_rotation_matrix = lrgeo.align_vectors(np.array([0,0,1]), np.zeros(3), umap_norm_vector, cloud_center)
                    horizontal_umap_emb = lrgeo.apply_rotation_to_cloud(umap_emb, horizontal_rotation_matrix, umap_emb.mean(axis=0))
                    horizontal_umap_emb -= horizontal_umap_emb.mean(axis=0)
                    horizontal_centroids, _ = lrgeo.get_centroids(horizontal_umap_emb, pos[:,0], mov_dir, n_centroids=80)

                    # aling ring to 3D: compute pos-gradient
                    centroids2D_pos = np.unique(centroids_pos)
                    centroids2D = np.array([np.nanmean(horizontal_centroids[centroids_pos[:, 0] == val, :], axis=0)
                                            for val in centroids2D_pos])

                    ### no logro entender cual es el objetivo de computar PCA en ese caso. La idea es computar los gradientes cierto ?
                    pos_gradient_pca = PCA(3)
                    pos_gradient_pca.fit(centroids2D)
                    pos_gradient_axis = pos_gradient_pca.components_[0]

                    # ensure correct position gradient
                    aligned_centroids2D = pos_gradient_pca.transform(centroids2D)
                    low, high = aligned_centroids2D[centroids2D_pos < np.percentile(pos[:, 0], 30)], \
                                aligned_centroids2D[centroids2D_pos > np.percentile(pos[:, 0], 70)]
                    if np.mean(low[:,0])>np.mean(high[:,0]):
                        pos_gradient_axis *= -1

                    # aling ring to 3D :rotate so pos-gradient is x-axis
                    pos_angle, pos_rotation_matrix = lrgeo.align_vectors(-np.array([1,0,0]), np.zeros(3), pos_gradient_axis, np.zeros(3))
                    aligned_umap_emb = lrgeo.apply_rotation_to_cloud(horizontal_umap_emb, pos_rotation_matrix, horizontal_umap_emb.mean(axis=0))
                    aligned_centroids = lrgeo.apply_rotation_to_cloud(horizontal_centroids, pos_rotation_matrix, horizontal_centroids.mean(axis=0))

                     # ensure correct direction gradient (left top, right bottom)
                    left, right = aligned_umap_emb[mov_dir==-1,1], \
                                aligned_umap_emb[mov_dir==+1,1]

                    if np.mean(left)<np.mean(right):
                        aligned_umap_emb[:,1] *= -1
                        aligned_centroids[:,1] *= -1

                    # aling ring to 3D :plot alignemnt
                    fig = lrgeo.plot_ring_alignment(
                        umap_emb, pos, mov_dir, trial_id, centroids, centroids_pos,
                        umap_norm_vector, horizontal_umap_emb, horizontal_centroids,
                        pos_gradient_axis, centroids2D, centroids2D_pos, aligned_umap_emb, aligned_centroids2D)

                    fig.suptitle(f'{mouse} ring alignment')
                    figure_name = f"{mouse}_{maze}_ring_alignment"
                    plt.savefig(os.path.join(msave_dir, figure_name+".svg"), dpi = 400,bbox_inches="tight")
                    plt.savefig(os.path.join(msave_dir, figure_name+".png"), dpi = 400,bbox_inches="tight")
                    plt.close(fig)

                    # fit elliposid
                    ellipse_points, ellipse_coeff = lrgeo.fit_ellipse_to_point_cloud(aligned_umap_emb, 10000, 300)

                    # assign to each point a pos and direction value according to closest k-neighbors
                    k = 300
                    ellipse_features = np.zeros((ellipse_points.shape[0],2))
                    for i in range(ellipse_points.shape[0]):
                        point = ellipse_points[i, :2]
                        d = ((aligned_umap_emb[:,:2] - point)**2).sum(axis=1)
                        neigh_idx = d.argsort()[:k]
                        ellipse_features[i,:] = [np.mean(pos[neigh_idx,0]), mode(mov_dir[neigh_idx].T.tolist())]

                    # ----------------------------
                    # DYNAMIC (SLIDING-WINDOW) METRICS
                    # ----------------------------

                    def compute_window_metrics(
                            aligned_umap_emb,
                            pos, speed, trial_id, mov_dir,
                            ellipse_points,
                            flow_diff=2,
                            k_neigh_for_ellipse_features=300,
                            offring_percentile=90
                    ):
                        """
                        Recompute the 'from here on' metrics (in/out points, angular_pos,
                        flow & its radial/tangential decomposition, etc.) for a given slice.
                        Returns a dict mirroring your static keys but only for the slice.
                        """
                        # --- Angles & ring proximity within the window ---
                        # (match original: angle only on indices [1:-1] then re-align sizes)
                        # For a self-contained window we do the same approach:
                        angular_pos = np.arctan2(aligned_umap_emb[1:-1, 1], aligned_umap_emb[1:-1, 0]) * 180 / np.pi
                        # pad to length N with NaNs at ends to preserve alignment with the window indices
                        N = aligned_umap_emb.shape[0]
                        angular_pos = np.concatenate([[np.nan], angular_pos, [np.nan]])

                        # distances to global ellipse (fixed)
                        dist_to_ellipse_full = pairwise_distances(aligned_umap_emb, ellipse_points)
                        closest_ellipse_idxs = np.argmin(dist_to_ellipse_full, axis=1)
                        dist_to_ellipse = dist_to_ellipse_full[np.arange(N), closest_ellipse_idxs]

                        # per-window threshold for in/out classification
                        thr = np.percentile(dist_to_ellipse, offring_percentile)
                        off_ring_points = dist_to_ellipse > thr
                        in_ring_points = ~off_ring_points

                        # --- Flow & decomposition (same as your static code, local to window) ---
                        # forward-backward difference inside the window
                        if N <= 2 * flow_diff:
                            # too short; return mostly NaNs to keep structure predictable
                            flow = np.full((max(N - flow_diff, 0), 3), np.nan)
                            flow_modulus = np.full((max(N - flow_diff, 0),), np.nan)
                            flow_xy_angle = np.full((max(N - flow_diff, 0),), np.nan)
                            flow_yz_angle = np.full((max(N - flow_diff, 0),), np.nan)
                            tangential_flow = np.full_like(flow, np.nan)
                            radial_flow = np.full_like(flow, np.nan)
                            residual_flow = np.full_like(flow, np.nan)
                            radial_modulus = np.full((max(N - flow_diff, 0),), np.nan)
                            tangential_modulus = np.full((max(N - flow_diff, 0),), np.nan)
                            radial_modulus_signed = np.full((max(N - flow_diff, 0),), np.nan)
                            residual_modulus = np.full((max(N - flow_diff, 0),), np.nan)
                        else:
                            flow = (aligned_umap_emb[flow_diff:] - aligned_umap_emb[:-flow_diff]) / 2
                            flow_modulus = np.linalg.norm(flow, axis=1)
                            flow_xy_angle = np.arctan2(flow[:, 1], flow[:, 0]) * 180 / np.pi
                            flow_yz_angle = np.arctan2(flow[:, 2], np.linalg.norm(flow[:, :2], axis=1)) * 180 / np.pi

                            tangential_flow = np.zeros_like(flow)
                            radial_flow = np.zeros_like(flow)
                            residual_flow = np.zeros_like(flow)

                            radial_modulus = np.zeros(flow.shape[0])
                            tangential_modulus = np.zeros(flow.shape[0])
                            radial_modulus_signed = np.zeros(flow.shape[0])

                            # prepare local nearest ellipse data for window points used in flow
                            # the flow vectors correspond to points i in [0 .. N-1-flow_diff]
                            # We'll align projections to point i (start of the symmetric diff).
                            # Use precomputed closest_ellipse_idxs & ellipse_points.
                            for i in range(flow.shape[0]):
                                emb_point = aligned_umap_emb[i, :]
                                ce_idx = closest_ellipse_idxs[i]
                                closest_ellipse_point = ellipse_points[ce_idx, :]

                                radial_direction = emb_point - closest_ellipse_point
                                tangential_direction = lrgeo.find_perpendicular_vector_on_xy_plane(radial_direction)

                                # project
                                radial_flow[i, :] = lrgeo.project_onto_vector(flow[i, :], radial_direction)
                                tangential_flow[i, :] = lrgeo.project_onto_vector(flow[i, :], tangential_direction)
                                residual_flow[i, :] = flow[i, :] - radial_flow[i, :] - tangential_flow[i, :]

                                radial_modulus[i] = np.linalg.norm(radial_flow[i])
                                radial_sign = np.sign(np.dot(radial_flow[i], radial_direction))
                                radial_modulus_signed[i] = radial_sign * radial_modulus[i]
                                tangential_modulus[i] = np.linalg.norm(tangential_flow[i])

                            residual_modulus = np.linalg.norm(residual_flow, axis=1)

                        return {
                            # inputs sliced externally are implied; we still return aligned sizes:
                            'angular_pos': angular_pos,  # (N,) with NaN at ends
                            'dist_to_ellipse': dist_to_ellipse,  # (N,)
                            'closest_ellipse_idxs': closest_ellipse_idxs,  # (N,)
                            'in_ring_points': in_ring_points,  # (N,)
                            'off_ring_points': off_ring_points,  # (N,)

                            'flow': flow,
                            'flow_modulus': flow_modulus,
                            'flow_xy_angle': flow_xy_angle,
                            'flow_yz_angle': flow_yz_angle,

                            'tangential_flow': tangential_flow,
                            'radial_flow': radial_flow,
                            'residual_flow': residual_flow,

                            'radial_modulus': radial_modulus,
                            'radial_modulus_signed': radial_modulus_signed,
                            'tangential_modulus': tangential_modulus,
                            'residual_modulus': residual_modulus
                        }


                    # --- Configure your sliding window ---
                    window_size = 500  # <- tune to your sampling rate / dynamics
                    step = 250  # <- stride between windows
                    flow_diff_win = 2
                    offring_percentile_win = 90
                    k_neighbors_ellipse = 300  # only used if you also want per-window ellipse features (optional)

                    N_total = aligned_umap_emb.shape[0]
                    dynamic_windows = []

                    # Slide over time indices
                    start = 0
                    while start < N_total:
                        end = min(start + window_size, N_total)
                        sl = slice(start, end)

                        # Slice signals for the window
                        emb_w = aligned_umap_emb[sl]
                        pos_w = pos[sl]
                        speed_w = speed[sl]
                        trial_w = trial_id[sl]
                        mov_w = mov_dir[sl]

                        # Compute window metrics (using global ellipse_points)
                        win_metrics = compute_window_metrics(
                            emb_w, pos_w, speed_w, trial_w, mov_w,
                            ellipse_points=ellipse_points,
                            flow_diff=flow_diff_win,
                            k_neigh_for_ellipse_features=k_neighbors_ellipse,
                            offring_percentile=offring_percentile_win
                        )

                        dynamic_windows.append({
                            'start_idx': int(start),
                            'end_idx': int(end),
                            # you can also include timestamps if you have them
                            'metrics': win_metrics
                        })

                        if end == N_total:
                            break
                        start += step

                    # Save into the same structure you’re already writing
                    if 'dynamic' not in dynamics_dict[maze]:
                        dynamics_dict[maze]['dynamic'] = {}
                    dynamics_dict[maze]['dynamic'].update({
                        'window_params': {
                            'window_size': window_size,
                            'step': step,
                            'flow_diff': flow_diff_win,
                            'offring_percentile': offring_percentile_win
                        },
                        'windows': dynamic_windows
                    })

                pickle_name = f"{mouse}_dynamics_dynamic_dict.pkl"
                lrgu.save_pickle(msave_dir, pickle_name, dynamics_dict)

###################################################################################################################
###################################################################################################################
#__________________________________________________________________________
#|                                                                        |#
#|     GROUPED PLOTS: FLOW XY ANGLE vs ANGULAR POS (DEEP vs SUPERFICIAL)  |#
#|________________________________________________________________________|#
import os, numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from statistics import mode
# ----------------- CONFIG -----------------
data_dir = os.path.join(base_dir, 'dynamical_dynamics')
out_dir  = os.path.join(data_dir, 'group_flow_evolution')
os.makedirs(out_dir, exist_ok=True)
mice_dict = {
    'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
    'deep':        ['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
}
signal_name = 'clean_traces'
smooth_win = 3      # rolling window for smoothing, in #windows
use_seconds = False # set True if you saved sf and want time in seconds
# --- Ideal relation ---
line_1 = np.column_stack((np.linspace(-180,  90, 1000), np.linspace(-90, 180, 1000)))
line_2 = np.column_stack((np.linspace(  90, 180, 1000), np.linspace(-180,-90, 1000)))
ideal_dist = np.concatenate((line_1, line_2), axis=0)
def compute_window_flow_pos_diff(ang, in_ring, flow_xy_angle):
    ang_core = ang[1:-1]
    in_ring_core = in_ring[1:-1]
    L = min(len(flow_xy_angle), len(ang_core))
    if L == 0:
        return np.nan
    ang_core = ang_core[:L]
    in_ring_core = in_ring_core[:L]
    flow_core = flow_xy_angle[:L]
    mask = (~np.isnan(ang_core)) & (~np.isnan(flow_core)) & (in_ring_core.astype(bool))
    if not np.any(mask):
        return np.nan
    real_dist = np.column_stack((ang_core[mask], flow_core[mask]))
    diff_dist = np.min(pairwise_distances(real_dist, ideal_dist), axis=1)
    return float(np.mean(diff_dist))
def load_mouse_evo(mouse, area):
    mdata_dir = os.path.join(data_dir, mouse)
    file_name = f"{mouse}_dynamics_dynamic_dict.pkl"
    dynamic_dict = lrgu.load_pickle(mdata_dir, file_name)
    rows = []
    session_names = sorted([k for k in dynamic_dict.keys() if isinstance(dynamic_dict[k], dict)])
    for session in session_names:
        sess_dict = dynamic_dict[session]
        if 'dynamic' not in sess_dict:
            continue
        session_type = 'lt' if 'lt' in session else ('rot' if 'rot' in session else 'unknown')
        #trial_id_full = sess_dict['trial_id'].ravel()
        for w_idx, w in enumerate(sess_dict['dynamic']['windows']):
            start_idx, end_idx = int(w['start_idx']), int(w['end_idx'])
            center_idx = (start_idx + end_idx) // 2
            try:
                center_trial = int(mode(trial_id_full[start_idx:end_idx]))
            except Exception:
                center_trial = None
            met = w['metrics']
            val = compute_window_flow_pos_diff(
                ang=np.asarray(met['angular_pos']),
                in_ring=np.asarray(met['in_ring_points']),
                flow_xy_angle=np.asarray(met['flow_xy_angle'])
            )
            rows.append({
                'area': area,
                'mouse': mouse,
                'session': session_type,
                'session_name': session,
                'window_idx': w_idx,
                'center_sample': center_idx,
                'flow_pos_diff': val
            })
    df = pd.DataFrame(rows)
    df['x'] = df['center_sample']
    return df
def maybe_smooth(y, win):
    if win and win > 1 and len(y) >= win:
        return pd.Series(y).rolling(win, min_periods=1, center=True).median().to_numpy()
    return np.asarray(y)
# ----------------- PLOT GROUPS -----------------
sns.set_context("talk")
for area in ['superficial', 'deep']:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for mouse in mice_dict[area]:
        evo_df = load_mouse_evo(mouse, area)
        if evo_df.empty:
            continue
        # aggregate across sessions (plot each session separately if you want)
        for sess_name, sub in evo_df.groupby('session_name'):
            sub = sub.sort_values('x')
            x = sub['x'].to_numpy()
            y = sub['flow_pos_diff'].to_numpy()
            y_s = maybe_smooth(y, smooth_win)
            ax.plot(x, y_s, linewidth=2, alpha=0.8, label=f"{mouse}-{sess_name}")
    ax.set_title(f"{area.capitalize()} animals — Flow XY angle vs Angular Pos (evolution)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Flow–Pos deviation")
    ax.set_ylim(0, 60)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, f"{area}_flow_pos_diff_evolution_{signal_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

#__________________________________________________________________________
#|                                                                        |#
#|   GROUPED PLOTS: RADIAL vs TANGENTIAL FLOW (IN vs OFF, by GROUP AREA)  |#
#|________________________________________________________________________|#
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from statistics import mode
# ----------------- CONFIG -----------------
data_dir = os.path.join(base_dir, 'dynamical_dynamics')
out_dir  = os.path.join(data_dir, 'group_rad_tan_evolution_in_off')
os.makedirs(out_dir, exist_ok=True)
mice_dict = {
    'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
    'deep':        ['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
}
signal_name = 'clean_traces'
smooth_win = 3      # rolling median smoothing (in #windows)
# ----------------- HELPERS -----------------
def maybe_smooth(y, win):
    if win and win > 1 and len(y) >= win:
        return pd.Series(y).rolling(win, min_periods=1, center=True).median().to_numpy()
    return np.asarray(y)
def load_mouse_rad_tan(mouse, area):
    """Extract per-window radial/tangential stats for one mouse."""
    mdata_dir = os.path.join(data_dir, mouse)
    file_name = f"{mouse}_dynamics_dynamic_dict.pkl"
    dynamic_dict = lrgu.load_pickle(mdata_dir, file_name)
    rows = []
    session_names = sorted([k for k in dynamic_dict.keys() if isinstance(dynamic_dict[k], dict)])
    for session in session_names:
        sess_dict = dynamic_dict[session]
        if 'dynamic' not in sess_dict:
            continue
        session_type = 'lt' if 'lt' in session else ('rot' if 'rot' in session else 'unknown')
        #trial_id_full = sess_dict['trial_id'].ravel()
        for w_idx, w in enumerate(sess_dict['dynamic']['windows']):
            start_idx, end_idx = int(w['start_idx']), int(w['end_idx'])
            center_idx = (start_idx + end_idx) // 2
            try:
                center_trial = int(mode(trial_id_full[start_idx:end_idx]))
            except Exception:
                center_trial = None
            met = w['metrics']
            rad = np.asarray(met['radial_modulus'])
            tan = np.asarray(met['tangential_modulus'])
            in_ring = np.asarray(met['in_ring_points'])
            off_ring = np.asarray(met['off_ring_points'])
            def safe_mean(arr):
                return np.nan if arr.size == 0 else float(np.nanmean(arr))
            mean_in_rad = safe_mean(rad[in_ring[1:-1]])
            mean_in_tan = safe_mean(tan[in_ring[1:-1]])
            mean_off_rad = safe_mean(rad[off_ring[1:-1]])
            mean_off_tan = safe_mean(tan[off_ring[1:-1]])
            rows.extend([
                {
                    'area': area, 'mouse': mouse, 'session': session_type,
                    'session_name': session, 'window_idx': w_idx,
                    'center_sample': center_idx, 'ring_location': 'in',
                    'rad_modulus': mean_in_rad, 'tan_modulus': mean_in_tan,
                    'rad_tan_ratio': (mean_in_rad / mean_in_tan) if (mean_in_tan not in [0, np.nan]) else np.nan
                },
                {
                    'area': area, 'mouse': mouse, 'session': session_type,
                    'session_name': session, 'window_idx': w_idx,
                    'center_sample': center_idx, 'ring_location': 'off',
                    'rad_modulus': mean_off_rad, 'tan_modulus': mean_off_tan,
                    'rad_tan_ratio': (mean_off_rad / mean_off_tan) if (mean_off_tan not in [0, np.nan]) else np.nan
                }
            ])
    return pd.DataFrame(rows)
# ----------------- PLOT -----------------
sns.set_context("talk")
for area in ['superficial', 'deep']:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for j, loc in enumerate(['in', 'off']):
        ax = axes[j]
        for mouse in mice_dict[area]:
            df_mouse = load_mouse_rad_tan(mouse, area)
            if df_mouse.empty:
                continue
            sub = df_mouse[df_mouse['ring_location'] == loc].sort_values('center_sample')
            if sub.empty:
                continue
            x = sub['center_sample'].to_numpy()
            y = sub['rad_tan_ratio'].to_numpy()
            y_s = maybe_smooth(y, smooth_win)
            ax.plot(x, y_s, linewidth=2, alpha=0.8, label=mouse)
        ax.set_title(f"{area.capitalize()} — {loc.upper()} ring")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Rad/Tan ratio")
        ax.set_ylim(0, 2)
        ax.grid(True, alpha=0.3)
        if j == 1:  # only add legend once
            ax.legend(fontsize=8, ncol=1)
    fig.suptitle(f"{area.capitalize()} animals — Rad/Tan ratio evolution", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, f"{area}_rad_tan_evolution_{signal_name}.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


