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
from structure_index import compute_structure_index
import elephant

from scipy.signal import convolve

from elephant.kernels import GaussianKernel
from quantities import ms
import quantities as pq

import numpy as np
from elephant.kernels import GaussianKernel
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
                print('Mouse: ' + mouse)
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
                    mov_dir_color = lrgu.get_dir_color(mov_dir)
                    time = np.arange(0,len(signal))
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

                    #compute in and out points
                    angular_pos = np.arctan2(aligned_umap_emb[1:-1,1], aligned_umap_emb[1:-1,0])*180/np.pi
                    dist_to_ellipse = np.min(pairwise_distances(aligned_umap_emb, ellipse_points),axis=1)
                    off_ring_points = dist_to_ellipse>np.percentile(dist_to_ellipse, 90)
                    in_ring_points = np.invert(off_ring_points)

                    # compute and decompose flows
                    flow_diff = 2;
                    flow = (aligned_umap_emb[flow_diff:] - aligned_umap_emb[:-flow_diff])/2
                    flow_modulus = np.linalg.norm(flow, axis=1)
                    flow_xy_angle = np.arctan2(flow[:,1], flow[:,0])*180/np.pi
                    flow_yz_angle = np.arctan2(flow[:,2], np.linalg.norm(flow[:,:2], axis=1))*180/np.pi
                    # get radial and tangential
                    tangential_flow = np.zeros(flow.shape)
                    radial_flow = np.zeros(flow.shape)
                    residual_flow = np.zeros(flow.shape)

                    radial_modulus = np.zeros(flow.shape[0])
                    tangential_modulus = np.zeros(flow.shape[0])
                    radial_modulus_signed = np.zeros(flow.shape[0])
                    closest_ellipse_idxs = np.argmin(pairwise_distances(aligned_umap_emb, ellipse_points),axis=1)
                    distance_to_ellipse = np.min(pairwise_distances(aligned_umap_emb, ellipse_points),axis=1)
                    for i in range(len(flow)):

                        emb_point = aligned_umap_emb[i,:]
                        closest_ellipse_point = ellipse_points[closest_ellipse_idxs[i],:]
                        radial_direction = emb_point - closest_ellipse_point
                        tangential_direction =  lrgeo.find_perpendicular_vector_on_xy_plane(radial_direction)

                        radial_flow[i,:] = lrgeo.project_onto_vector(flow[i,:], radial_direction)
                        tangential_flow[i,:] = lrgeo.project_onto_vector(flow[i,:], tangential_direction)
                        residual_flow[i,:] = flow[i,:] - radial_flow[i,:] - tangential_flow[i,:]

                        radial_modulus[i] = np.linalg.norm(radial_flow[i])
                        radial_sign = np.sign(np.dot(radial_flow[i], radial_direction))  # +1 if same direction, -1 if opposite
                        radial_modulus_signed[i] = radial_sign*radial_modulus[i]
                        tangential_modulus[i] = np.linalg.norm(tangential_flow[i])

                    residual_modulus = np.linalg.norm(residual_flow, axis=1)

                    in_ring_points_list = []
                    in_ring_points_list.append(in_ring_points)
                    off_ring_points_list = []
                    off_ring_points_list.append(off_ring_points)

                    # compute flux deviation
                    flow_xy_angle_list = []
                    flow_xy_angle_list.append(flow_xy_angle)
                    angular_pos_list = []
                    angular_pos_list.append(angular_pos)

                    radial_modulus_list = []
                    radial_modulus_list.append(radial_modulus)

                    tangential_modulus_list=[]
                    tangential_modulus_list.append(tangential_modulus)

                    kernels = [1,2,4,6,8,10,12,14,16,18]
                    for index , filter_size in enumerate(kernels):
                        rates = smooth_calcium_signals(signal, filter_size)
                        pca = PCA(n_components=10)
                        rates_pca = pca.fit_transform(rates)
                        vel_rates_pca = np.diff(rates_pca, axis=0)
                        rates_pca = rates_pca[:-1, :]  # skip last
                        #compute umap
                        umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist,                                                              random_state=42)
                        umap_model.fit(rates)
                        umap_emb = umap_model.transform(rates)
                        #mean center umap to facilitate future steps
                        umap_emb -= umap_emb.mean(axis=0)

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

                        #compute in and out points
                        angular_pos = np.arctan2(aligned_umap_emb[1:-1,1], aligned_umap_emb[1:-1,0])*180/np.pi
                        dist_to_ellipse = np.min(pairwise_distances(aligned_umap_emb, ellipse_points),axis=1)
                        off_ring_points = dist_to_ellipse>np.percentile(dist_to_ellipse, 90)
                        in_ring_points = np.invert(off_ring_points)

                        # compute and decompose flows
                        flow_diff = 2;
                        flow = (aligned_umap_emb[flow_diff:] - aligned_umap_emb[:-flow_diff])/2
                        flow_modulus = np.linalg.norm(flow, axis=1)
                        flow_xy_angle = np.arctan2(flow[:,1], flow[:,0])*180/np.pi
                        flow_yz_angle = np.arctan2(flow[:,2], np.linalg.norm(flow[:,:2], axis=1))*180/np.pi
                        # get radial and tangential
                        tangential_flow = np.zeros(flow.shape)
                        radial_flow = np.zeros(flow.shape)
                        residual_flow = np.zeros(flow.shape)

                        radial_modulus = np.zeros(flow.shape[0])
                        tangential_modulus = np.zeros(flow.shape[0])
                        radial_modulus_signed = np.zeros(flow.shape[0])
                        closest_ellipse_idxs = np.argmin(pairwise_distances(aligned_umap_emb, ellipse_points),axis=1)
                        distance_to_ellipse = np.min(pairwise_distances(aligned_umap_emb, ellipse_points),axis=1)
                        for i in range(len(flow)):

                            emb_point = aligned_umap_emb[i,:]
                            closest_ellipse_point = ellipse_points[closest_ellipse_idxs[i],:]
                            radial_direction = emb_point - closest_ellipse_point
                            tangential_direction =  lrgeo.find_perpendicular_vector_on_xy_plane(radial_direction)

                            radial_flow[i,:] = lrgeo.project_onto_vector(flow[i,:], radial_direction)
                            tangential_flow[i,:] = lrgeo.project_onto_vector(flow[i,:], tangential_direction)
                            residual_flow[i,:] = flow[i,:] - radial_flow[i,:] - tangential_flow[i,:]

                            radial_modulus[i] = np.linalg.norm(radial_flow[i])
                            radial_sign = np.sign(np.dot(radial_flow[i], radial_direction))  # +1 if same direction, -1 if opposite
                            radial_modulus_signed[i] = radial_sign*radial_modulus[i]
                            tangential_modulus[i] = np.linalg.norm(tangential_flow[i])

                        residual_modulus = np.linalg.norm(residual_flow, axis=1)

                        in_ring_points_list.append(in_ring_points)
                        off_ring_points_list.append(off_ring_points)
                        flow_xy_angle_list.append(flow_xy_angle)
                        angular_pos_list.append(angular_pos)
                        radial_modulus_list.append(radial_modulus)
                        tangential_modulus_list.append(tangential_modulus)


                    dynamics_dict[maze] = {
                        'signal': signal,
                        'pos': pos,
                        'angular_pos': angular_pos_list,
                        'speed': speed,
                        'trial_id': trial_id,
                        'mov_dir': mov_dir,

                        'in_ring_points': in_ring_points_list,
                        'off_ring_points': off_ring_points_list,

                        'flow_xy_angle': flow_xy_angle_list,
                        'flow_yz_angle': flow_yz_angle,

                        'radial_modulus': radial_modulus_list,
                        'tangential_modulus': tangential_modulus_list,
                    }
                pickle_name = f"{mouse}_dynamics_filters_0-18.pkl"
                lrgu.save_pickle(msave_dir,pickle_name, dynamics_dict)



#__________________________________________________________________________
#|                                                                        |#
#|                  PLOT   FLOW XY ANGLE VS ANGULAR POS                   |#
#|________________________________________________________________________|#

import seaborn as sns
import pandas as pd
from scipy import stats

data_dir = os.path.join(base_dir, 'results')
filters = np.array([0,1,2,4,6,8,10,12,14,16,18])*50

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }
mice_area = list(mice_dict.keys())
signal_name = 'clean_traces'

line_1 = np.column_stack((np.linspace(-180, 90, 1000), np.linspace(-90, 180, 1000)))
line_2 = np.column_stack((np.linspace(90, 180, 1000), np.linspace(-180, -90, 1000)))
ideal_dist = np.concatenate((line_1, line_2), axis=0)

area_list = []
mouse_name_list = []
session_list = []
flow_output_list = []
filter_size_list = []

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = data_dir
        file_name =  f"{mouse}_dynamics_filters_0-18.pkl"
        print(file_name)
        dynamic_dict = lrgu.load_pickle(mdata_dir,file_name)
        session_names = list(dynamic_dict.keys())
        session_names.sort()
        for idx, session in enumerate(session_names):
            if 'lt' in session:
                session_type = 'lt'
            if 'rot' in session:
                session_type = 'rot'
            for i, k in enumerate(filters):
                area_list.append(area)
                session_list.append(session_type)
                mouse_name_list.append(mouse)
                filter_size_list.append(k)
                in_ring_points = dynamic_dict[session]['in_ring_points'][i]
                off_ring_points = dynamic_dict[session]['off_ring_points'][i]
                #compute flux deviation
                flow_xy_angle = dynamic_dict[session]['flow_xy_angle'][i][in_ring_points[1:-1]]
                angular_pos = dynamic_dict[session]['angular_pos'][i][in_ring_points[1:-1]]
                real_dist = np.column_stack((angular_pos, flow_xy_angle))
                diff_dist = np.min(pairwise_distances(real_dist, ideal_dist), axis=1)
                flow_output_list.append(diff_dist.mean())

flow_pd = pd.DataFrame(data={'area':area_list,
                            'mouse': mouse_name_list,
                             'session': session_list,
                             'filter':filter_size_list,
                             'flow_pos_diff': flow_output_list})

palette = ['purple', 'yellow']  # Define your own list of colors

fig, axes = plt.subplots(1,1, figsize = (8,5))
session_selected = flow_pd[flow_pd['session'] == 'lt']
sns.barplot(session_selected, x="filter", y="flow_pos_diff", hue="area", palette=palette, ax=axes)
sns.stripplot(session_selected, x='filter', y='flow_pos_diff', hue='area', palette = ['k','k'],dodge=True, size=5, jitter=True)
plt.savefig(os.path.join(data_dir, f'flow_pos_diff_filters_{signal_name}.png'), dpi=400, bbox_inches="tight")



#__________________________________________________________________________
#|                                                                        |#
#|                          PLOT   RAD VS TAN                             |#
#|________________________________________________________________________|#

area_list = []
mouse_name_list = []
session_list = []
rad_tan_ratio = []
rad_modulus_list = []
tan_modulus_list = []
ring_location = []
filter_size_list = []

for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = data_dir
        file_name =  f"{mouse}_dynamics_filters_0-18.pkl"
        print(file_name)
        dynamic_dict = lrgu.load_pickle(mdata_dir,file_name)
        session_names = list(dynamic_dict.keys())
        session_names.sort()
        for idx, session in enumerate(session_names):
            if 'lt' in session:
                session_type = 'lt'
            if 'rot' in session:
                session_type = 'rot'

            for i, k in enumerate(filters):
                mouse_name_list += [mouse] * 2
                session_list += [session_type] * 2
                area_list += [area] * 2
                filter_size_list+= [k] * 2

                in_ring_points = dynamic_dict[session]['in_ring_points'][i]
                off_ring_points = dynamic_dict[session]['off_ring_points'][i]

                # compute flux deviation
                rad_modulus = dynamic_dict[session]['radial_modulus'][i]
                tan_modulus = dynamic_dict[session]['tangential_modulus'][i]

                mean_in_rad = rad_modulus[in_ring_points[1:-1]].mean()
                mean_in_tan = tan_modulus[in_ring_points[1:-1]].mean()

                mean_off_rad = rad_modulus[off_ring_points[1:-1]].mean()
                mean_off_tan = tan_modulus[off_ring_points[1:-1]].mean()

                rad_modulus_list += [mean_in_rad, mean_off_rad]
                tan_modulus_list += [mean_in_tan, mean_off_tan]
                rad_tan_ratio += [mean_in_rad / mean_in_tan, mean_off_rad / mean_off_tan]
                ring_location += ['in', 'off']


flow_pd = pd.DataFrame(data={
                    'area':area_list,
                    'mouse': mouse_name_list,
                     'session': session_list,
                    'filter':filter_size_list,
                     'rad_modulus': rad_modulus_list,
                     'tan_modulus': tan_modulus_list,
                     'rad_tan_ratio': rad_tan_ratio,
                     'ring_location': ring_location})



palette = ['purple', 'yellow']  # Define your own list of colors

fig = plt.figure(figsize = (20,8))
ax = plt.subplot(321)
session_selected = flow_pd[flow_pd['session'] == 'lt']
session_selected_ = session_selected[session_selected['ring_location'] == 'in']
sns.barplot(x = 'filter', y = 'rad_modulus', hue = 'area', data = session_selected_, ax = ax, palette=palette)
sns.stripplot(session_selected_, x='filter', y='rad_modulus', hue='area', palette = ['k','k'],dodge=True, size=5, jitter=True, ax = ax)
ax.set_title('IN')
ax.legend('[]')

ax = plt.subplot(322)
session_selected_ = session_selected[session_selected['ring_location'] == 'off']
sns.barplot(x = 'filter', y = 'rad_modulus', hue = 'area', data = session_selected_, ax = ax, palette=palette)
sns.stripplot(session_selected_, x='filter', y='rad_modulus', hue='area', palette = ['k','k'],dodge=True, size=5, jitter=True, ax = ax)
ax.set_title('OFF')
ax.legend('[]')

ax = plt.subplot(323)
session_selected = flow_pd[flow_pd['session'] == 'lt']
session_selected_ = session_selected[session_selected['ring_location'] == 'in']
sns.barplot(x = 'filter', y = 'tan_modulus', hue = 'area', data = session_selected_, ax = ax, palette=palette)
sns.stripplot(session_selected_, x='filter', y='tan_modulus', hue='area', palette = ['k','k'],dodge=True, size=5, jitter=True, ax = ax)
ax.set_title('IN')
ax.legend('[]')

ax = plt.subplot(324)
session_selected_ = session_selected[session_selected['ring_location'] == 'off']
sns.barplot(x = 'filter', y = 'tan_modulus', hue = 'area', data = session_selected_, ax = ax, palette=palette)
sns.stripplot(session_selected_, x='filter', y='tan_modulus', hue='area', palette = ['k','k'],dodge=True, size=5, jitter=True, ax = ax)
ax.set_title('OFF')
ax.legend('[]')

ax = plt.subplot(325)
session_selected = flow_pd[flow_pd['session'] == 'lt']
session_selected_ = session_selected[session_selected['ring_location'] == 'in']
sns.barplot(x = 'filter', y = 'rad_tan_ratio', hue = 'area', data = session_selected_, ax = ax, palette=palette)
sns.stripplot(session_selected_, x='filter', y='rad_tan_ratio', hue='area', palette = ['k','k'],dodge=True, size=5, jitter=True, ax = ax)
ax.set_title('IN')
ax.legend('[]')

ax = plt.subplot(326)
session_selected_ = session_selected[session_selected['ring_location'] == 'off']
sns.barplot(x = 'filter', y = 'rad_tan_ratio', hue = 'area', data = session_selected_, ax = ax, palette=palette)
sns.stripplot(session_selected_, x='filter', y='rad_tan_ratio', hue='area', palette = ['k','k'],dodge=True, size=5, jitter=True, ax = ax)
ax.set_title('OFF')
ax.legend('[]')

fig.suptitle(f'rad_tan_idx_filters_{signal_name}')
plt.savefig(os.path.join(data_dir, f'rad_tan_idx_filters_sep_{signal_name}.png'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'rad_tan_idx_filters_sep_{signal_name}.svg'), dpi = 400,bbox_inches="tight")


##################################################3
fig = plt.figure(figsize = (15,5))
ax = plt.subplot(131)
sns.barplot(x = 'session', y = 'rad_modulus', hue = 'area', data = flow_pd, ax = ax, palette=palette)
ax = plt.subplot(132)
sns.barplot(x = 'session', y = 'tan_modulus', hue = 'area', data = flow_pd, ax = ax, palette=palette)
ax = plt.subplot(133)
sns.barplot(x = 'session', y = 'rad_tan_ratio', hue = 'area', data = flow_pd, ax = ax, palette=palette)
fig.suptitle(f'rad_tan_idx_{signal_name}')
plt.savefig(os.path.join(data_dir, f'rad_tan_idx_sep_session_{signal_name}.png'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'rad_tan_idx_sep_session_{signal_name}.svg'), dpi = 400,bbox_inches="tight")
####################################################
