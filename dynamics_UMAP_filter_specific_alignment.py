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

num_neigh = 120
dim = 3
min_dist = 0.1

data_dir = os.path.join(base_dir, 'results','dynamic_dict')
save_dir = os.path.join(base_dir, 'results','filtered_alignment')

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

mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
            'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
            }

filter = 8
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
for area in mice_area:
    mice_list = mice_dict[area]
    for mouse in mice_list:
        mdata_dir = data_dir
        msave_dir = os.path.join(save_dir, mouse)  # mouse save dir
        if not os.path.isdir(msave_dir): os.makedirs(msave_dir)

        file_name =  f"{mouse}_dynamic_dict.pkl"
        print(file_name)
        dynamic_dict = lrgu.load_pickle(mdata_dir,file_name)
        session_names = list(dynamic_dict.keys())
        session_names.sort()
        new_dynamics_dict = {}
        for idx, session in enumerate(session_names):
            area_list.append(area)
            mouse_name_list.append(mouse)
            if 'lt' in session:
                session_type = 'lt'
            if 'rot' in session:
                session_type = 'rot'
            session_list.append(session_type)

            signal = dynamic_dict[session]['signal']
            rates = smooth_calcium_signals(signal, filter)
            #pca = PCA(n_components=10)
            #rates_pca = pca.fit_transform(rates)
            #vel_rates_pca = np.diff(rates_pca, axis=0)
            #rates_pca = rates_pca[:-1, :]  # skip last

            mdata_dir = data_dir
            out_file_name = f"{mouse}_dynamic_dict_UMAP_{filter}.pkl"

            # compute umap
            umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist, random_state=42)
            umap_model.fit(rates)
            umap_emb_real = umap_model.transform(dynamic_dict[session]['signal'])
            # mean center umap to facilitate future steps
            umap_emb = umap_emb_real - umap_emb_real.mean(axis=0)
            mov_dir_color = lrgu.get_dir_color(dynamic_dict[session]['mov_dir'])
            trial_id = dynamic_dict[session]['trial_id']

            time = np.arange(0,umap_emb_real[:, 0].shape[0])

            # aling ring to 3D: make it horizontal
            mov_dir = dynamic_dict[session]['mov_dir']
            pos = dynamic_dict[session]['pos']
            trial_id = trial_id
            speed = np.diff(pos[:,0])

            centroids, centroids_pos = lrgeo.get_centroids(umap_emb, pos[:, 0], mov_dir, n_centroids=80, n_dimensions=3)
            umap_norm_vector, cloud_center = lrgeo.parametrize_plane(umap_emb)

            horizontal_angle, horizontal_rotation_matrix = lrgeo.align_vectors(np.array([0, 0, 1]), np.zeros(3),
                                                                               umap_norm_vector, cloud_center)
            horizontal_umap_emb = lrgeo.apply_rotation_to_cloud(umap_emb, horizontal_rotation_matrix,
                                                                umap_emb.mean(axis=0))
            horizontal_umap_emb -= horizontal_umap_emb.mean(axis=0)
            horizontal_centroids, _ = lrgeo.get_centroids(horizontal_umap_emb, pos[:, 0], mov_dir, n_centroids=80)

            # aling ring to 3D: compute pos-gradient
            centroids2D_pos = np.unique(centroids_pos)
            centroids2D = np.array([np.nanmean(horizontal_centroids[centroids_pos[:, 0] == val, :], axis=0)
                                    for val in centroids2D_pos])
            pos_gradient_pca = PCA(3)
            pos_gradient_pca.fit(centroids2D)
            pos_gradient_axis = pos_gradient_pca.components_[0]

            # ensure correct position gradient
            aligned_centroids2D = pos_gradient_pca.transform(centroids2D)
            low, high = aligned_centroids2D[centroids2D_pos < np.percentile(pos[:, 0], 30)], \
                aligned_centroids2D[centroids2D_pos > np.percentile(pos[:, 0], 70)]
            if np.mean(low[:, 0]) > np.mean(high[:, 0]):
                pos_gradient_axis *= -1

            # aling ring to 3D :rotate so pos-gradient is x-axis
            pos_angle, pos_rotation_matrix = lrgeo.align_vectors(-np.array([1, 0, 0]), np.zeros(3), pos_gradient_axis,
                                                                 np.zeros(3))
            aligned_umap_emb = lrgeo.apply_rotation_to_cloud(horizontal_umap_emb, pos_rotation_matrix,
                                                             horizontal_umap_emb.mean(axis=0))
            aligned_centroids = lrgeo.apply_rotation_to_cloud(horizontal_centroids, pos_rotation_matrix,
                                                              horizontal_centroids.mean(axis=0))

            # ensure correct direction gradient (left top, right bottom)
            left, right = aligned_umap_emb[mov_dir == -1, 1], \
                aligned_umap_emb[mov_dir == +1, 1]

            if np.mean(left) < np.mean(right):
                aligned_umap_emb[:, 1] *= -1
                aligned_centroids[:, 1] *= -1

            # aling ring to 3D :plot alignemnt
            fig = plot_ring_alignment(
                umap_emb, pos, mov_dir, trial_id, centroids, centroids_pos,
                umap_norm_vector, horizontal_umap_emb, horizontal_centroids,
                pos_gradient_axis, centroids2D, centroids2D_pos, aligned_umap_emb, aligned_centroids2D)

            fig.suptitle(f'{mouse} ring alignment')
            figure_name = f"{mouse}_ring_alignment_filterd_{filter}"
            plt.savefig(os.path.join(msave_dir, figure_name + ".svg"), dpi=400, bbox_inches="tight")
            plt.savefig(os.path.join(msave_dir, figure_name + ".png"), dpi=400, bbox_inches="tight")
            plt.close(fig)

            # fit elliposid
            ellipse_points, ellipse_coeff = lrgeo.fit_ellipse_to_point_cloud(aligned_umap_emb, 10000, 300)

            # assign to each point a pos and direction value according to closest k-neighbors
            k = 300
            ellipse_features = np.zeros((ellipse_points.shape[0], 2))
            for i in range(ellipse_points.shape[0]):
                point = ellipse_points[i, :2]
                d = ((aligned_umap_emb[:, :2] - point) ** 2).sum(axis=1)
                neigh_idx = d.argsort()[:k]
                ellipse_features[i, :] = [np.mean(pos[neigh_idx, 0]), mode(mov_dir[neigh_idx].T.tolist())]

            # compute in and out points
            angular_pos = np.arctan2(aligned_umap_emb[1:-1, 1], aligned_umap_emb[1:-1, 0]) * 180 / np.pi
            dist_to_ellipse = np.min(pairwise_distances(aligned_umap_emb, ellipse_points), axis=1)
            off_ring_points = dist_to_ellipse > np.percentile(dist_to_ellipse, 90)
            in_ring_points = np.invert(off_ring_points)

            # compute and decompose flows
            flow_diff = 2;
            flow = (aligned_umap_emb[flow_diff:] - aligned_umap_emb[:-flow_diff]) / 2
            flow_modulus = np.linalg.norm(flow, axis=1)
            flow_xy_angle = np.arctan2(flow[:, 1], flow[:, 0]) * 180 / np.pi
            flow_yz_angle = np.arctan2(flow[:, 2], np.linalg.norm(flow[:, :2], axis=1)) * 180 / np.pi
            # get radial and tangential
            tangential_flow = np.zeros(flow.shape)
            radial_flow = np.zeros(flow.shape)
            residual_flow = np.zeros(flow.shape)

            radial_modulus = np.zeros(flow.shape[0])
            tangential_modulus = np.zeros(flow.shape[0])
            radial_modulus_signed = np.zeros(flow.shape[0])
            closest_ellipse_idxs = np.argmin(pairwise_distances(aligned_umap_emb, ellipse_points), axis=1)
            distance_to_ellipse = np.min(pairwise_distances(aligned_umap_emb, ellipse_points), axis=1)
            for i in range(len(flow)):
                emb_point = aligned_umap_emb[i, :]
                closest_ellipse_point = ellipse_points[closest_ellipse_idxs[i], :]
                radial_direction = emb_point - closest_ellipse_point
                tangential_direction = lrgeo.find_perpendicular_vector_on_xy_plane(radial_direction)

                radial_flow[i, :] = lrgeo.project_onto_vector(flow[i, :], radial_direction)
                tangential_flow[i, :] = lrgeo.project_onto_vector(flow[i, :], tangential_direction)
                residual_flow[i, :] = flow[i, :] - radial_flow[i, :] - tangential_flow[i, :]

                radial_modulus[i] = np.linalg.norm(radial_flow[i])
                radial_sign = np.sign(np.dot(radial_flow[i], radial_direction))  # +1 if same direction, -1 if opposite
                radial_modulus_signed[i] = radial_sign * radial_modulus[i]
                tangential_modulus[i] = np.linalg.norm(tangential_flow[i])

            residual_modulus = np.linalg.norm(residual_flow, axis=1)

            speed = np.diff(pos)
            fig = plot_ring_flow(aligned_umap_emb, pos, mov_dir, trial_id, angular_pos, speed,
                                       ellipse_points, ellipse_features, in_ring_points, off_ring_points, flow,
                                       flow_modulus,
                                       flow_xy_angle, tangential_flow, tangential_modulus, radial_flow,
                                       radial_modulus_signed)

            fig.suptitle(f'{mouse} ring flow')
            figure_name = f"{mouse}_ring_flow_filter_{filter}"
            plt.savefig(os.path.join(msave_dir, figure_name + ".svg"), dpi=400, bbox_inches="tight")
            plt.savefig(os.path.join(msave_dir, figure_name + ".png"), dpi=400, bbox_inches="tight")
            plt.close(fig)

            new_dynamics_dict[session] = {
                        'signal': dynamic_dict[session]['signal'],
                        'pos': pos,
                        'angular_pos': angular_pos,
                        'speed': speed,
                        'trial_id': trial_id,
                        'mov_dir': mov_dir,
                        'umap_emb': umap_emb_real,
                        'umap_params': {
                            'n_neighbors': num_neigh,
                            'n_components': dim,
                            'min_dist': min_dist,
                            'random_state': 42
                        },
                        'centroids': centroids,
                        'centroids_pos': centroids_pos,
                        'umap_norm_vector': umap_norm_vector,
                        'cloud_center': cloud_center,
                        'horizontal_angle': horizontal_angle,
                        'horizontal_rotation_matrix': horizontal_rotation_matrix,

                        'horizontal_umap_emb': horizontal_umap_emb,
                        'horizontal_centroids': horizontal_umap_emb,

                        'centroids2D': centroids2D,
                        'centroids2D_pos': centroids2D_pos,
                        'pos_gradient_axis': pos_gradient_axis,

                        'pos_angle': pos_angle,
                        'pos_rotation_matrix': pos_rotation_matrix,
                        'aligned_umap_emb': aligned_umap_emb,
                        'aligned_centroids': aligned_centroids,


                        'ellipse_points': ellipse_points,
                        'ellipse_features': ellipse_features,
                        'ellipse_coeff': ellipse_coeff,

                        'dist_to_ellipse': dist_to_ellipse,
                        'closest_ellipse_idxs': closest_ellipse_idxs,
                        'in_ring_points': in_ring_points,
                        'off_ring_points': off_ring_points,

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
                        'residual_modulus':residual_modulus
                    }
        pickle_name = f"{mouse}_dynamic_dict_filter_{filter}.pkl"
        lrgu.save_pickle(msave_dir,pickle_name, new_dynamics_dict)


