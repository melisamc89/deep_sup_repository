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
import MARBLE

base_dir =  '/home/melma31/Documents/deepsup_project/'
#__________________________________________________________________________
#|                                                                        |#
#|                               DYNAMICS                                 |#
#|________________________________________________________________________|#

data_dir = os.path.join(base_dir, 'data')
save_dir = os.path.join(base_dir, 'results')
if not os.path.isdir(save_dir): os.makedirs(save_dir)

mice_list= ['CalbZurich9_ref','CarlbZurich9_alt']
num_neigh = 120
dim = 3
min_dist = 0.1

for mouse in mice_list:
    dynamics_dict = {}
    msave_dir = os.path.join(save_dir, 'alternation_task') #mouse save dir
    if not os.path.isdir(msave_dir): os.makedirs(msave_dir)
    file_name =  mouse+'.pkl'
    mouse_dict = lrgu.load_pickle(data_dir,file_name)

    params = {'columns_to_rename': {'Fs': 'sf', 'pos': 'position', 'vel': 'speed'}}
    session_pd = copy.deepcopy(mouse_dict)
    for old, new in params['columns_to_rename'].items():
        if old in session_pd.columns: session_pd.rename(columns={old: new}, inplace=True)

    #session_pd = lrgu.add_mov_direction_mat_field(session_pd)
    #session_pd = lrgu.add_inner_trial_time_field(session_pd)
    #session_pd = lrgu.add_trial_id_mat_field(session_pd)

    signal = lrgu.get_signal(session_pd, 'raw_traces')
    clean_traces = lrgu.preprocess_traces(signal, sig_filt=6, sig_up=4, sig_down=12, peak_th=0.05)
    signal = clean_traces.copy()
    pos = lrgu.get_signal(session_pd, 'position')
    speed = lrgu.get_signal(session_pd, 'speed')

    pca = PCA(n_components=10)
    rates_pca = pca.fit_transform(signal)
    vel_rates_pca = np.diff(rates_pca, axis=0)
    rates_pca = rates_pca[:-1, :]  # skip last
    data = MARBLE.construct_dataset(
        anchor=rates_pca,
        vector=vel_rates_pca,
        k=15,
        delta=1.5,
    )
    params = {
        "epochs": 25,  # optimisation epochs
        "order": 2,  # order of derivatives
        "hidden_channels": 100,  # number of internal dimensions in MLP
        "out_channels": 3,
        "inner_product_features": False,
        "diffusion": True,
    }

    model = MARBLE.net(data, params=params)
    mdata_dir = data_dir
    out_file_name = f"{mouse}_dict_MARBLE.pkl"

    model.fit(data, outdir=mdata_dir + '/' + out_file_name)
    data = model.transform(data)

    marble_emb = data.emb

    # compute umap
    umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist, random_state=42)
    umap_model.fit(signal)
    umap_emb = umap_model.transform(signal)

    # mean center umap to facilitate future steps
    umap_emb -= umap_emb.mean(axis=0)

    pos = pos[:,0]
    row = 2
    col = 2
    fig = plt.figure(figsize=(6, 9))
    ax = fig.add_subplot(row, col, 1, projection='3d')
    b = ax.scatter(*umap_emb[:, :3].T, c=pos, s=1, alpha=0.5)
    ax.set_title('UMAP Position')

    ax = fig.add_subplot(row, col, 2, projection='3d')
    b = ax.scatter(
        marble_emb[:, 0],  # x-coordinates
        marble_emb[:, 1],  # y-coordinates
        marble_emb[:, 2],  # z-coordinates
        s=1, alpha=0.5,
        c=pos[:-1]
    )
    ax.set_title('MARBLE Position')

    ax = fig.add_subplot(row, col, 3, projection='3d')
    b = ax.scatter(*umap_emb[:, :3].T, c=speed, s=1, alpha=0.5)
    ax.set_title('UMAP Speed')

    ax = fig.add_subplot(row, col, 4, projection='3d')
    b = ax.scatter(
        marble_emb[:, 0],  # x-coordinates
        marble_emb[:, 1],  # y-coordinates
        marble_emb[:, 2],  # z-coordinates
        s=1, alpha=0.5,
        c=speed[:-1]
    )
    ax.set_title('MARBLE Speed')

    fig.suptitle(f'{mouse}')
    figure_name = f"{mouse}_UMAP_vs_MARBLE"
    msave_dir = os.path.join(base_dir, 'figures')  # mouse save dir
    plt.savefig(os.path.join(msave_dir, figure_name + "_delta1.5_emb_aware_25epocs.svg"), dpi=400, bbox_inches="tight")
    plt.savefig(os.path.join(msave_dir, figure_name + "_delta1.5_emb_aware_25epocs.png"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)
    emb_aware= marble_emb
    params = {
        "epochs": 25,  # optimisation epochs
        "order": 2,  # order of derivatives
        "hidden_channels": 100,  # number of internal dimensions in MLP
        "out_channels": 3,
        "inner_product_features": True,
        "diffusion": True,
    }

    model = MARBLE.net(data, params=params)
    mdata_dir = data_dir
    out_file_name = f"{mouse}_dict_MARBLE.pkl"

    model.fit(data, outdir=mdata_dir + '/' + out_file_name)
    data = model.transform(data)

    marble_emb = data.emb
    # compute umap
    # umap_model.fit(marble_emb)
    # marble_emb = umap_model.transform(marble_emb)

    # mean center umap to facilitate future steps
    umap_emb -= umap_emb.mean(axis=0)

    emb_agnostic = marble_emb
    row = 2
    col = 2
    fig = plt.figure(figsize=(6, 9))
    ax = fig.add_subplot(row, col, 1, projection='3d')
    b = ax.scatter(*umap_emb[:, :3].T, c=pos, s=1, alpha=0.5)
    ax.set_title('UMAP Position')

    ax = fig.add_subplot(row, col, 2, projection='3d')
    b = ax.scatter(
        marble_emb[:, 0],  # x-coordinates
        marble_emb[:, 1],  # y-coordinates
        marble_emb[:, 2],  # z-coordinates
        s=1, alpha=0.5,
        c=pos[:-1]
    )
    ax.set_title('MARBLE Position')

    ax = fig.add_subplot(row, col, 3, projection='3d')
    b = ax.scatter(*umap_emb[:, :3].T, c=speed, s=1, alpha=0.5)
    ax.set_title('UMAP Speed')

    ax = fig.add_subplot(row, col, 4, projection='3d')
    b = ax.scatter(
        marble_emb[:, 0],  # x-coordinates
        marble_emb[:, 1],  # y-coordinates
        marble_emb[:, 2],  # z-coordinates
        s=1, alpha=0.5,
        c=speed[:-1]
    )
    ax.set_title('MARBLE Speed')

    fig.suptitle(f'{mouse}')
    figure_name = f"{mouse}_UMAP_vs_MARBLE"
    msave_dir = os.path.join(base_dir, 'figures')  # mouse save dir
    plt.savefig(os.path.join(msave_dir, figure_name + "_delta1.5_emb_agn_25epocs.svg"), dpi=400, bbox_inches="tight")
    plt.savefig(os.path.join(msave_dir, figure_name + "_delta1.5_emb_agn_25epocs.png"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)
    pos = lrgu.get_signal(session_pd, 'position')
    dynamics_dict = {
        'signal': signal,
        'pos': pos,
        'speed': speed,
        'umap_emb': umap_emb,
        'marble_aware': np.array(emb_aware),
        'marble_agnostic' : np.array(emb_agnostic)
    }
    pickle_name = f"{mouse}_dynamic_dict.pkl"
    lrgu.save_pickle(msave_dir, pickle_name, dynamics_dict)


