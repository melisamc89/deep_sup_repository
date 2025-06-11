t_start = 340
t_end = 820
sr = 20
time = np.arange(t_start,t_end)/sr
dy = 1
dx = 1/sr
x, y = np.mgrid[t_start/sr:t_end/sr:dx, 0:rates.shape[1]:dy]

figure, axes = plt.subplots(3,1, figsize = (6,12))
axes[0].plot(time,pos[t_start:t_end])
axes[1].plot(time,rates[t_start:t_end,3])
axes[2].pcolormesh(x,y,rates[t_start:t_end,:])

axes[0].set_title('GC2', fontsize = 15)
axes[2].set_xlabel('Time (s)', fontsize = 15)
axes[0].set_ylabel('Position', fontsize = 15)
axes[1].set_ylabel('Rates', fontsize = 15)
axes[2].set_ylabel('Neurons', fontsize = 15)
plt.show()
figure.savefig('/home/melma31/Documents/deepsup_project/figures/' + 'calcium_traces.png')



rates = dynamic_dict[session]['signal']
pca = PCA(n_components=10)
rates_pca = pca.fit_transform(rates)
vel_rates_pca = np.diff(rates_pca, axis=0)
rates_pca = rates_pca[:-1, :]  # skip last

data = MARBLE.construct_dataset(
    anchor=rates_pca,
    vector=vel_rates_pca,
    k=15,
    delta=1.4,
)

params = {
    "epochs": 20,  # optimisation epochs
    "order": 2,  # order of derivatives
    "hidden_channels": 100,  # number of internal dimensions in MLP
    "out_channels": 3,
    "inner_product_features": False,
    "diffusion": True,
}

model = MARBLE.net(data, params=params)
mdata_dir = data_dir
out_file_name = f"{mouse}_dynamic_dict_MARBLE.pkl"

model.fit(data, outdir=mdata_dir + '/' + out_file_name)
data = model.transform(data)

marble_emb = data.emb

# compute umap
umap_model = umap.UMAP(n_neighbors=num_neigh, n_components=dim, min_dist=min_dist, random_state=42)
umap_model.fit(dynamic_dict[session]['signal'])
umap_emb = umap_model.transform(dynamic_dict[session]['signal'])

# mean center umap to facilitate future steps
umap_emb -= umap_emb.mean(axis=0)

# compute umap
# umap_model.fit(marble_emb)
# marble_emb = umap_model.transform(marble_emb)

# mean center umap to facilitate future steps
umap_emb -= umap_emb.mean(axis=0)

mov_dir_color = lrgu.get_dir_color(dynamic_dict[session]['mov_dir'])

row = 4
col = 2
fig = plt.figure(figsize=(6, 9))
ax = fig.add_subplot(row, col, 1, projection='3d')
b = ax.scatter(*umap_emb[:, :3].T, color=mov_dir_color, s=10)
ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
ax.set_title('UMAP Direction')

ax = fig.add_subplot(row, col, 2, projection='3d')
b = ax.scatter(
    marble_emb[:, 0],  # x-coordinates
    marble_emb[:, 1],  # y-coordinates
    marble_emb[:, 2],  # z-coordinates
    s=10,  # marker size
    color=mov_dir_color[:-1]
)
ax.scatter([], [], color=lrgu.get_dir_color(np.array([0])), label='none')
ax.scatter([], [], color=lrgu.get_dir_color(np.array([-1])), label='left')
ax.scatter([], [], color=lrgu.get_dir_color(np.array([1])), label='right')
ax.set_title('MARBLE Direction')

pos = dynamic_dict[session]['pos'][:, 0]
ax = fig.add_subplot(row, col, 3, projection='3d')
b = ax.scatter(*umap_emb[:, :3].T, c=pos, s=10)
ax.set_title('UMAP Position')

ax = fig.add_subplot(row, col, 4, projection='3d')
b = ax.scatter(
    marble_emb[:, 0],  # x-coordinates
    marble_emb[:, 1],  # y-coordinates
    marble_emb[:, 2],  # z-coordinates
    s=10,  # marker size
    c=pos[:-1]
)
ax.set_title('MARBLE Position')

trial_id = dynamic_dict[session]['trial_id']
ax = fig.add_subplot(row, col, 5, projection='3d')
b = ax.scatter(*umap_emb[:, :3].T, c=trial_id, s=10)
ax.set_title('UMAP trial')

ax = fig.add_subplot(row, col, 6, projection='3d')
b = ax.scatter(
    marble_emb[:, 0],  # x-coordinates
    marble_emb[:, 1],  # y-coordinates
    marble_emb[:, 2],  # z-coordinates
    s=10,  # marker size
    c=trial_id[:-1]
)
ax.set_title('MARBLE trial')

speed = dynamic_dict[session]['speed']
ax = fig.add_subplot(row, col, 7, projection='3d')
b = ax.scatter(*umap_emb[:, :3].T, c=speed, s=10)
ax.set_title('UMAP trial')

ax = fig.add_subplot(row, col, 8, projection='3d')
b = ax.scatter(
    marble_emb[:, 0],  # x-coordinates
    marble_emb[:, 1],  # y-coordinates
    marble_emb[:, 2],  # z-coordinates
    s=10,  # marker size
    c=speed[:-1]
)
ax.set_title('MARBLE trial')

fig.suptitle(f'{mouse}')
figure_name = f"{mouse}_UMAP_vs_MARBLE"
msave_dir = os.path.join(base_dir, 'figures')  # mouse save dir
plt.savefig(os.path.join(msave_dir, figure_name + "_delta1.4.svg"), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join(msave_dir, figure_name + "_delta1.4.png"), dpi=400, bbox_inches="tight")
plt.close(fig)