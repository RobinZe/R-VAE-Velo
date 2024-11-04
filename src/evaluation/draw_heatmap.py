# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# string path or list of paths
DATA_PATH = '/public/home/jiangyz/velo/data/bm_lstm_vel.npy'
TITLE = 'lstm_vel'
CHOOSE_SIZE = (200, 400)  # lest that heatmap would be too massive
CHOOSE_NUM = None


def load_per_data(data_path):
    data_suffix = data_path.split('.')[-1]
    if data_suffix == 'npy':
        return np.load(data_path)
    elif data_suffix == 'csv' or data_suffix == 'txt':
        return np.loadtxt(data_path, delimiter=',')
    else:
        print('Invalid DATA PATH:', DATA_PATH)
        exit(1)


def load_data(data_path=DATA_PATH):
    ''' main function to call '''
    if type(data_path) == str:
        data_to_plot = load_per_data(data_path)
        for axs_i, shape_i in enumerate(data_to_plot.shape):
            if shape_i > CHOOSE_SIZE:
                index_min = int(np.random.choice(range(shape_i-CHOOSE_SIZE), 1))
                data_to_plot = sample_slice(data_to_plot, range(index_min, index_min+CHOOSE_SIZE), axs_i)
    elif len(data_path) == 1:
        data_to_plot = load_per_data(data_path[0])
        for axs_i, shape_i in enumerate(data_to_plot.shape):
            if shape_i > CHOOSE_SIZE:
                index_min = np.random.choice(range(shape_i-CHOOSE_SIZE), 1)
                data_to_plot = sample_slice(data_to_plot, range(index_min, index_min+CHOOSE_SIZE), axs_i)
    else:
        data_list = []
        for data_i in data_path:
            data_list.append(load_per_data(data_i))
        data_to_plot = merge_data(data_list)
    return data_to_plot


def sample_slice(np_data, slice_list, axis_num):
    assert np_data.shape[axis_num] >= max(slice_list)
    if axis_num == 0:
        return np_data[slice_list]
    elif axis_num == 1:
        return np_data[:, slice_list]
    elif axis_num == 2:
        return np_data[:, :, slice_list]
    else:
        print('Too large axis for plotting!')


def merge_data(lst):
    shp = lst[0].shape
    axs = np.argmin(shp)
    
    if shp[axs] * len(lst) > CHOOSE_SIZE:
        choose_fold_size = int(CHOOSE_SIZE / len(lst))
        index_min = np.random.choice(range(shp[axs]-choose_fold_size), 1)
        for lst_i in lst:
            shp[lst_i] = np_data(shp[lst_i], range(index_min, index_min+choose_fold_size), axs)
        print('Sampled on merged axis for being too large, please consider plot them seperately.')
    
    merged_data = np.concatenate(lst, axis=axs)
    for shp_i in shp:
        if shp_i == axs or shp[shp_i] <= CHOOSE_SIZE:
            continue
        else:
            index_min = np.random.choice(range(shp[shp_i] - CHOOSE_SIZE), 1)
            merged_data = sample_slice(merged_data, range(index_min, index_min+CHOOSE_SIZE), shp_i)
    return merged_data



def choose_data(adata, thres=CHOOSE_SIZE, clusters=None):
    choose_disp = 0
    n_genes = adata.var['dispersion'] > choose_disp
    max_disp = adata.var['dispersion'].max()
    per_change = max_disp / 100

    while n_genes.sum() < thres[0] or n_genes.sum() > thres[1]:
        if n_genes.sum() < thres[0]:
            per_change /= 10
            choose_disp -= per_change
        else:
            choose_dist += per_change
        n_genes = adata.var['dispersion'] > choose_dist
    
    velo = adata.layers['velocity'][:, n_genes]
    if clusters is not None:
        velo = velo[adata.obs['clusters'] == clusters]


def plot_heatmap(data, ttl='heatmap'):
    plt.figure()
    # sn.set()
    sn.heatmap(data, cmap=plt.get_cmap('Greens'))
    # v_min, v_max; c_map https://blog.csdn.net/ztf312/article/details/102474190
    # cmap=sns.light_palette("#2ecc71", as_cmap=True)
    plt.title(ttl)
    plt.savefig(ttl)


dd = load_data(DATA_PATH)
plot_heatmap(dd, ttl=TITLE)
