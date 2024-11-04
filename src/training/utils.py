# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

import warnings
import numpy as np
import torch
import torchsnooper
import scanpy as sc
import scvelo as scv

from parameters import MODEL_DIR, use_M, SCALE, S_PNG, cached_adata, USE_HVG, EPSILON, DEVICE
from parameters import model as eval_model




def invertt(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if type(x) == np.ndarray:
            x_inv = np.clip(1 / x, 0, 1e6) * (x != 0)
        else:
            x_dev = x.device
            x = x.cpu().detach().numpy()
            x_inv = np.clip(1 / x, 0, 1e6) * (x != 0)
            x_inv = torch.from_numpy(x_inv).to(x_dev)
        # except RuntimeError or TypeError:
    return x_inv


# @torchsnooper.snoop()
def clipped_log(x, lb: float = 0, ub: float = 1, eps: float = 1e-6):
    return torch.log(torch.clamp(x, lb + eps, ub - eps))




# functions for Scanpy data

def loada(NAME):
    if NAME == 'bonemarrow':
        adata = scv.datasets.bonemarrow()
    elif NAME == 'forebrain':
        adata = scv.datasets.forebrains()
    elif NAME == 'dentategyrus':
        adata = scv.datasets.dentategyrus()
    elif NAME == 'pancreas':
        adata = scv.datasets.pancreas()
    else:
        print('Uncommon dataset name :', NAME)
        exit(1)
    return adata


def prep_and_dlim(adata, filt_val=None):
    # qc
    adata.var_names_make_unique()
    if filt_val is not None:
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        '''
        sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
        '''
        adata = adata[adata.obs.n_genes_by_counts < filt_val[0], :]
        adata = adata[adata.obs.pct_counts_mt < filt_val[1], :]

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    else:
        scv.pp.filter_and_normalize(adata)
    # preprocessing
    scv.pp.moments(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)

    # UMAP dimension reduction
    if 'umap' not in adata.uns.keys():
        if 'pca' not in adata.uns.keys():
            sc.tl.pca(adata)
        if 'neighbors' not in adata.uns.keys():
            sc.pp.neighbors(adata)
        sc.tl.umap(adata)



# @torchsnooper.snoop()
def scale_us(x, scale, scaled_max, final_scale=None):
    x = x / scale[None, :]  # np.clip(s / scale[None, :], 0, 1e4)
    x = x * invertt(scaled_max)[None, :]  # np.clip(s * invertt(scaled_max)[None, :], 0, 1e4)
    if final_scale:
        x *= final_scale
    return x


def calculate_velocity(adata=None, model=eval_model, model_name=None, eval_mode='reparam', rst_mode=None):
    if adata is None:
        adata = scv.read(cached_adata)
    if model_name != None:
        model.load_state_dict(torch.load(MODEL_DIR + '/%s.pkl' % model_name))
    if DEVICE:
        model.to(DEVICE)

    u, s = (adata.layers['Mu'], adata.layers['Ms']) if use_M else (adata.layers['unspliced'], adata.layers['spliced'])
    # u, s = u / adata.var['u_scale'], s / adata.var['s_scale']
    if SCALE:
        if type(SCALE) == str:
            u = scale_us(u, np.ones(u.shape[1]), adata.var['scaled_u_max'].values/adata.var['scaled_s_max'].values)
            s = scale_us(s, np.ones(s.shape[1]), np.ones(s.shape[1]))
        else:
            u = scale_us(u, adata.var['u_scale'].values, adata.var['scaled_u_max'].values, final_scale=SCALE)
            s = scale_us(s, adata.var['s_scale'].values, adata.var['scaled_s_max'].values, final_scale=SCALE)            

    if USE_HVG:
        hvgs = adata.var['highly_variable']
        u, s = u[:, hvgs], s[:, hvgs]

    us = torch.from_numpy(np.concatenate([u[:,:,None], s[:,:,None]], axis=2)).float()
    us = us.to(DEVICE) if DEVICE else us
    kinetic_params = model(us, mode=eval_mode).to('cpu').detach().numpy()  # --- numpy ---  # avg or reparam ?
    if rst_mode == 'kinetic parameter':
        return kinetic_params

    if kinetic_params.shape[0] < 3:
        kinetic_params, kinetic_params_var = kinetic_params[0], kinetic_params[1]
    beta, gamma = kinetic_params[1], kinetic_params[2]
    velo = np.zeros(adata.X.shape)
    if velo.shape[1] == kinetic_params.shape[1]:
        velo = beta * u - gamma * s
    else:
        velo[:, hvgs] = beta * u - gamma * s
    adata.layers['velocity'] = velo
    print('velocity in range of :', velo.min(), velo.max())
    
    if rst_mode == 'both':
        return kinetic_params
    

def plot_v(model, fig_name, adata=None, plt_mode=None, need_inference=False, s_png=S_PNG):
    if adata is None:
        adata = scv.read(cached_adata)
    if need_inference or 'velocity' not in adata.layers.keys():
        calculate_velocity(adata=adata, model=model, model_name=fig_name)  # , eval_mode='avg'
    if s_png:
        fig_name = fig_name + '.png'
    
    try:
        scv.tl.velocity_graph(adata, gene_subset=adata.var_names[adata.var['highly_variable']])
    except ValueError as ve:
        return ve

    if plt_mode=='cell':
        scv.pl.velocity_embedding(adata, basis = 'umap', save=fig_name)  # small arrow per cell
    elif plt_mode=='grid':
        scv.pl.velocity_embedding_grid(adata, basis = 'umap', save=fig_name)
    else:
        scv.pl.velocity_embedding_stream(adata, basis = 'umap', save=fig_name, color=adata.obs['colour'])
    # scv.pl.velocity_graph(adata, save='')  # nodes and vectors
    return 'Figure %s plotted' % fig_name


def cache_data(adata, model, model_name, cache_mode='both', save_type='numpy', only_hvg=True):
    if 'velocity' in adata.layers.keys():
        kinetic_params = calculate_velocity(adata, model, model_name, rst_mode='kinetic parameter')
    else:
        kinetic_params = calculate_velocity(adata, model, model_name, rst_mode='both')
    
    if only_hvg:
        velo = adata.layers['velocity'][:, adata.vars['highly_variable']]
    if save_type == 'numpy':
        np.save('data/%s_velocity.npy', velo)
        np.save('data/%s_param.npy', kinetic_params)
    elif save_type == 'txt':
        np.savetxt('data/%s_velocity.csv', velo, delimiter=',')
        np.savetxt('data/%s_param.npy', kinetic_params, delimiter=',')


