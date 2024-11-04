# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

import warnings
import numpy as np
import torchsnooper
import scvelo as scv
import matplotlib.pyplot as plt
from parameters import use_M, SCALE, cached_adata
from utils import invertt, clipped_log, calculate_velocity, scale_us
# import torchsnooper


# functions in NumPy version

def clipped_log(x, lb: float = 0, ub: float = 1, eps: float = 1e-6):
    return np.log(np.clip(x, lb + eps, ub - eps))


def get_solution(alpha, beta, gamma, u0=0, s0=0, t=0):
    expu = np.exp(-beta * t)
    exps = np.exp(-gamma * t)
    unspliced = u0 * expu + alpha / beta * (1-expu)
    c = (alpha - u0 * beta) * invertt(gamma - beta)
    spliced = (s0 * exps + alpha / gamma * (1-exps) + c * (exps - expu))
    return unspliced, spliced


# @torchsnooper.snoop()
def tau_inv(u, s=None, u0=None, s0=None, alpha=None, beta=None, gamma=None):  # infer back to t with continue method
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inv_u = (gamma >= beta) if gamma is not None else True
        inv_us = ~inv_u
    any_invu = np.any(inv_u) or s is None
    any_invus = np.any(inv_us) and s is not None

    if any_invus:  # tau_inv(u, s)
        beta_ = beta * invertt(gamma - beta)
        xinf = alpha / gamma - beta_ * (alpha / beta)
        tau = (
            -1 / gamma * clipped_log((s - beta_ * u - xinf) / (s0 - beta_ * u0 - xinf))
        )

    if any_invu:  # tau_inv(u)
        uinf = alpha / beta
        tau_u = -1 / beta * clipped_log((u - uinf) / (u0 - uinf))
        tau = tau_u * inv_u + tau * inv_us if any_invus else tau_u  ### ???
    return tau



# @torchsnooper.snoop()
def plot_scatter(gene_name, ki_params, adata=None, labl=None, s_epoch=None):
    if adata is None:
        adata = scv.read(cached_adata)
    idx = ['Ms', 'Mu'] if use_M else ['spliced', 'unspliced']
    x_s = adata.layers[idx[0]][:, adata.var_names == gene_name]
    y_u = adata.layers[idx[1]][:, adata.var_names == gene_name]

    if SCALE:
        u0_, s0_ = adata.var['scaled_u_max'][adata.var_names == gene_name].values, adata.var['scaled_s_max'][adata.var_names == gene_name].values
        if type(SCALE) == str:
            y_u = scale_us(y_u, np.ones(y_u.shape[1]), u0_ / s0_)
            x_s = scale_us(x_s, np.ones(x_s.shape[1]), np.ones(x_s.shape[1]))
        else:
            # get_max_us(x_s[:, None], y_u[:, None])
            x_s = scale_us(x_s, adata.var['s_scale'][adata.var_names == gene_name].values, s0_, final_scale=SCALE)
            y_u = scale_us(y_u, adata.var['u_scale'][adata.var_names == gene_name].values, u0_, final_scale=SCALE)

        # positiv = ((x_s > 0) & (y_u > 0)).squeeze()
        x_s_max = x_s.max()
        y_u_max = y_u.max()
    else:
        x_s_max = adata.var['s_max'].values[adata.var_names == gene_name]
        y_u_max = adata.var['u_max'].values[adata.var_names == gene_name]
    
    ki_params = np.squeeze(ki_params)
    if len(ki_params.shape) != 1:
        # gene_idx = adata.var_names[adata.var['highly_variable']] == gene_name
        # kip = ki_params[:, gene_idx]
        kip = ki_params[:, adata.var_names[adata.var['highly_variable']] == gene_name]
    
    fig, ax = plt.subplots()
    ax.scatter(x_s, y_u, c=adata.uns['clusters_colors'][adata.obs['clusters'].astype('category').cat.codes.to_numpy()], label='expression value', alpha=0.5)
    
    tu = tau_inv(y_u_max, x_s_max, u0=0, s0=0, alpha=kip[0], beta=kip[1], gamma=kip[2]).squeeze()  # total time of up phase .values
    tpoints = np.linspace(0, tu, num=np.clip(x_s.shape[0], 50, 2000))
    solution_tpoints = get_solution(kip[0], kip[1], kip[2], t=tpoints)
    ax.plot(solution_tpoints[1], solution_tpoints[0], color='y', ls='-', label='up stream')

    td = tau_inv(np.where(x_s > 0, y_u, np.max(y_u)).min(), 0, u0=y_u_max, s0=x_s_max, alpha=0, beta=kip[1], gamma=kip[2]).squeeze()  # total time of down phase
    print('Times for up/down phase : ', tu, td)
    tpoints = np.linspace(0, td, num=np.clip(x_s.shape[0], 50, 2000))
    solution_tpoints = get_solution(0, kip[1], kip[2], u0=y_u_max, s0=x_s_max, t=tpoints)
    ax.plot(solution_tpoints[1], solution_tpoints[0], color='g', ls='-', label='down stream')
    
    ax.legend('Phase Scatter for' + gene_name)
    plt.savefig('figures/' + labl + gene_name + '_' + s_epoch + '_scatter.png')




