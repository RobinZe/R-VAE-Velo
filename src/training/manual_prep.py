# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import rcParams


DATA_PATH = 'data/BoneMarrow/complete.h5ad'

sc.settings.verbosity = 3
# sc.logging.print_versions()

adata = sc.read(DATA_PATH)
sc.pl.highest_expr_genes(adata, n_top=20, save='bm_counts.pdf')  # highest expression genes

adata.var_names_make_unique()

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save='_fb')

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', save='_fb_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', save='_fb_counts')

adata = adata[adata.obs.n_genes_by_counts < 3000, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]
# dentate gyrus 2500, 5
# bone marrow   3000, 20
# gastrulation  5000
# forebrain     3000, 5

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)  # min_mean=0.0125, max_mean=3, min_disp=0.5

'''
# plot hvgs
sc.pl.highly_variable_genes(adata, save='_fb_hvgs')
df_hvg = pd.DataFrame({'gene_name':adata.var_names, 'hvg':adata.var['highly_variable'], \
                        'dispersion':adata.var['dispersions']}).sort_values(by='dispersion', ascending=False)
df_hvg[:10]
df_hvg.to_csv('data/BoneMarrow/highly_variable.csv')

# calculate, plot and save marker genes
sc.tl.rank_genes_groups(adata, 'clusters')  # marker_genes, saved in .uns
sc.pl.rank_genes_groups(adata, n_genes=20, save='_bm')
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv('data/BoneMarrow/marker_genes.csv')

# plot UMAP with top 2 hvgs for X/V
sc.pl.umap(adata, color=['HBB', 'LYZ'], save='_bm_hvg')
sc.pl.umap(adata, color=['HBB', 'LYZ'], layer='velocity', save='_bm_velo_hvg')

# trajectory
sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3,3), facecolor='white')
adata.X = adata.X.astype('float64')

sc.tl.paga(adata, groups='clusters')
sc.pl.paga(adata, color=['clusters', 'HBB', 'LYZ'], save='_bm_hvg')
color=adata.obs['colour'], groups=['cluster1', 'cluster3']
'''


# get scale & max before

def nonzero_std(s):  # return standard deviation for non-zero part
    sz = s > 0
    ssz = sz.sum(0)
    ssz = np.where(ssz == 0, 1, ssz)
    ms = s.sum(0) / ssz
    sstd = np.sum(sz * (s - ms) ** 2, axis=0) / ssz
    return np.sqrt(sstd)


def get_scale(adata):
    u, s = (adata.layers['Mu'], adata.layers['Ms']) if use_M else (adata.layers['unspliced'], adata.layers['spliced'])
    u_scale = nonzero_std(u)
    s_scale = nonzero_std(s)
    u_scale[u_scale == 0] = 1
    s_scale[s_scale == 0] = 1
    adata.var['u_scale'] = u_scale
    adata.var['s_scale'] = s_scale



def get_max(u, s):
    data = np.array([u, s])
    # u, s = (adata.layers['Mu'], adata.layers['Ms']) if use_M else (adata.layers['unspliced'], adata.layers['spliced'])
    normalized_data = np.sum(data / data.max(axis=1, keepdims=True).clip(1e-3, None), axis=0)
    # slope = (u * greater).sum(0) / (s * greater).sum(0)

    cell_idx = np.argmax(normalized_data, axis=0)
    u_max = u[cell_idx, range(u.shape[1])]
    s_max = s[cell_idx, range(s.shape[1])]
    return u_max, s_max


def get_max(u, s, perc=99):
    data = np.array([u, s])
    # u, s = (adata.layers['Mu'], adata.layers['Ms']) if use_M else (adata.layers['unspliced'], adata.layers['spliced'])
    normalized_data = np.sum(data / data.max(axis=1, keepdims=True).clip(1e-3, None), axis=0)
    upper_bound = np.percentile(normalized_data, perc, axis=0)
    greater = (normalized_data >= upper_bound).astype(bool)

    u_max = (u * greater).sum(0)/greater.sum(0)
    s_max = (s * greater).sum(0)/greater.sum(0)
    return u_max, s_max


def get_proj(adata):
    x = adata.X.A
    #
    x_scale = nonzero_std(x)
    x_scale[x_scale < 1e-4] = 1
    scaled_x = x / x_scale[None, :]
    #
    data = np.array([adata.layers['Mu'], adata.layers['Ms']])
    normalized_data = np.sum(data / data.max(axis=1, keepdims=True).clip(1e-3, None), axis=0)
    cell_idx = np.argmax(normalized_data, axis=0)
    scaled_x_max = scaled_x[cell_idx, range(x.shape[1])]
    #
    x_proj = x + adata.layers['velocity'] * (x_scale * scaled_x_max)[None, :]
    adata.layers['proj'] = x_proj


''' 
# @torchsnooper.snoop()
def get_max(u, s, perc, decl_rate, mode):
    # if switch_type == 'torch':
    #     u, s = u.numpy(), s.numpy()
    positiv = (u > 0) & (s > 0)

    u_ = np.percentile(u * positiv, perc, axis=0)
    s_ = np.percentile(s * positiv, perc, axis=0)
    
    u_greater = u > u_
    s_greater = s > s_
    if mode == 'seperate_mean':
        u_max = np.sum(u * u_greater, axis=0) / np.sum(u_greater, axis=0)
        s_max = np.sum(s * s_greater, axis=0) / np.sum(s_greater, axis=0)
    else:
        both_greater = (u_greater) & (s_greater)
        # either_greater = ((u_greater) | (s_greater)) & positiv
        have_both_greater = np.any(both_greater, axis=0)

        max_prec = np.zeros_like(u)
        # max_prec = np.where(have_both_greater[None, :], both_greater, either_greater)
        while not np.all(have_both_greater):
            max_prec[:, have_both_greater] = both_greater[:, have_both_greater]
            max_prec[:, ~have_both_greater] = either_greater[:, ~have_both_greater]

        if mode == 'slope':
            slope = np.sum(u * max_prec, axis=0) / np.sum(s * max_prec, axis=0)
            # rint((slope == 0).sum())
            cell_idx = np.argmax(max_prec * (u * slope + s), axis=0)
            u_max = u[cell_idx, range(u.shape[1])]
            s_max = s[cell_idx, range(s.shape[1])]
        elif mode == 'mean':
            u_max = np.sum(u * max_prec, axis=0) / np.sum(max_prec, axis=0)
            s_max = np.sum(s * max_prec, axis=0) / np.sum(max_prec, axis=0)
    return u_max, s_max

    max_prec = np.zeros_like(u)

    while not np.all(np.any(max_prec, axis=0)) and perc > 80:
        u_ = np.percentile(u * positiv, perc, axis=0)
        s_ = np.percentile(s * positiv, perc, axis=0)
        both_greater = (u > u_) & (s > s_)
        have_both_greater = np.any(both_greater, axis=0)
        gene_idx = np.any(max_prec, axis=0)
        max_prec[:, ~gene_idx] = both_greater[:, ~gene_idx]
        perc *= decl_rate

    slope = np.zeros(s.shape[1])
    slope = u.max(0)/s.max(0)
    slope[np.any(max_prec, axis=0)] = np.sum(u*max_prec, axis=0)[np.any(max_prec, axis=0)]/np.sum(s*max_prec, axis=0)[np.any(max_prec, axis=0)]
    cell_idx = np.argmax(max_prec * (u * slope + s), axis=0)
    cell_idx[~np.any(max_prec, axis=0)] = np.argmax((s>0)*(u * slope + s), axis=0)[~np.any(max_prec, axis=0)]

    u_max = u[cell_idx, range(u.shape[1])]
    s_max = s[cell_idx, range(s.shape[1])]
    return u_max, s_max '''



def get_max_us(adata, perc=99, decl_rate=.99, mode='slope'):
    if 'scaled_u_max' in adata.var.keys():
        return adata.var['scaled_u_max'], adata.var['scaled_s_max']
    else:
        u, s = adata.layers['Mu'], adata.layers['Ms'] if use_M else adata.layers['unspliced'], adata.layers['spliced']
        u /= adata.var['u_scale']
        s /= adata.var['s_scale']
        return get_max(u, s, perc, decl_rate, mode)


