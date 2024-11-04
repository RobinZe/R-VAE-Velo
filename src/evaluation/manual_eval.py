# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

import os
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import rcParams


# prepare hvgs
sc.pl.highly_variable_genes(adata, save='_bm_hvgs')
df_hvg = pd.DataFrame({'gene_name':adata.var_names, 'hvg':adata.var['highly_variable'], \
                        'dispersion':adata.var['dispersions']}).sort_values(by='dispersion', ascending=False)
df_hvg[:10]
df_hvg.to_csv('data/BoneMarrow/highly_variable.csv')

# calculate, plot and save marker genes
sc.tl.rank_genes_groups(adata, 'clusters')  # marker_genes, saved in .uns
sc.pl.rank_genes_groups(adata, n_genes=20, save='_bm')
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv('data/BoneMarrow/marker_genes.csv')


# eval 1: trajectory
sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3,3), facecolor='white')
adata.X = adata.X.astype('float64')

sc.tl.paga(adata, groups='clusters')
sc.pl.paga(adata, color=['clusters', 'HBB', 'LYZ'], save='_bm_hvg')

# eval 2: UMAP
# plot UMAP with top 2 hvgs for X/V
sc.pl.umap(adata, color=['HBB', 'LYZ'], save='_bm_hvg')
sc.pl.umap(adata, color=['HBB', 'LYZ'], layer='velocity', save='_bm_velo_hvg')

# eval 3: scatter plot

# eval 4: heatmap

# eval 5: inner-/extra- coherence
