# ~/miniconda3/envs/dlim
# -*- coding:utf-8 -*-

import pandas as pd
import scvelo as scv
from ..training.utils import calculate_velocity
from ..training.scatter import *
from ..training.parameters import cached_adata, LABL, MODEL_INIT, GENE_LIST



adata = scv.read(cached_adata)

ki_params = calculate_velocity(adata, model_name=MODEL_INIT, rst_mode='both')


if len(GENE_LIST) == 0:
    df_hvg = pd.DataFrame({'gene_name':adata.var_names, 'dispersion':adata.var['dispersions']}).sort_values(by='dispersion', ascending=False)
    GENE_LIST = df_hvg[:10]['gene_name'].values



for gi in GENE_LIST:
    plot_scatter(adata, gene_name=gi, ki_params=ki_params, labl=LABL)

