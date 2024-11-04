# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

import os
import numpy as np
import torch
import torchsnooper
import scanpy as sc
import scvelo as scv
from scipy.sparse import issparse
from train import train
from parameters import NAME, EPOCHES, MODEL_DIR, LABL, cached_adata, model, fig_epoches, USE_HVG, use_M, GENE_LIST, SCALE
# from model import LSTM_VAE as Model
from utils import loada, prep_and_dlim, plot_v, invertt, scale_us, calculate_velocity
from scatter import plot_scatter
# model = Model()




# @torchsnooper.snoop()
def cal_param(datasets=None):
    global adata, model
    
    if use_M:
        try:
            u, s = adata.layers['Mu'], adata.layers['Ms']
        except KeyError:
            print(adata.layers.keys())
            datasets = 'raw'
    elif not use_M or datasets == 'raw':
        u, s = adata.layers['unspliced'], adata.layers['spliced']
    
    s = s.toarray() if issparse(s) else s
    u = u.toarray() if issparse(u) else u
    # u0_, s0_ = adata.var['scaled_u_max'].values, adata.var['scaled_s_max'].values

    if SCALE:
        if type(SCALE) == str:
            u = scale_us(u, np.ones(u.shape[1]), adata.var['scaled_u_max'].values/adata.var['scaled_s_max'].values)
            s = scale_us(s, np.ones(s.shape[1]), np.ones(s.shape[1]))
        else:
            u = scale_us(u, adata.var['u_scale'].values, adata.var['scaled_u_max'].values, final_scale=SCALE)
            s = scale_us(s, adata.var['s_scale'].values, adata.var['scaled_s_max'].values, final_scale=SCALE)
        # u0_, s0_ = get_max_us(u, s, switch_type='torch')
        # positiv = (u > 0) & (s > 0)
        # u0_, s0_ = (u * positiv).max(0), (s * positiv).max(0)
        u0_, s0_ = u.max(0), s.max(0)
    else:
        u0_, s0_ = adata.var['u_max'].values, adata.var['s_max'].values
    
    if USE_HVG == True:
        hvgs = adata.var['highly_variable'].values
        u, s = u[:, hvgs], s[:, hvgs]
        u0_, s0_ = u0_[hvgs], s0_[hvgs]

    u, s = torch.from_numpy(u).float(), torch.from_numpy(s).float()  # --- torch --- *
    # u, s = u[:30, :20], s[:30, :20]
    epoch = train(u, s, u0_, s0_)  #
    return epoch

    ''' model.load_state_dict(torch.load(MODEL_DIR + '/lstm%s.pkl' % EPOCHES-1))
    
    us = torch.cat([u[:,:,None], s[:,:,None]], dim=2)
    alpha, beta, gamma, _, __ = model(us, trn=False)

    velo = torch.zeros(adata.X.shape)
    if gene_sets == 'hvg':
        velo[:, hvgs] = beta * u - gamma * s
    else:
        velo = beta * u - gamma * s
    adata.layers['velocity'] = velo.detach().numpy()  # --- numpy --- * '''




if os.path.exists(cached_adata):
    adata = scv.read(cached_adata)
    print('Loaded cached data', cached_adata)
else:
    adata = loada(NAME)
    prep_and_dlim(adata, filt_val=[2500, 5])
    adata.write(cached_adata)

epoch = cal_param()
print('End at epoch', epoch)

# if epoch is not None:
#     fig_epoches = range(0, epoch)
'''
for epoch in fig_epoches:
    model.load_state_dict(torch.load(MODEL_DIR+'/'+LABL+'%s.pkl'%epoch))
    ki_params = calculate_velocity(adata=adata, model=model, rst_mode='both')
    for gi in GENE_LIST:
        print(gi)
        plot_scatter(adata=adata, gene_name=gi, ki_params=ki_params, labl=LABL, s_epoch=str(epoch))
    pv = plot_v(adata=adata, model=model, fig_name=LABL+str(epoch))  # need_inference=True, plt_mode='cell'
    print('Epoch', epoch, ':', pv)
'''
