# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

# Large part of this script refer to 'VeloAE/veloproj/eval_utls.py'

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scvelo as scv
import torchsnooper

# K_CLUSTER = 'cluster'
# K_VELOCITY = 'velocity'
ADATA_PATH = '/home/cuiwei/velo/data/ForebrainGlut/dm_fb.h5ad'  # '/home/cuiwei/velo/data/DentateGyrus/scomplete.h5ad'
MODEL_NAME = 'fb6_29'

# dentate gyrus
'''C_NEIGHBORS = [['Astrocytes', 'Radial Glia-like'], ['Radial Glia-like', 'nIPC'], ['nIPC', 'Neuroblast'], ['Neuroblast', 'Mossy'], 
                ['Mossy', 'Granule immature'], ['Mossy', 'Cck-Tox'], ['Mossy', 'GABA'], ['Cck-Tox', 'GABA'], 
                ['Granule immature', 'Granule mature'], ['OPC', 'OL'], ['Microglia', 'Endothelial']]'''

# bone marrow
'''C_NEIGHBORS = [['Ery_2', 'Ery_1'], ['Ery_1', 'Mega'], ['HSC_1', 'HSC_2'], ['HSC_2', 'Precursors'], ['Precursors', 'Mono_2'], ['Precursors', 'Mono_1'],
                ['Mono_2','Mono_1'], ['Mono_2', 'DCs']] '''

# gastrulation e75

# forebrain glut
C_NEIGHBORS = [['0', '1'], ['1', '2'], ['2', '3'], ['3', '4'], ['4', '5'], ['5', '6']]



def fetch_neighbor(conn_matrix, obs_ind=None):
    ''' Fetch neighbors for each obs in obs_ind by identifying those not 0 '''
    if obs_ind is not None:
        conn_matrix = conn_matrix[obs_ind]
    conn_matrix = conn_matrix.toarray()
    return conn_matrix != 0  # no sparsity outside this function



# assistant functions

def summary_scores(all_scores):
    """Summarize group scores.
    
    Args:
        all_scores (dict{str,list}): {group name: score list of individual cells}.
    
    Returns:
        dict{str,float}: Group-wise aggregation scores.
        float: score aggregated on all samples
        
    """
    sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s }  # save all non-zero scores' mean as Dict
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])  # numpy mean overall
    return sep_scores, overal_agg



# Coherence between cells' velocity and its same-cluster neighbor
# @torchsnooper.snoop()
def inner_cluster_coh(adata, clusters=None, k_cluster='clusters', k_velocity='velocity', return_raw=False):
    """In-cluster Coherence Score.
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    if clusters is None:
        clusters = np.unique(adata.obs[k_cluster])
    mean_scores, all_scores = {}, {}
    for cat in clusters:
        sel = (adata.obs[k_cluster] == cat).values  # numpy bool (cell,)
        nbs = fetch_neighbor(adata.uns['neighbors']['connectivities'], sel)  # numpy bool (sel, cell)
        same_cat_nodes = sel[None, :] * nbs  # numpy bool (sel, cell)

        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[ith].reshape(1, -1), velocities[nodes]).mean() 
                     for ith, nodes in enumerate(same_cat_nodes) 
                     if nodes.sum() > 0]
        all_scores[cat] = cat_score
        mean_scores[cat] = np.mean(cat_score)
    
    if return_raw:
        return all_scores
    
    return mean_scores, np.mean([sc for sc in mean_scores.values()])


# Coherence between neighbor cells' velocity each from different clusters.
# @torchsnooper.snoop()
def cross_boundary_coh(adata, clusters, k_cluster='clusters', k_velocity='velocity', return_raw=False):
    """Cross-Boundary Velocity Coherence Score (A->B).
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        clusters (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by clusters
        or
        dict: mean scores indexed by clusters
        float: averaged score over all cells.
        
    """
    all_scores, mean_scores = {}, {}

    for u, v in clusters:
        sel = (adata.obs[k_cluster] == u).values
        sel2 = (adata.obs[k_cluster] == v).values
        nbs = fetch_neighbor(adata.uns['neighbors']['connectivities'], sel)  # numpy bool (sel, cell)
        boundary_nodes = sel2[None, :] * nbs
        
        velocities = adata.layers[k_velocity]
        v_us = velocities[sel]
        type_score = [cosine_similarity(v_us[ith].reshape(1, -1), velocities[nodes]).mean()
                      for ith, nodes in enumerate(boundary_nodes) 
                      if nodes.sum() > 0]
        mean_scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    
    if return_raw:
        return all_scores
    
    return mean_scores, np.mean([sc for sc in mean_scores.values()])


# Coherence between cells' velocity and distance change to its other-cluster neighbor on UMAP
# Original UMAP version temporarily no use due to no velocity UMAP. Here is the PCA version
# @torchsnooper.snoop()
def cross_boundary_correctness(adata, clusters, k_cluster='clusters', k_velocity='velocity', return_raw=False):  # x_emb="X_umap"
    """Cross-Boundary Direction Correctness Score (A->B)
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        clusters (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        x_emb (str): key to x embedding for visualization.
        
    Returns:
        dict: all_scores indexed by clusters
        or
        dict: mean scores indexed by clusters
        float: averaged score over all cells.
        
    """
    mean_scores, all_scores = {}, {}
    
    # if x_emb == "X_umap":
    #     v_emb = adata.obsm['{}_umap'.format(k_velocity)]
    # else:
    #     v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    x_emb = adata.obsm['X_pca']
    v_emb = np.matmul(adata.layers[k_velocity][:, adata.var['highly_variable']], adata.varm['PCs'][adata.var['highly_variable']])
        
    for u, v in clusters:
        sel = (adata.obs[k_cluster] == u).values  # numpy bool (cell,)
        sel2 = (adata.obs[k_cluster] == v).values
        nbs = fetch_neighbor(adata.uns['neighbors']['connectivities'], sel)  # numpy bool (sel, cell)
        boundary_nodes = sel2[None, :] * nbs  # numpy bool (sel, cell); sel with their v-cluster neighbor
        
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]  # (sel, nPC)
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if nodes.sum() == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            type_score.append(np.mean(dir_scores))
        
        mean_scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
        
    if return_raw:
        return all_scores 

    return mean_scores, np.mean([sc for sc in mean_scores.values()])



def eval_scores(adata):
    cluster_neighbors = []
    for ni in C_NEIGHBORS:
        # for nii, nij in ni:
        cluster_neighbors.append([ni[0], ni[1]])
        cluster_neighbors.append([ni[1], ni[0]])
    
    in_v = inner_cluster_coh(adata)
    ex_v = cross_boundary_coh(adata, cluster_neighbors)
    ex_x = cross_boundary_correctness(adata, cluster_neighbors)
    # print('Inner Cluster Coherence:', in_v[0])
    # print("\twith mean:", in_v[1])
    # print('Cross Cluster Coherence:', ex_v[0])
    # print("\twith mean:", ex_v[1])
    # print('Cross Cluster Correctness:', ex_x[0])
    # print("\twith mean:", ex_x[1])
    def transform_dict(d):
        dd = dict()
        for ki in d.keys():
            dd[ki] = [d[ki]]
        return dd
    pd.DataFrame(transform_dict(in_v[0])).to_csv('inner_coh.csv')
    pd.DataFrame(transform_dict(ex_v[0])).to_csv('cross_coh.csv')
    pd.DataFrame(transform_dict(ex_x[0])).to_csv('cross_cor.csv')


adata = scv.read(ADATA_PATH)
# if 'velocity' not in adata.layers.keys()
from utils import calculate_velocity
calculate_velocity(adata=adata, model_name=MODEL_NAME)

adata = adata[:, adata.var['highly_variable']]
adata.layers['velocity'][:, adata.var['highly_variable']][np.isnan(adata.layers['velocity'][:, adata.var['highly_variable']])] = 0
eval_scores(adata)

'''
def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type
    
    Args:
        adata (Anndata): Anndata object.
        nodes (list): Indexes for cells
        target (str): Cluster name.
        k_cluster (str): Cluster key in adata.obs dataframe

    Returns:
        list: Selected cells.

    """
    return nodes[adata.obs[k_cluster][nodes].values == target]
'''

# Mean of velocity confidence based on all cells in each cluster
# Non-sense function due to lack of velocity_confidence in our algorithm. Is confidence necessary？
'''
def in_cluster_scvelo_coh(adata, k_cluster, k_confidence, return_raw=False):
    """In-Cluster Confidence Score.
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_confidence (str): key to the column of cell velocity confidence in adata.obs.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
    
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}
    
    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        type_score = adata.obs[k_confidence][sel].values.tolist()
        scores[cat] = np.mean(type_score)
        all_scores[cat] = type_score
        
    if return_raw:
        return all_scores
    
    return scores, np.mean([s for _, s in scores.items()])
'''

# Mean of transition probability to its other-cluster neighbor
# No use due to lack of trans_g
'''
def cross_boundary_scvelo_probs(adata, k_cluster, clusters, k_trans_g, return_raw=False):
    """Compute Cross-Boundary Confidence Score (A->B).
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        clusters (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        k_trans_g (str): key to the transition graph computed using velocity.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by clusters
        or
        dict: mean scores indexed by clusters
        float: averaged score over all cells.
        
    """
    
    scores = {}
    all_scores = {}
    for u, v in clusters:
        sel = (adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        type_score = [trans_probs.toarray()[:, nodes].mean() 
                      for trans_probs, nodes in zip(adata.uns[k_trans_g][sel], boundary_nodes) 
                      if len(nodes) > 0]
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
    if return_raw:
        return all_scores    
    return scores, np.mean([sc for sc in scores.values()])
'''

