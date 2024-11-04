# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchsnooper
# from model import LSTM_VAE as Model
from parameters import EPOCHES, MODEL_DIR, LR, LABL, model, MODEL_INIT, EPSILON, LOSS_LAMBDA, US_LAMBDA, N_batch, SCALE, GENE_LIST, SLOPE, DEVICE
from utils import invertt, clipped_log, calculate_velocity, plot_v
from scatter import plot_scatter


### --- model --- ###

# Input : Expression Matrix * 2 (u/s)
# Latent : kinetic parameters for each Gene, state and latent time for each Cell
# Output : expression phase portrait recovered by latent variables (discrete sampling and continue integration)

# Loss evaluation : difference between real and recovered expression portrait
# Optimizer : ?



### 1. Latent Paramters Inference

# Parameters need to inference by NN model : alpha, beta, gamma, t_, scaling

# model = Model()
if MODEL_INIT is not None:
    model.load_state_dict(torch.load(MODEL_DIR+'/'+MODEL_INIT+'.pkl'))

if DEVICE:
    model.to(DEVICE)


### 2. Phase Portrait Inference : functions copied from scvelo.tools.dynamics_model_utils.py

# @torchsnooper.snoop()
def get_solution(alpha, beta, gamma, u0=0, s0=0, t=0):  # 根据时间反推u&s
    expu = torch.exp(-beta * t)
    exps = torch.exp(-gamma * t)
    # expu = torch.clamp(torch.exp(-beta * t), max=1e4)
    # exps = torch.clamp(torch.exp(-gamma * t), max=1e4)

    unspliced = u0 * expu + alpha / beta * (1 - expu)
    c = (alpha - u0 * beta) * invertt(gamma - beta)
    spliced = (s0 * exps + alpha / gamma * (1 - exps) + c * (exps - expu))

    nan = torch.isnan(unspliced) | torch.isnan(spliced)
    if torch.any(nan):
        reduced_nan = torch.any(nan, 0)
        print('alpha=%s, beta=%s, gamma=%s, expu=%s, exps=%s, unspliced=%s, c=%s, spliced=%s' 
                % (alpha[reduced_nan], beta[reduced_nan], gamma[reduced_nan], expu[:, reduced_nan], exps[:, reduced_nan], 
                    unspliced[:, reduced_nan], c[reduced_nan], spliced[:, reduced_nan])) 
        raise ValueError('NaN emerging!')
    return unspliced, spliced


# @torchsnooper.snoop()
def tau_inv(u, s=None, u0=None, s0=None, alpha=None, beta=None, gamma=None):  # infer back to t with continue method
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inv_u = (gamma >= beta) if gamma is not None else torch.tensor(True)
        inv_us = ~inv_u
    any_invu = torch.any(inv_u) or s is None
    any_invus = torch.any(inv_us) and s is not None

    if any_invus:  # tau_inv(u, s)
        beta_ = beta * invertt(gamma - beta)
        xinf = alpha / gamma - beta_ * (alpha / beta)
        tau = (
            -1 / gamma * clipped_log((s - beta_ * u - xinf) / (s0 - beta_ * u0 - xinf + EPSILON))
        )

    if any_invu:  # tau_inv(u)
        uinf = alpha / beta
        tau_u = (u - uinf) / (u0 - uinf + EPSILON)
        tau_u = -1 / beta * clipped_log(tau_u)  # ((u - uinf) * invertt(u0 - uinf))
        tau = tau_u * inv_u + tau * inv_us if any_invus else tau_u  ### ???
    return tau



### 3. Loss Calculation

def loss_min(loss1, loss2):
    loss12 = torch.cat([loss1[None, :], loss2[None, :]], dim=0)
    per_min_loss = torch.min(loss12, dim=0)[0]
    return per_min_loss


def up_rng(u, s, um, sm):
    ss = u * sm[None, :] * invertt(s * um[None, :])
    ss_dev = ss.device
    ss = ss.cpu()
    try:
        ss = ss.detach().numpy()
    except ValueError or RuntimeError:
        ss = ss.numpy()
    up_dn = (ss >= 1.).astype(bool)  # & (ss < SLOPE).astype(bool)
    return torch.from_numpy(up_dn).to(ss_dev)


def dn_rng(u, s, um, sm):
    ss = u * sm[None, :] * invertt(s * um[None, :])
    ss_dev = ss.device
    ss = ss.cpu()
    try:
        ss = ss.detach().numpy()
    except ValueError or RuntimeError:
        ss = ss.numpy()
    up_dn = (ss < 1.01).astype(bool)  # & (ss > 1/SLOPE).astype(bool)
    return torch.from_numpy(up_dn).to(ss_dev)



# @torchsnooper.snoop()
def project_loss(
    u, s, alpha, beta, gamma, u0_=None, s0_=None, t_=None, scaling=1
):
    x_obs = torch.cat([u[None, :], s[None, :]], dim=0)  # .transpose(1, 0, 2)  # (2, C, G) -> (C, 2, G)
    t0 = tau_inv(torch.min(torch.where(s > 0, u, torch.max(u)), dim=0)[0], 0,
                    u0=u0_, s0=s0_, alpha=0, beta=beta, gamma=gamma)  # total time of down phase
    if t_ == None:
        t_ = tau_inv(u0_, s0_, u0=0, s0=0, alpha=alpha, beta=beta, gamma=gamma)
    tt = torch.max(torch.cat([t_[None, :], t0[None, :]], dim=0), dim=0)[0].cpu().detach().numpy()

    with torch.no_grad():
        num = np.clip(int(len(u) / 5), 200, 800)
        tpoints = torch.from_numpy(np.linspace(0, tt, num=num)).to(DEVICE)
        # tpoints = torch.from_numpy(np.linspace(0, t_.detach(), num=num))
        # tpoints_ = torch.from_numpy(np.linspace(0, t0.detach(), num=num)[1:])
    xt = get_solution(alpha=alpha, beta=beta, gamma=gamma, t=tpoints)
    xt_ = get_solution(alpha=0, beta=beta, gamma=gamma, u0=u0_, s0=s0_, t=tpoints)
    xt, xt_ = torch.cat([xt[0][None,:], xt[1][None,:]], dim=0), torch.cat([xt_[0][None,:], xt_[1][None,:]], dim=0)

    # assign time points (oth. projection onto 'on' and 'off' curve)
    loss = torch.min(
        ((xt[:, None, :] - x_obs[:, :, None]) ** 2).sum(dim=0), dim=1  # (C, G)
    )[0] * up_rng(u, s, u0_, s0_)
    loss_ = torch.min(
        ((xt_[:, None, :] - x_obs[:, :, None]) ** 2).sum(dim=0), dim=1
    )[0] * dn_rng(u, s, u0_, s0_)

    # loss_us = loss_min(loss, loss_)
    # loss_us = torch.sum(loss_us[rng(u, s, u0_, s0_)])
    loss_us = torch.sum(torch.mean(loss, dim=0)) + torch.sum(torch.mean(loss_, dim=0))
    print('project_loss : ', loss_us.item())
    return loss_us


def assign_loss(
    u, s, alpha, beta, gamma, t_=None, u0_=None, s0_=None, scaling=1
):
    with torch.no_grad():
        tau = tau_inv(u, s, 0, 0, alpha, beta, gamma)
        tau = torch.clamp(tau, 0, t_)

        tau_ = tau_inv(u, s, u0_, s0_, 0, beta, gamma)
        max_tau_ = torch.max(torch.where(s > 0, tau_, 0), dim=0)[0]
        tau_ = torch.clamp(tau_, 0, max_tau_)
    
    u1, s1 = get_solution(alpha, beta, gamma, u0=0, s0=0, t=tau)
    u2, s2 = get_solution(0, beta, gamma, u0=u0_, s0=s0_, t=tau_)

    loss1 = scaling * (u - u1) ** 2 + (s - s1) ** 2
    loss2 = scaling * (u - u2) ** 2 + (s - s2) ** 2
    loss_us = loss_min(loss1, loss2)

    return loss_us


def l1_regul(model):
    ''' Manually assign L1 regularization. '''
    l_loss = 0
    for module in model.module:
        if type(module) in ():
            l_loss += torch.abs(module.weight).sum()
    return l_loss





### execution
# @torchsnooper.snoop()
def execution(u, s, epoch, u0_=0, s0_=0):  # 
    global model, LR, LOSS_LAMBDA, US_LAMBDA
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)

    us = torch.cat([u[:,:,None], s[:,:,None]], dim=2)
    # mu0_max = torch.cat([torch.zeros(3, s0_.shape[0]), u0_[None, :], s0_[None, :]], dim=0)
    mdl_rst, c_loss = model(us)  # (C, G, 2) -> (G,) , mu0=mu0_max
    (alpha, beta, gamma, u_pred, s_pred) = mdl_rst

    reconst_loss = project_loss(u, s, alpha, beta, gamma, u0_=u_pred, s0_=s_pred)
    loss = reconst_loss + c_loss * LOSS_LAMBDA + US_LAMBDA * (F.mse_loss(u_pred, u0_) + F.mse_loss(s_pred, s0_))

    optimizer.zero_grad()
    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
    optimizer.step()
    torch.save(model.state_dict(), MODEL_DIR+'/'+LABL+'%s.pkl'%epoch)

'''
def execution(u, s, epoch):
    global model
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)

    us = torch.cat([u[:,:,None], s[:,:,None]], dim=2)
    mdl_rst, c_loss = model(us) # (C, G, 2) -> (G,)
    try:
        (alpha, beta, gamma, t_, scaling) = mdl_rst
    except ValueError:
        alpha, beta, gamma, t_, scaling = mdl_rst[0], mdl_rst[1], mdl_rst[2], mdl_rst[3], mdl_rst[4]

    # with torch.no_grad():
    u0_, s0_ = get_solution(alpha, beta, gamma, u0=0, s0=0, t=t_)

    reconst_loss = project_loss(u, s, alpha, beta, gamma, t_, u0_, s0_)  # .detach()
    loss = reconst_loss + c_loss * LOSS_LAMBDA

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
    optimizer.step()
    # return alpha, beta, gamma
    torch.save(model.state_dict(), MODEL_DIR+'/'+LABL+'%s.pkl'%epoch) '''


# @torchsnooper.snoop()
def train(u, s, u0_, s0_):  #
    global LOSS_LAMBDA, LR, US_LAMBDA
    u0_, s0_ = torch.from_numpy(u0_).float(), torch.from_numpy(s0_).float()
    if DEVICE:
        u0_, s0_ = u0_.to(DEVICE), s0_.to(DEVICE)
        u, s = u.to(DEVICE), s.to(DEVICE)
    try:
        for epoch in EPOCHES:
            print("\tEpoch", epoch)
            for bi in range(N_batch):
                execution(u[:, bi::N_batch], s[:, bi::N_batch], epoch, u0_[bi::N_batch], s0_[bi::N_batch])  #

            if epoch % 5 == 4 and epoch > 5:
                model.load_state_dict(torch.load(MODEL_DIR+'/'+LABL+'%s.pkl'%epoch))
                ki_params = calculate_velocity(model=model, rst_mode='both')
                for gi in GENE_LIST:
                    plot_scatter(gene_name=gi, ki_params=ki_params, labl=LABL, s_epoch=str(epoch))  # adata
                pv = plot_v(model, fig_name=LABL+str(epoch))  # adata, need_inference=True, plt_mode='cell'
                print('Epoch', epoch, ':', pv)

                LOSS_LAMBDA *= 1.1
                LR *= .5
                US_LAMBDA *= 1.1
    except ValueError as e:
        print(e)
        return epoch


# u, s = torch.zeros(300, 1000), torch.ones(300, 1000)

# train(u, s)
