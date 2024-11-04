# ~/miniconda3/envs/dlim
# -*- coding : utf-8 -*-

from model import Log_LSTM_VAE as Model  # LSTM_expVAE
# model = Model(num_layers=2, lstm_hid_size=6, linear_hid_size=8, output_size=6)  # 
model = Model(bi_di=True)

LABL = 'finetune5_'
###  Feb 21
# rvae1     try parameters          30 epochs   all NaN
# rvae1_    lr  0.001 -> 0.01       10 epochs   improvement in loss, NaN after epoch 6
# rvae2_    mu0  1 -> 0             10 epochs   even better performance
# rvae3_    lstm_hid_size 10 -> 8   10 epochs   a little worse than above two
###  Feb 22
# rvae4_    adjust layer width      10 epochs   all NaN until 10-15-10, worse result than ever, specific see 'feb22.txt'
# rvae5_    adjust layer width      10 epochs   NaN until 10-16-10, a little worse than best result
#  Feb 23
# rvae6_    2 layer of lstm         10 epochs   better than ever, best at 'rvae6_9'
# rvae6__   forgotten(all NaN)      20 epochs
###  Feb 27 28
# dent_     dentategyrus dataset    20 epochs   trapped with NaN problem no matter what parameters
###  Mar 1
# tsnan1_   add invert              20 epochs
# tsnan2_   add clip to exp         20 epochs   nan after a few epochs
###  Mar 8
# lognorm1  log-norm for latent     20 epochs   not bad result, but LABL remained tsnan1__, forgot LOSS_LAMBDA
###  Mar 9
# lognorm1  continue log-norm       20 epochs   mainly similar results with above, some very low loss but best at 17 regarding integraty
# lognorm2  lr=0.1, start randomly  20 epochs   failed
# lognorm3  lr=0.001, start by 17   20 epochs   No improvement. Annnnnd, I've not changed model? OH NOOOO!
###  Mar 10
# lognorm4  hope no wrong           20 epochs   velo has +/-, but extremely unbalance. Found wrong VAE calculation
# lognorm5  corrected VAE, lr=0.1   20 epochs   Inf emerge on epoch 10, continue from 9th model with lower lr
# lognorm6  lr=0.01                 20 epochs   similar results, extremely large beta and positive velocity
###  Mar 11
# maxsimp1  simplify u0_&s0_        20 epochs
###  Mar 12
# scale1    scale u/s by non 0 std  20 epochs	out of memories
###  Mar 13
# scale1    scale also with max     20 epochs   stopped by bug before plotting
# scale2    correct var by std      20 epochs   still +inf slope for x(t)
###  Mar 14 & 15
# distribution1 norm_z**2 to +      10 epochs   largely impacted by (u, s) = (~inf, ~0) datas
# distribution2 exp Dis with **2    10 epochs   NaN ?
# distribution3 norm_z**2, slope100 10 epochs   still impacted
# distribution4 norm_z**2, slope10  10 epochs   forgot to switch off SCALE
# distrubution5 mu**2+std, slop10   30 epochs
###  Mar 16
# extreme1  separate slope; no scale; 10 epochs         (Ablation-1: no scale)
# extreme2  separate slope; scale 10; 10 epochs; scale is necessary
# extreme3  separate scale by s.max;  30 epochs  bad result (Ablation-1: s.max scale)
# extreme4  separate scale by SCALE;  30 epochs  not satisfying, curves fit unperfectly
# extreme5  SLOPE=5, LOSS_LAMBDA=0.01;30 epochs  not satisfying
###  Mar 17
# finetune1 try exp distribution    20 epochs   failed
# finetune2 try u/s_max by model&bi 20 epochs
# finetune3 adjust u/s_max optim    20 epochs
# finetune4 per5epoch,0.1*u/s_max,>=1; 25 epochs; epoch 14/19 seems best
# finetune5 same setting for bm     25 epochs;  	(SCALE 10, square)
###  Mar 19
# finetune6				(SCALE 1, square)
# super1		dentate	SCALE100	square(slope)
# super2		dentate	SCALE10	square(slope)
# super3		dentate	SCALE1	square(slope)
# super4		dentate	SCALE4	square(slope)
# super5		dentate	SCALE1	exp(slope)
# super6		dentate	SCALE10	exp(slope)
# super7		dentate	SCALE100	exp(slope)

# super8		dentate	SCALE1	exp
# super9		dentate	SCALE1	square
###  Mar 20
# super10		dentate	SCALE10	square
# super11		dentate	SCALE10	exp
# super12		dentate	SCALE50	exp
# super13		dentate	SCALE50	square

# ablation1	dentate	2l-LSTM
# ablation2	dentate	attention
# ablation3	dentate	1l-LSTM+attention

# retry1		dentate	complete structure, 99 percentage-meaned scaled_max; US_LOSS 0.1 * 2 太高，尤其下调；中间反
# retry2		dentate	US_LOSS 0.01 * 3	                                                 整体不错，下调略高；中间反
# retry3		dentate	99.7 percentage-meaned scaled_max; US_LOSS 0.1 * 1.5                 下调太高；整体混乱，ME反
# retry04	√	dentate	US_LOSS 0.5 * 1.5	                                                训到最后还不错；方向也差不多
# retry4		dentate	US_LOSS 0.5 * 1.1	                                                不错；开始和最后ME反了
# retry5		dentate	US_LOSS 0.3 * 1.5, LR 0.01 * 0.6                                    上半与整相位判断失误
# retry6		dentate	US_LOSS 0.2 * 1.5	                                                上半下半不太稳定；O部分是反的
# retry7		dentate	US_LOSS 0.5; LOSS_LAMBDA 0.01                                       不错；ME和O是错的或空的
# retry7		dentate	US_LOSS 0.16 * 1.5	                                                下调过低；还可以，O部分箭头消失
# retry8		dentate	US_LOSS 0.4 * 1.1	                                                中途下调略低，后面还可以；ME是反的
# retry9	√	dentate	US_LOSS 0.4 * 1.2	                                                训到最后还可以；小部分消失
# retry10		dentate	US_LOSS 0.2 * 1.5; per 10 epochs                                    中途还行，最后下调偏高或上调偏低；方向最后还可以
# retry11		dentate	US_LOSS 0.2 * 2 	                                                偏低；最后的ME方向错了；
# retry12		dentate	LOSS_LAMBDA 0.01 * 2                                                前面还行，后面乱了；箭头方向还可以
# retry13		dentate	LOSS_LAMBDA 0.01	                                                上调偏低；前中方向还可以
# retry15		dentate	LOSS_LAMBDA 0.01 * 1.1, US_LOSS 0.2 * 2, LR 0.1 * 0.1               下调极低；ME是反的
# retry16	√	dentate	LR 0.01 * 0.1	                                                    还不错；小类方向消失
# retry17		dentate	US_LOSS 0.5 * 2                                                     偏低；前面ME是错的，后面小类消失
# retry18		dentate	US_LOSS 0.8 * 1.1	                                                还可以，上半相位判断错误；小类错误
# retry19		dentate	US_LOSS 0.4 * 1.5	                                                下半相位下调偏低；ME错误
# retry20		dentate	US_LOSS 0.5 * 1.4	                                                下调太高到离谱；小类消失


EPSILON = 1e-3
LOSS_LAMBDA = 0.01  # 0.5 before Mar11 simplification
US_LAMBDA = 0.5

use_M = True
SCALE = 10  # 's'  # None
SLOPE = 8

NAME = 'bonemarrow'
cached_adata = '/public/home/jiangyz/velo/data/BoneMarrow/scomplete.h5ad'
MODEL_DIR = '/public/home/jiangyz/velo/model'
MODEL_INIT = None  # 'maxsimp1_5'  # No 'pkl' !
# rvae for BomeMarrow with best super-parameter group: '/rvae6_9.pkl' x

USE_HVG = True
N_batch = 3
EPOCHES = range(30)
LR = 0.01

# GENE_LIST = ['CA1', 'EBF1', 'FOS', 'IRF8', 'CXCR4', 'SCT'] + ['ARPP21', 'AZU1', 'MPO'] + ['AHSP', 'AVP', 'HBB']  # bone marrow
# GENE_LIST = ['Hbb-bt', 'Hba-a1', 'Ptgds', 'Cst3', 'Bsg', 'Vtn'] + ['Tmsb10', 'Ppp3ca', 'Hn1']  # dentate gyrus
GENE_LIST = ['Apoa1', 'Phlda2', 'Noto'] + ['Krt18', 'Dab2', 'Emb', 'Slc2a3'] + ['Krt18', 'Mest', 'Pmp22']  # gastrulation
GENE_LIST = ['STMN2'] + ['ACTB', 'ACTG1', 'HSPB1', 'PTN'] + ['FOS', 'HSP90AA1', 'NEUROD6']

fig_epoches = range(9, 30, 10)
S_PNG = True
