import argparse
import os
import sys
import scipy.io as sio
import torch
import numpy as np
from scipy import optimize

sys.path.append("../Ops/FWI")
from FWI_ops import FWI_obscalc, FWI_Lame_Den
import fwi_utils as ft
from obj_wrapper import PyTorchObjective


## Get parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument('--generate_data', action='store_true')
parser.add_argument('--exp_name', type=str, default='/scratch/users/haipeng/Anomaly-Lame-Den')
parser.add_argument('--nIter', type=int, default=5)
parser.add_argument('--ngpu', type=int, default=1)
args = vars(parser.parse_args())
generate_data = args['generate_data']
exp_name = args['exp_name']
nIter = args['nIter']
ngpu = args['ngpu']

## Define paramete
dz = 20.0
dx = 20.0
nz = 101
nx = 201
dt = 0.002
nt = 1501
nPml = 32
nPad = int(32 - np.mod((nz+2*nPml), 32))
nz_pad = nz + 2*nPml + nPad
nx_pad = nx + 2*nPml

# Generate mask
Mask = np.zeros((nz_pad, nx_pad))
Mask[nPml:nPml+nz, nPml:nPml+nx] = 1.0
Mask[nPml:nPml+4,:] = 0.0
th_mask = torch.tensor(Mask, dtype=torch.float32)

# Set data filter
filter = [[0.0, 0.0, 2.0, 2.5],
         [0.0, 0.0, 2.0, 3.5],
         [0.0, 0.0, 2.0, 4.5],
         [0.0, 0.0, 2.0, 5.5],
         [0.0, 0.0, 2.0, 6.5],
         [0.0, 0.0, 2.0, 7.5]]

# Set source and receiver parameters
f0_vec = [10.0]
if_src_update = False
if_win = False

ind_src_x =   np.arange(10, nx-10, 10).astype(int)
ind_src_z = 1*np.ones(ind_src_x.shape[0]).astype(int)
ind_rec_x =   np.arange(10, nx-10).astype(int)
ind_rec_z =95*np.ones(ind_rec_x.shape[0]).astype(int)


exp_name = exp_name + '/'
para_fname    = exp_name + '/para_file.json'
survey_fname  = exp_name + '/survey_file.json'
data_dir_name = exp_name + '/Data'
ft.paraGen(nz_pad, nx_pad, dz, dx, nt, dt, f0_vec[0], nPml, nPad, para_fname, survey_fname, data_dir_name)
ft.surveyGen(ind_src_z, ind_src_x, ind_rec_z, ind_rec_x, survey_fname)

Stf = ft.sourceGene(f0_vec[0], nt, dt)
th_Stf = torch.tensor(Stf, dtype=torch.float32, requires_grad=False).repeat(len(ind_src_x), 1)
Shot_ids = torch.tensor(np.arange(0,len(ind_src_x)), dtype=torch.int32)


## Generate obs data
if generate_data:

    # load true model
    vp_true  = np.loadtxt('./Models/Anomaly_P-WAVE_VELOCITY_101_201_Lame_Den.txt').astype('float32')
    vs_true  = np.loadtxt('./Models/Anomaly_S-WAVE_VELOCITY_101_201_Lame_Den.txt').astype('float32')
    den_true = np.loadtxt('./Models/Anomaly_DENSITY_101_201_Lame_Den.txt').astype('float32')

    # pad the model
    vp_true_pad  = ft.padding_numpy_array(vp_true, nPml, nPad)
    vs_true_pad  = ft.padding_numpy_array(vs_true, nPml, nPad)
    den_true_pad = ft.padding_numpy_array(den_true, nPml, nPad)
    print(f'vp_true_pad shape = {vp_true_pad.shape}')

    # convert to torch tensor
    th_vp_pad  = torch.tensor(vp_true_pad,  dtype=torch.float32, requires_grad=False)
    th_vs_pad  = torch.tensor(vs_true_pad,  dtype=torch.float32, requires_grad=False)
    th_den_pad = torch.tensor(den_true_pad, dtype=torch.float32, requires_grad=False)
  
    # generate obs data
    fwi_obscalc = FWI_obscalc(th_vp_pad, th_vs_pad, th_den_pad, th_Stf, para_fname)
    fwi_obscalc(Shot_ids, ngpu=ngpu)

    sys.exit('End of Data Generation')


## Full Waveform Inversion
else:
    # save parameters for solver
    opt = {}
    opt['nz'] = nz
    opt['nx'] = nx 
    opt['nz_orig'] = nz
    opt['nx_orig'] = nx
    opt['nPml'] = nPml
    opt['nPad'] = nPad
    opt['para_fname'] = para_fname

    # load initial model
    vp_init  = np.loadtxt('./Models/Anomaly_P-WAVE_VELOCITY_101_201_Lame_Den_init.txt').astype('float32')
    vs_init  = np.loadtxt('./Models/Anomaly_S-WAVE_VELOCITY_101_201_Lame_Den_init.txt').astype('float32')
    den_init = np.loadtxt('./Models/Anomaly_DENSITY_101_201_Lame_Den_init.txt').astype('float32')

    print('Lame-Den scheme')
    lam_init = den_init * (vp_init**2 - 2.0 * vs_init**2) / 1e6
    mu_init  = den_init * vs_init**2 / 1e6

    th_lam_inv = torch.tensor(lam_init, dtype=torch.float32, requires_grad=True)
    th_mu_inv  = torch.tensor(mu_init,  dtype=torch.float32, requires_grad=True)
    th_den_inv = torch.tensor(den_init, dtype=torch.float32, requires_grad=True)

    Lam_bounds = None
    Mu_bounds = None
    Den_bounds = None

    fwi = FWI_Lame_Den(th_lam_inv, th_mu_inv, th_den_inv, th_Stf, opt, Mask=th_mask, Lam_bounds=Lam_bounds, \
            Mu_bounds=Mu_bounds, Den_bounds=Den_bounds)

 
    # set obj
    compLoss = lambda : fwi(Shot_ids, ngpu=ngpu)
    obj = PyTorchObjective(fwi, compLoss)

    # set callback function
    __iter = 0
    result_dir_name = exp_name + '/Results'
    def save_prog(x):
        global __iter

        # save misfit
        os.makedirs(result_dir_name, exist_ok=True)
        with open(result_dir_name + '/loss.txt', 'a') as text_file:
            text_file.write("%d %s\n" % (__iter, obj.f))
        
        # save model and gradient
        sio.savemat(result_dir_name + '/Lam'     + str(__iter) + '.mat', {'Lam':fwi.Lam.cpu().detach().numpy()})
        sio.savemat(result_dir_name + '/Mu'      + str(__iter) + '.mat', {'Mu':fwi.Mu.cpu().detach().numpy()})
        sio.savemat(result_dir_name + '/Den'     + str(__iter) + '.mat', {'Den':fwi.Den.cpu().detach().numpy()})
        sio.savemat(result_dir_name + '/grad_Lam'+ str(__iter) + '.mat', {'grad_Lam':fwi.Lam.grad.cpu().detach().numpy()})
        sio.savemat(result_dir_name + '/grad_Mu' + str(__iter) + '.mat', {'grad_Mu':fwi.Mu.grad.cpu().detach().numpy()})
        sio.savemat(result_dir_name + '/grad_Den'+ str(__iter) + '.mat', {'grad_Den':fwi.Den.grad.cpu().detach().numpy()})
    
        __iter = __iter + 1

    # run optimization
    optimize.minimize(obj.fun, obj.x0, method = 'L-BFGS-B', 
                    jac = obj.jac, bounds = obj.bounds, 
                    tol = None, callback = save_prog, 
                    options={
                        'disp': True, 
                        'iprint': 101, 
                        'gtol': 1e-016, 
                        'maxiter': nIter, 
                        'ftol': 1e-12, 
                        'maxcor': 5, 
                        'maxfun': 1500, 
                        'maxls':6})
