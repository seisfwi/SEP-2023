import argparse
import os
import sys
import torch
import numpy as np

sys.path.append("../Ops/FWI")
from FWI_ops import FWI_obscalc
import fwi_utils as ft


# Get parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument('--generate_data', action='store_true')
parser.add_argument('--exp_name', type=str, default='/scratch/users/haipeng/000-Forward-Benchmark')
parser.add_argument('--nIter', type=int, default=5)
parser.add_argument('--ngpu', type=int, default=1)
args = vars(parser.parse_args())
generate_data = args['generate_data']
exp_name = args['exp_name']
nIter = args['nIter']
ngpu = args['ngpu']

# Define parameters
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


# Set source and receiver parameters
f0_vec = [10.0]
ind_src_x =   np.arange(10, nx-10, 10).astype(int)
ind_src_z = 1*np.ones(ind_src_x.shape[0]).astype(int)
ind_rec_x =   np.arange(10, nx-10).astype(int)
ind_rec_z = 90*np.ones(ind_rec_x.shape[0]).astype(int)

# Generate parameter files
para_fname    = os.path.join(exp_name, 'para_file.json')
survey_fname  = os.path.join(exp_name, 'survey_file.json')
data_dir_name = os.path.join(exp_name, 'Data')
ft.paraGen(nz_pad, nx_pad, dz, dx, nt, dt, f0_vec[0], nPml, nPad, para_fname, survey_fname, data_dir_name)
ft.surveyGen(ind_src_z, ind_src_x, ind_rec_z, ind_rec_x, survey_fname)

# Set source time function
Stf = ft.sourceGene(f0_vec[0], nt, dt)
th_Stf = torch.tensor(Stf, dtype=torch.float32, requires_grad=False).repeat(len(ind_src_x), 1)
Shot_ids = torch.tensor(np.arange(0,len(ind_src_x)), dtype=torch.int32)


## Generate obs data
if generate_data:

    # load the model
    vp_true  = np.loadtxt('./Models/HOMO-P-WAVE-VELOCITY-101-201.txt').astype('float32')
    vs_true  = np.loadtxt('./Models/HOMO-S-WAVE-VELOCITY-101-201.txt').astype('float32')
    den_true = np.loadtxt('./Models/HOMO-DENSITY-101-201.txt').astype('float32')

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
    
    # exit
    sys.exit('End of Data Generation')


else:
    sys.exit('Only Forward Modeling')
