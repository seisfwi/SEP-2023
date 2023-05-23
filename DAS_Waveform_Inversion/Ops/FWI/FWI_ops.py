import torch
import torch.nn as nn
import numpy as np 
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import os
from scipy import optimize
import fwi_utils as ft
from collections import OrderedDict

abs_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(abs_path, 'Src')
os.makedirs(path+'/build/', exist_ok=True)

def load_fwi(path):
    fwi = load(name="fwi",
            sources=[path+'/Torch_Fwi.cpp', path+'/Parameter.cpp', path+'/libCUFD.cu', path+'/el_stress.cu', path+'/el_velocity.cu', path+'/el_stress_adj.cu', path+'/el_velocity_adj.cu', path+'/Model.cu', path+'/Cpml.cu', path+'/utilities.cu',	path+'/Src_Rec.cu', path+'/Boundary.cu'],
            extra_cflags=[
                '-O3 -fopenmp -lpthread'
            ],
            extra_include_paths=['/usr/local/cuda/include', path+'/rapidjson'],
            extra_ldflags=['-L/usr/local/cuda/lib64 -lnvrtc -lcuda -lcudart -lcufft'],
            build_directory=path+'/build/',
            verbose=True)
    return fwi

fwi_ops = load_fwi(path)

# class FWIFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, Lambda, Mu, Den, Stf, gpu_id, Shot_ids, para_fname):
#         misfit, = fwi_ops.forward(Lambda, Mu, Den, Stf, gpu_id, Shot_ids, para_fname)
#         variables = [Lambda, Mu, Den, Stf]
#         ctx.save_for_backward(*variables)
#         ctx.gpu_id = gpu_id
#         ctx.Shot_ids = Shot_ids
#         ctx.para_fname = para_fname
#         return misfit

#     @staticmethod
#     def backward(ctx, grad_misfit):
#         outputs = fwi_ops.backward(*ctx.saved_variables, ctx.gpu_id, ctx.Shot_ids, ctx.para_fname)
#         grad_Lambda, grad_Mu, grad_Den, grad_stf = outputs
#         return grad_Lambda, grad_Mu, grad_Den, grad_stf, None, None, None

class FWIFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Lambda, Mu, Den, Stf, ngpu, Shot_ids, para_fname):
        outputs = fwi_ops.backward(Lambda, Mu, Den, Stf, ngpu, Shot_ids, para_fname)
        ctx.outputs = outputs[1:]
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_misfit):
        grad_Lambda, grad_Mu, grad_Den, grad_stf = ctx.outputs

        # # save grad for backward on disk
        # np.save('grad_Lambda.npy', grad_Lambda.cpu().numpy())
        # np.save('grad_Mu.npy', grad_Mu.cpu().numpy())
        # np.save('grad_Den.npy', grad_Den.cpu().numpy())
        # np.save('grad_stf.npy', grad_stf.cpu().numpy())

        return grad_Lambda, grad_Mu, grad_Den, grad_stf, None, None, None


class FWI(torch.nn.Module):
    def __init__(self, Vp, Vs, Den, Stf, opt, Mask=None, Vp_bounds=None, \
        Vs_bounds=None, Den_bounds=None):
        super(FWI, self).__init__()

        self.nz = opt['nz']
        self.nx = opt['nx']
        self.nz_orig = opt['nz_orig']
        self.nx_orig = opt['nx_orig']
        self.nPml = opt['nPml']
        self.nPad = opt['nPad']

        self.Bounds = {}
        Vp_pad, Vs_pad, Den_pad = ft.padding(Vp, Vs, Den,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
        Vp_ref = Vp_pad.clone().detach()
        Vs_ref = Vs_pad.clone().detach()
        Den_ref = Den_pad.clone().detach()
        self.register_buffer('Vp_ref', Vp_ref)
        self.register_buffer('Vs_ref', Vs_ref)
        self.register_buffer('Den_ref', Den_ref)
        if Vp.requires_grad:
            self.Vp = nn.Parameter(Vp)
            if Vp_bounds != None:
                self.Bounds['Vp'] = Vp_bounds
        else:
            self.Vp = Vp
        if Vs.requires_grad:
            self.Vs = nn.Parameter(Vs)
            if Vs_bounds != None:
                self.Bounds['Vs'] = Vs_bounds
        else:
            self.Vs = Vs
        if Den.requires_grad:
            self.Den = nn.Parameter(Den)
            if Den_bounds != None:
                self.Bounds['Den'] = Den_bounds
        else:
            self.Den = Den

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml+self.nPad, \
                self.nx+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']


    def forward(self, Shot_ids, ngpu=1):
        Vp_pad, Vs_pad, Den_pad = ft.padding(self.Vp, self.Vs, self.Den,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
     
        Vp_mask_pad = self.Mask * Vp_pad + (1.0 - self.Mask) * self.Vp_ref
        Vs_mask_pad = self.Mask * Vs_pad + (1.0 - self.Mask) * self.Vs_ref
        Den_mask_pad = self.Mask * Den_pad + (1.0 - self.Mask) * self.Den_ref
        
        Lambda = (Vp_mask_pad**2 - 2.0 * Vs_mask_pad**2) * Den_mask_pad / 1e6
        Mu = Vs_mask_pad**2 * Den_mask_pad / 1e6

        return FWIFunction.apply(Lambda, Mu, Den_mask_pad, self.Stf, ngpu, Shot_ids, self.para_fname)



class FWI_obscalc(torch.nn.Module):
    def __init__(self, Vp, Vs, Den, Stf, para_fname):
        super(FWI_obscalc, self).__init__()
        self.Lambda = (Vp**2 - 2.0 * Vs**2) * Den  / 1e6
        self.Mu = Vs**2 * Den  / 1e6
        self.Den = Den
        self.Stf = Stf
        self.para_fname = para_fname

    def forward(self, Shot_ids, ngpu=1):
        fwi_ops.obscalc(self.Lambda, self.Mu, self.Den, self.Stf, ngpu, Shot_ids, self.para_fname)




class FWI_Lame_Den(torch.nn.Module):
    def __init__(self, Lam, Mu, Den, Stf, opt, Mask=None, Lam_bounds=None, \
        Mu_bounds=None, Den_bounds=None):
        super(FWI_Lame_Den, self).__init__()

        self.nz = opt['nz']
        self.nx = opt['nx']
        self.nz_orig = opt['nz_orig']
        self.nx_orig = opt['nx_orig']
        self.nPml = opt['nPml']
        self.nPad = opt['nPad']

        self.Bounds = {}
        Lam_pad, Mu_pad, Den_pad = ft.padding(Lam, Mu, Den,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
        Lam_ref = Lam_pad.clone().detach()
        Mu_ref = Mu_pad.clone().detach()
        Den_ref = Den_pad.clone().detach()
        self.register_buffer('Lam_ref', Lam_ref)
        self.register_buffer('Mu_ref', Mu_ref)
        self.register_buffer('Den_ref', Den_ref)
        
        if Lam.requires_grad:
            self.Lam = nn.Parameter(Lam)
            if Lam_bounds != None:
                self.Bounds['Lam'] = Lam_bounds
        else:
            self.Lam = Lam
        if Mu.requires_grad:
            self.Mu = nn.Parameter(Mu)
            if Mu_bounds != None:
                self.Bounds['Mu'] = Mu_bounds
        else:
            self.Mu = Mu
        if Den.requires_grad:
            self.Den = nn.Parameter(Den)
            if Den_bounds != None:
                self.Bounds['Den'] = Den_bounds
        else:
            self.Den = Den

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml+self.nPad, \
                self.nx+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']

    def forward(self, Shot_ids, ngpu=1):
        Lam_pad, Mu_pad, Den_pad = ft.padding(self.Lam, self.Mu, self.Den,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
     
        Lam_mask_pad = self.Mask * Lam_pad + (1.0 - self.Mask) * self.Lam_ref 
        Mu_mask_pad  = self.Mask * Mu_pad  + (1.0 - self.Mask) * self.Mu_ref
        Den_mask_pad = self.Mask * Den_pad + (1.0 - self.Mask) * self.Den_ref
  
        return FWIFunction.apply(Lam_mask_pad, Mu_mask_pad, Den_mask_pad, self.Stf, ngpu, Shot_ids, self.para_fname)



class FWI_IP_IS_Den(torch.nn.Module):
    def __init__(self, IP, IS, Den, Stf, opt, Mask=None, IP_bounds=None, \
        IS_bounds=None, Den_bounds=None):
        super(FWI_IP_IS_Den, self).__init__()

        self.nz = opt['nz']
        self.nx = opt['nx']
        self.nz_orig = opt['nz_orig']
        self.nx_orig = opt['nx_orig']
        self.nPml = opt['nPml']
        self.nPad = opt['nPad']

        self.Bounds = {}
        IP_pad, IS_pad, Den_pad = ft.padding(IP, IS, Den, self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
        IP_ref = IP_pad.clone().detach()
        IS_ref = IS_pad.clone().detach()
        Den_ref = Den_pad.clone().detach()
        self.register_buffer('IP_ref', IP_ref)
        self.register_buffer('IS_ref', IS_ref)
        self.register_buffer('Den_ref', Den_ref)
        if IP.requires_grad:
            self.IP = nn.Parameter(IP)
            if IP_bounds != None:
                self.Bounds['IP'] = IP_bounds
        else:
            self.IP = IP
        if IS.requires_grad:
            self.IS = nn.Parameter(IS)
            if IS_bounds != None:
                self.Bounds['IS'] = IS_bounds
        else:
            self.IS = IS
        if Den.requires_grad:
            self.Den = nn.Parameter(Den)
            if Den_bounds != None:
                self.Bounds['Den'] = Den_bounds
        else:
            self.Den = Den

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml+self.nPad, \
                self.nx+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']

    def forward(self, Shot_ids, ngpu=1):

        IP_pad, IS_pad, Den_pad = ft.padding(self.IP, self.IS, self.Den, self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
     
        IP_mask_pad  = self.Mask * IP_pad + (1.0 - self.Mask) * self.IP_ref
        IS_mask_pad  = self.Mask * IS_pad  + (1.0 - self.Mask) * self.IS_ref
        Den_mask_pad = self.Mask * Den_pad + (1.0 - self.Mask) * self.Den_ref

        Lambda = (IP_mask_pad ** 2 - 2. * IS_mask_pad ** 2) / Den_mask_pad 
        Mu = IS_mask_pad ** 2 / Den_mask_pad

        return FWIFunction.apply(Lambda, Mu, Den_mask_pad, self.Stf, ngpu, Shot_ids, self.para_fname)


class FWI_Vp_Vs_IP(torch.nn.Module):
    def __init__(self, Vp, Vs, IP, Stf, opt, Mask=None, Vp_bounds=None, \
        Vs_bounds=None, IP_bounds=None):
        super(FWI_Vp_Vs_IP, self).__init__()

        self.nz = opt['nz']
        self.nx = opt['nx']
        self.nz_orig = opt['nz_orig']
        self.nx_orig = opt['nx_orig']
        self.nPml = opt['nPml']
        self.nPad = opt['nPad']

        self.Bounds = {}
        Vp_pad, Vs_pad, IP_pad = ft.padding(Vp, Vs, IP, self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
        Vp_ref = Vp_pad.clone().detach()
        Vs_ref = Vs_pad.clone().detach()
        IP_ref = IP_pad.clone().detach()
        self.register_buffer('Vp_ref', Vp_ref)
        self.register_buffer('Vs_ref', Vs_ref)
        self.register_buffer('IP_ref', IP_ref)
        if Vp.requires_grad:
            self.Vp = nn.Parameter(Vp)
            if Vp_bounds != None:
                self.Bounds['Vp'] = Vp_bounds
        else:
            self.Vp = Vp
        if Vs.requires_grad:
            self.Vs = nn.Parameter(Vs)
            if Vs_bounds != None:
                self.Bounds['Vs'] = Vs_bounds
        else:
            self.Vs = Vs
        if IP.requires_grad:
            self.IP = nn.Parameter(IP)
            if IP_bounds != None:
                self.Bounds['IP'] = IP_bounds
        else:
            self.IP = IP

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml+self.nPad, \
                self.nx+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']


    def forward(self, Shot_ids, ngpu=1):
        Vp_pad, Vs_pad, IP_pad = ft.padding(self.Vp, self.Vs, self.IP, self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
     
        Vp_mask_pad = self.Mask * Vp_pad + (1.0 - self.Mask) * self.Vp_ref
        Vs_mask_pad = self.Mask * Vs_pad + (1.0 - self.Mask) * self.Vs_ref
        IP_mask_pad = self.Mask * IP_pad + (1.0 - self.Mask) * self.IP_ref
        
        Lambda = (IP_mask_pad * Vp_mask_pad - 2. * IP_mask_pad / Vp_mask_pad * Vs_mask_pad ** 2)
        Mu = IP_mask_pad / Vp_mask_pad * Vs_mask_pad ** 2
        Den = IP_mask_pad / Vp_mask_pad

        return FWIFunction.apply(Lambda, Mu, Den, self.Stf, ngpu, Shot_ids, self.para_fname)


class FWI_Vp_Vs_IS(torch.nn.Module):
    def __init__(self, Vp, Vs, IS, Stf, opt, Mask=None, Vp_bounds=None, \
        Vs_bounds=None, IS_bounds=None):
        super(FWI_Vp_Vs_IS, self).__init__()

        self.nz = opt['nz']
        self.nx = opt['nx']
        self.nz_orig = opt['nz_orig']
        self.nx_orig = opt['nx_orig']
        self.nPml = opt['nPml']
        self.nPad = opt['nPad']

        self.Bounds = {}
        Vp_pad, Vs_pad, IS_pad = ft.padding(Vp, Vs, IS,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
        Vp_ref = Vp_pad.clone().detach()
        Vs_ref = Vs_pad.clone().detach()
        IS_ref = IS_pad.clone().detach()
        self.register_buffer('Vp_ref', Vp_ref)
        self.register_buffer('Vs_ref', Vs_ref)
        self.register_buffer('IS_ref', IS_ref)
        if Vp.requires_grad:
            self.Vp = nn.Parameter(Vp)
            if Vp_bounds != None:
                self.Bounds['Vp'] = Vp_bounds
        else:
            self.Vp = Vp
        if Vs.requires_grad:
            self.Vs = nn.Parameter(Vs)
            if Vs_bounds != None:
                self.Bounds['Vs'] = Vs_bounds
        else:
            self.Vs = Vs

        if IS.requires_grad:
            self.IS = nn.Parameter(IS)
            if IS_bounds != None:
                self.Bounds['IS'] = IS_bounds
        else:
            self.IS = IS

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml+self.nPad, \
                self.nx+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']

    def forward(self, Shot_ids, ngpu=1):
        Vp_pad, Vs_pad, IS_pad = ft.padding(self.Vp, self.Vs, self.IS,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
     
        Vp_mask_pad = self.Mask * Vp_pad + (1.0 - self.Mask) * self.Vp_ref
        Vs_mask_pad = self.Mask * Vs_pad + (1.0 - self.Mask) * self.Vs_ref
        IS_mask_pad = self.Mask * IS_pad + (1.0 - self.Mask) * self.IS_ref
        
        Lambda = (IS_mask_pad/ Vs_mask_pad *  Vp_mask_pad**2 - 2.0 *  IS_mask_pad * Vs_mask_pad)
        Mu = IS_mask_pad * Vs_mask_pad
        Den = IS_mask_pad / Vs_mask_pad

        return FWIFunction.apply(Lambda, Mu, Den, self.Stf, ngpu, Shot_ids, self.para_fname)




### Rock physics model with PCS parameters: porosity, clay conten, and saturation
class FWI_Rock_Physics_VRH(torch.nn.Module):
    def __init__(self, PHI, CC, SW, Stf, opt, Mask=None, PHI_bounds=None, CC_bounds=None, SW_bounds=None):
        super(FWI_Rock_Physics_VRH, self).__init__()

        self.nz = opt['nz']
        self.nx = opt['nx']
        self.nz_orig = opt['nz_orig']
        self.nx_orig = opt['nx_orig']
        self.nPml = opt['nPml']
        self.nPad = opt['nPad']

        self.Bounds = {}
        PHI_pad, CC_pad, SW_pad = ft.padding(PHI, CC, SW,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
        PHI_ref = PHI_pad.clone().detach()
        CC_ref = CC_pad.clone().detach()
        SW_ref = SW_pad.clone().detach()
        self.register_buffer('PHI_ref', PHI_ref)
        self.register_buffer('CC_ref', CC_ref)
        self.register_buffer('SW_ref', SW_ref)
        if PHI.requires_grad:
            self.PHI = nn.Parameter(PHI)
            if PHI_bounds != None:
                self.Bounds['PHI'] = PHI_bounds
        else:
            self.PHI = PHI
        if CC.requires_grad:
            self.CC = nn.Parameter(CC)
            if CC_bounds != None:
                self.Bounds['CC'] = CC_bounds
        else:
            self.CC = CC

        if SW.requires_grad:
            self.SW = nn.Parameter(SW)
            if SW_bounds != None:
                self.Bounds['SW'] = SW_bounds
        else:
            self.SW = SW

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml+self.nPad, \
                self.nx+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']


    def forward(self, Shot_ids, ngpu=1):
        PHI_pad, CC_pad, SW_pad = ft.padding(self.PHI, self.CC, self.SW,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
     
        PHI_mask_pad = self.Mask * PHI_pad + (1.0 - self.Mask) * self.PHI_ref
        CC_mask_pad  = self.Mask * CC_pad + (1.0 - self.Mask) * self.CC_ref
        SW_mask_pad  = self.Mask * SW_pad + (1.0 - self.Mask) * self.SW_ref
        
        # Lambda, mu, and density based on Voigt-Reuss-Hill boundary
        k_q = 37.00 * 1e9
        k_c = 21.00 * 1e9
        k_w =  2.25 * 1e9
        k_h =  0.04 * 1e9
        
        mu_q = 44.00 * 1e9
        mu_c = 10.00 * 1e9

        rho_q = 2.65 * 1e3
        rho_c = 2.55 * 1e3
        rho_w = 1.00 * 1e3
        rho_h = 0.10 * 1e3
        
        kv = (1 - PHI_mask_pad) * (k_c * CC_mask_pad + k_q * (1 - CC_mask_pad)) + PHI_mask_pad * (k_w * SW_mask_pad + k_h * (1 - SW_mask_pad))
        kr_1 = (1 - PHI_mask_pad) * (CC_mask_pad/k_c + (1 - CC_mask_pad)/k_q) + PHI_mask_pad * (SW_mask_pad/k_w + (1 - SW_mask_pad)/k_h)
        kr = 1/kr_1
        k = 0.5 * (kv + kr)
        
        muv = (1 - PHI_mask_pad) * (mu_c * CC_mask_pad + mu_q * (1 - CC_mask_pad))
        mur = 0
        
        # mu is zero by Reuss
        Mu = 0.5 * (muv + mur)
        
        # Properties of the effective fluid
        rho_f = rho_w * SW_mask_pad + rho_h * (1 - SW_mask_pad)
        
        # Properties of the effective skeleton
        rho_s = rho_c * CC_mask_pad + rho_q * (1 - CC_mask_pad)    
        
        # Effective unrained properties
        Den = rho_f * PHI_mask_pad + rho_s * (1 - PHI_mask_pad)
        
        # vp = np.sqrt((k + 4./3. * mu) / rho)
        Lambda = (k - 2./3. * Mu ) 

        Lambda = Lambda / 1e6
        Mu = Mu / 1e6

        # VP = torch.sqrt((Lambda + 2.0 * Mu) / Den)
        # VS = torch.sqrt(Mu / Den)

        # print('VP: ', VP.min(), VP.max())
        # print('VS: ', VS.min(), VS.max())
        # print('Den: ', Den.min(), Den.max())

        # exit()

        return FWIFunction.apply(Lambda, Mu, Den, self.Stf, ngpu, Shot_ids, self.para_fname)




### Rock physics model with PCS parameters: porosity, clay conten, and saturation
# The rock physics model is based on Gassmann's equation and the related code is from
# https://github.com/AmirMardan/PyFWI
class FWI_Rock_Physics_gassmann(torch.nn.Module):
    def __init__(self, PHI, CC, SW, Stf, opt, Mask=None, PHI_bounds=None, CC_bounds=None, SW_bounds=None):
        super(FWI_Rock_Physics_gassmann, self).__init__()

        self.nz = opt['nz']
        self.nx = opt['nx']
        self.nz_orig = opt['nz_orig']
        self.nx_orig = opt['nx_orig']
        self.nPml = opt['nPml']
        self.nPad = opt['nPad']

        self.Bounds = {}
        PHI_pad, CC_pad, SW_pad = ft.padding(PHI, CC, SW,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
        PHI_ref = PHI_pad.clone().detach()
        CC_ref = CC_pad.clone().detach()
        SW_ref = SW_pad.clone().detach()
        self.register_buffer('PHI_ref', PHI_ref)
        self.register_buffer('CC_ref', CC_ref)
        self.register_buffer('SW_ref', SW_ref)
        if PHI.requires_grad:
            self.PHI = nn.Parameter(PHI)
            if PHI_bounds != None:
                self.Bounds['PHI'] = PHI_bounds
        else:
            self.PHI = PHI
        if CC.requires_grad:
            self.CC = nn.Parameter(CC)
            if CC_bounds != None:
                self.Bounds['CC'] = CC_bounds
        else:
            self.CC = CC

        if SW.requires_grad:
            self.SW = nn.Parameter(SW)
            if SW_bounds != None:
                self.Bounds['SW'] = SW_bounds
        else:
            self.SW = SW

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml+self.nPad, \
                self.nx+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']



    def forward(self, Shot_ids, ngpu=1):
        PHI_pad, CC_pad, SW_pad = ft.padding(self.PHI, self.CC, self.SW,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
     
        PHI_mask_pad = self.Mask * PHI_pad + (1.0 - self.Mask) * self.PHI_ref
        CC_mask_pad  = self.Mask * CC_pad + (1.0 - self.Mask) * self.CC_ref
        SW_mask_pad  = self.Mask * SW_pad + (1.0 - self.Mask) * self.SW_ref
        
        k_q = 37.00 * 1e9
        k_c = 21.00 * 1e9
        k_w =  2.25 * 1e9
        k_h =  0.04 * 1e9
        
        mu_q = 44.00 * 1e9
        mu_c = 10.00 * 1e9

        rho_q = 2.65 * 1e3
        rho_c = 2.55 * 1e3
        rho_w = 1.00 * 1e3
        rho_h = 0.10 * 1e3
        cs = 20.0

        rho_f = rho_w * SW_mask_pad + rho_h * (1 - SW_mask_pad)
        k_f = k_w * SW_mask_pad + k_h * (1 - SW_mask_pad)
        k_s = k_c * CC_mask_pad + k_q * (1 - CC_mask_pad)
        mu_s = mu_c * CC_mask_pad + mu_q * (1 - CC_mask_pad)
        rho_s = rho_c * CC_mask_pad + rho_q * (1 - CC_mask_pad)

        k_d = k_s * ((1 - PHI_mask_pad) / (1 + cs * PHI_mask_pad))
        mu_d = mu_s * ((1 - PHI_mask_pad) / (1 + 1.5 * cs * PHI_mask_pad))

        Delta = ((1 - PHI_mask_pad) / PHI_mask_pad) * (k_f / k_s) * (1 - (k_d / (k_s - k_s * PHI_mask_pad)))

        denom = PHI_mask_pad * (1 + Delta)

        k_u = (PHI_mask_pad * k_d + (1 - (1 + PHI_mask_pad) * (k_d / k_s)) * k_f) / denom
        # C = k_f * (1 - k_d / k_s) / denom
        # M = k_f / denom

        rho = rho_f * PHI_mask_pad + rho_s * (1 - PHI_mask_pad)
        vp = torch.sqrt((k_u + 0.75 * mu_d) / rho)
        vs = torch.sqrt(mu_d / rho)

        ## Lambda, mu, and density based on Gasmmann's equation
        # vp, vs, rho = ft.pcs2dv_gassmann(PHI_mask_pad, CC_mask_pad, SW_mask_pad, method='Voigt')

        Lambda = rho * (vp**2 - 2 * vs**2) /1e6
        Mu = rho * vs**2 /1e6
        Den = rho

        # exit()

        return FWIFunction.apply(Lambda, Mu, Den, self.Stf, ngpu, Shot_ids, self.para_fname)





