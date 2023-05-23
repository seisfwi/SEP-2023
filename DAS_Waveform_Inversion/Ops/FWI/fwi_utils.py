import torch
import torch.nn.functional as F
from collections import OrderedDict
import json
import os
import numpy as np




def padding_numpy_array(arr, npml, npad):
    # Get the original shape of the array
    n_rows, n_cols = arr.shape
    
    # Create a new array with the padded dimensions
    padded_arr = np.zeros((n_rows + 2 * npml + npad, n_cols + npml*2), dtype=arr.dtype)
    
    # Copy the original array into the padded array
    padded_arr[npml:n_rows+npml, npml:n_cols+npml] = arr
    
    # Fill the boundary values with the corresponding values from the original array
    padded_arr[:npml, npml:-npml] = arr[0, :]
    padded_arr[-(npml+npad):, npml:-npml] = arr[-1, :]
    padded_arr[:, :npml] = padded_arr[:, npml].repeat(npml).reshape(-1, npml)
    padded_arr[:, -npml:] = padded_arr[:, -npml-1].repeat(npml).reshape(-1, npml)
    
    return padded_arr



def padding(cp, cs, den, nz_orig, nx_orig, nz, nx, nPml, nPad):
    tran_cp = cp.view(1, 1, nz_orig, nx_orig)
    tran_cs = cs.view(1, 1, nz_orig, nx_orig)
    tran_den = den.view(1, 1, nz_orig, nx_orig)
    tran_cp2 = F.interpolate(tran_cp, size=(nz,nx), mode='bilinear', align_corners=False)
    tran_cs2 = F.interpolate(tran_cs, size=(nz,nx), mode='bilinear', align_corners=False)
    tran_den2 = F.interpolate(tran_den, size=(nz,nx), mode='bilinear', align_corners=False)
    tran_cp3 = F.pad(tran_cp2, pad=(nPml, nPml, nPml, (nPml+nPad)), mode='replicate')
    tran_cs3 = F.pad(tran_cs2, pad=(nPml, nPml, nPml, (nPml+nPad)), mode='replicate')
    tran_den3 = F.pad(tran_den2, pad=(nPml, nPml, nPml, (nPml+nPad)), mode='replicate')
    cp_pad = tran_cp3.view(nz+2*nPml+nPad, nx+2*nPml)
    cs_pad = tran_cs3.view(nz+2*nPml+nPad, nx+2*nPml)
    den_pad = tran_den3.view(nz+2*nPml+nPad, nx+2*nPml)
    return cp_pad, cs_pad, den_pad

def paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, para_fname, survey_fname, \
    data_dir_name, if_win=False, filter_para=None, if_src_update=False, \
    scratch_dir_name='', if_cross_misfit=False):
    # para = OrderedDict()
    para = {}
    para['nz'] = nz
    para['nx'] = nx
    para['dz'] = dz
    para['dx'] = dx
    para['nSteps'] = nSteps
    para['dt'] = dt
    para['f0'] = f0
    para['nPoints_pml'] = nPml
    para['nPad'] = nPad

    if if_win != False:
        para['if_win'] = True

    if filter_para != None:
        para['filter'] = filter_para

    if if_src_update != False:
        para['if_src_update'] = True
    
    para['survey_fname'] = survey_fname
    para['data_dir_name'] = data_dir_name

    os.makedirs(data_dir_name, exist_ok=True)
 
    if if_cross_misfit != False:
        para['if_cross_misfit'] = True
    
    if(scratch_dir_name != ''):
        para['scratch_dir_name'] = scratch_dir_name
        os.makedirs(scratch_dir_name, exist_ok=True)

    with open(para_fname, 'w') as fp:
        json.dump(para, fp)


# all shots share the same number of receivers
def surveyGen(z_src, x_src, z_rec, x_rec, survey_fname, Windows=None, \
    Weights=None, Src_Weights=None, Src_rxz=None, Rec_rxz=None):
    x_src = x_src.tolist()
    z_src = z_src.tolist()
    x_rec = x_rec.tolist()
    z_rec = z_rec.tolist()
    nsrc = len(x_src)
    nrec = len(x_rec)
    survey = {}
    survey['nShots'] = nsrc
    for i in range(0, nsrc):
        shot = {}
        shot['z_src'] = z_src[i]
        shot['x_src'] = x_src[i]
        shot['nrec'] = nrec
        shot['z_rec'] = z_rec
        shot['x_rec'] = x_rec
        if Windows != None:
            shot['win_start'] = Windows['shot' + str(i)][:start]
            shot['win_end'] = Windows['shot' + str(i)][:end]
    
        if Weights != None:
            shot['weights'] = Weights['shot' + str(i)][:weights]
            
        if Src_Weights != None:
            shot['src_weight'] = Src_Weights[i]
            
        if Src_rxz != None:
            shot['src_rxz'] = Src_rxz[i]
            
        if Rec_rxz != None:
            Rec_rxz.tolist()
            shot['rec_rxz'] = Rec_rxz
        
        survey['shot' + str(i)] = shot
    
    with open(survey_fname, 'w') as fp:
        json.dump(survey, fp)


def sourceGene(f, nStep, delta_t):
#  Ricker wavelet generation and integration for source
#  Dongzhuo Li @ Stanford
#  May, 2015

  e = np.pi * np.pi * f * f
  t_delay = 1.2/f
  source = np.zeros((nStep))
  amp = 1.0e7
  for it in range(0,nStep):
      source[it] = (1-2*e*(delta_t*(it)-t_delay)**2)*np.exp(-e*(delta_t*(it)-t_delay)**2) * amp
      
  # change by Haipeng Li
  return source

#   for it in range(1,nStep):
#       source[it] = source[it] + source[it-1]

#   source = source * delta_t

#   return source
  



## velocity and density based on Voigt-Reuss-Hill boundary
def pcs2dv_vrh(phi, cc, sw):

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
    
    kv = (1 - phi) * (k_c * cc + k_q * (1 - cc)) + phi * (k_w * sw + k_h * (1 - sw))
    kr_1 = (1 - phi) * (cc/k_c + (1 - cc)/k_q) + phi * (sw/k_w + (1 - sw)/k_h)
    kr = 1/kr_1
    k = 0.5 * (kv + kr)
    
    muv = (1 - phi) * (mu_c * cc + mu_q * (1 - cc))
    mur = 0
    
    # mu is zero by Reuss
    mu = 0.5 * (muv + mur)
    
    # Properties of the effective fluid
    rho_f = rho_w * sw + rho_h * (1 - sw)
    
    # Properties of the effective skeleton
    rho_s = rho_c * cc + rho_q * (1 - cc)    
    
    # Effective unrained properties
    rho = rho_f * phi + rho_s * (1 - phi)
    
    # vp = np.sqrt((k + 4./3. * mu) / rho)
    lam = k - 2./3. * mu
    vp = np.sqrt((lam + 2. * mu) / rho)

    # I modify here to make a larger vp/vs ratio
    vs = np.sqrt(mu / rho)
    
    return vp, vs, rho





def weighted_average(prop1, prop2, volume1):
        """
        weighted_average is a function to
        calculate the wighted average of properties

        Parameters
        ----------
            prop1 : float
                Property of the material 1
            prop2 : float
                Property of the material 2
            volume1 : float
                Value specifying the rational volume of the material 1

        Returns
        -------
            mixed : float
                Property of mixed material
        """
        volume2 = 1 - volume1
        mixed = prop1 * volume1 + prop2 * volume2

        return mixed
    
def vrh(prop1, prop2, volume1, method):
    """
    vrh performs Voigt-Reuss-Hill boundary

    [extended_summary]

    Parameters
    ----------
    prop1 : float
        Property of the material 1
    prop2 : float
        Property of the material 2
    volume1 : float
        Value specifying the rational volume of the material 1

    Returns
    -------
    mixed : float
        Property of mixed material
    """
    
    volume2 = 1 - volume1
    prop_voigt = (volume1 * prop1 + volume2 * prop2)
    prop_reuss = (1 / ((volume1/prop1) + (volume2/prop2)))
    
    mixed = 0.5 * (prop_voigt + prop_reuss)
    
    if method == 'Voigt':
        return prop_voigt
    
    elif method =='Reuss':
        return prop_reuss
    
    elif method in ['VRH', 'vrh']:
        return mixed
    
def biot_gassmann(phi, k_f, k_s, k_d):
    Delta = delta_biot_gassmann(phi, k_f, k_s, k_d)

    denom = phi * (1 + Delta)

    k_u = (phi * k_d + (1 - (1 + phi) * (k_d / k_s)) * k_f) / denom
    C = k_f * (1 - k_d / k_s) / denom
    M = k_f / denom
    return k_u, C, M


def delta_biot_gassmann(phi, k_f, k_s, k_d):
    if (phi >= 1).any():
        phi = np.copy(phi) / 100
    return ((1 - phi) / phi) * (k_f / k_s) * (1 - (k_d / (k_s - k_s * phi)))


def drained_moduli(phi, k_s, g_s, cs):
    """
    drained_moduli computes the effective mechanical moduli KD and GD

    [extended_summary]

    Parameters
    ----------
    phi : float
        Porosity
    k_s : float
        Solid bulk modulus
    g_s : float
        Solid shear modulus
    cs : float
        general consolidation parameter

    Returns
    -------
    k_d : float
        Effective drained bulk modulus
    
    g_d : float
        Effective drained shear modulus
        
    References
    ----------
    Dupuy et al., 2016, Estimation of rock physics properties from seismic attributes — Part 1: Strategy and sensitivity analysis,
    Geophysics
    """
    if (phi >= 1).any():
        phi = np.copy(phi) / 100

    k_d = k_s * ((1 - phi) / (1 + cs * phi))

    g_d = g_s * ((1 - phi) / (1 + 1.5 * cs * phi))
    return k_d, g_d



def pcs2dv_gassmann(phi, cc, sw, method='Voigt'):
    
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
    
    # Properties of the effective fluid
    rho_f = weighted_average(rho_w, rho_h, sw)
    k_f = weighted_average(k_w, k_h, sw)
    
    # Properties of the effective skeleton
    k_s = vrh(k_c, k_q, cc, method)
    mu_s = vrh(mu_c, mu_q, cc, method)
    rho_s = weighted_average(rho_c, rho_q, cc)

    k_d, mu_d = drained_moduli(phi, k_s, mu_s, cs) 
    
    # Effective unrained properties
    k_u, _, _ = biot_gassmann(phi, k_f, k_s, k_d)
    rho = weighted_average(rho_f, rho_s, phi)
    
    vp = np.sqrt((k_u + 0.75 * mu_d) / rho)
    vs = np.sqrt(mu_d / rho)
    
    return vp, vs, rho