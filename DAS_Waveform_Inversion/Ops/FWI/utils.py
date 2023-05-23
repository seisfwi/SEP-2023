import json
import os
import numpy as np
from torch.utils.cpp_extension import load


def padding_np_array(arr, npml, npad):
    ''' Pad a numpy array based on the number of PML layers and padding layers.
        The boundary values are extended from the original array.
    '''

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


def source_ricker(f, nStep, delta_t):
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


# path of the source code and the build directory
abs_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(abs_path, 'Src')
os.makedirs(path+'/build/', exist_ok=True)

def load_fwi(path):
    fwi = load(name="fwi",
            sources=[path+'/Torch_Fwi.cpp', 
                     path+'/Parameter.cpp', 
                     path+'/libCUFD.cu', 
                     path+'/el_stress.cu', 
                     path+'/el_velocity.cu', 
                     path+'/el_stress_adj.cu', 
                     path+'/el_velocity_adj.cu', 
                     path+'/Model.cu', 
                     path+'/Cpml.cu', 
                     path+'/utilities.cu',
                     path+'/Src_Rec.cu', 
                     path+'/Boundary.cu'],

            extra_cflags=['-O3 -fopenmp -lpthread'],
            extra_include_paths=['/usr/local/cuda/include', path+'/rapidjson'],
            extra_ldflags=['-L/usr/local/cuda/lib64 -lnvrtc -lcuda -lcudart -lcufft'],
            build_directory=path+'/build/',
            verbose=True)
    return fwi

fwi_ops = load_fwi(path)