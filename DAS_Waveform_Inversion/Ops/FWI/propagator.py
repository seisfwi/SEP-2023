# The class of defining the propagator for different wave equations

import torch
import numpy as np
from utils import padding_np_array, source_ricker, paraGen, surveyGen
from utils import fwi_ops

class Propagator(object):
    ''' The class of defining the propagator for different wave equations
    '''

    def __init__(self, model, survey):
        ''' Initialize the propagator

        Parameters:
        -----------
        model: Model object
            The model object containing the model parameters
        survey: survey object
            The survey object containing the source and receiver parameters
        '''
        self.model = model
        self.survey = survey

    def apply_forward(self, data):
        ''' Forward propagation

        Parameters:
        -----------
        data: data object
            The data object containing the waveform and wavefield data
        '''
        raise NotImplementedError
    
    def apply_adjoint(self, data):
        ''' Adjoint propagation

        Parameters:
        -----------
        data: data object
            The data object containing the waveform and wavefield data
        '''
        raise NotImplementedError
    
    def apply_gradient(self, gradient):
        ''' Gradient calculation

        Parameters:
        -----------
        gradient: gradient object
            The gradient object containing the gradient of the objective 
            function w.r.t. the model parameters.
        '''
        raise NotImplementedError
    


class ElasticPropagator(Propagator):
    ''' The class of defining the propagator for the isotropic elastic wave 
        equation (stress-velocity form), which is solved by the finite 
        difference method.
    '''

    def __init__(self, model, survey):
        ''' Initialize the propagator

        Parameters:
        -----------
        model: Model object
            The model object containing the model parameters
        survey: survey object
            The survey object containing the source and receiver parameters
        '''

        # initialize the propagator
        super().__init__(model, survey)



    # rewrite the forward propagation function
    def apply_forward(self, ngpu=1):
        ''' Forward propagation
        '''
        
        # get the model parameters
        nx = self.model.nx
        nz = self.model.nz
        dx = self.model.dx
        dz = self.model.dz
        nt = self.model.nt
        dt = self.model.dt
        nPml = self.model.nPml

        # set derived parameters
        nPad = int(32 - np.mod((nz+2*nPml), 32))
        nz_pad = nz + 2*nPml + nPad
        nx_pad = nx + 2*nPml

        # get the material parameters
        vp = self.model.vp
        vs = self.model.vs
        rho = self.model.rho
        lam = rho * (vp**2 - 2 * vs**2)
        mu = rho * vs**2

        # get the source parameters
        f0 = self.survey.f0
        src_x = self.survey.src_x   # refer to the index, not the coordinate
        src_z = self.survey.src_z   # refer to
        src_num = len(src_x)
        stf = source_ricker(f0, nt, dt)
        th_stf = torch.tensor(stf, dtype=torch.float32, requires_grad=False).repeat(src_num, 1)
        shot_ids = torch.tensor(np.arange(0, src_num), dtype=torch.int32)

        # get the receiver parameters
        rec_x = self.survey.rec_x  # refer to the index, not the coordinate
        rec_z = self.survey.rec_z  # refer to the index, not the coordinate

        # pad the material parameters and convert to torch tensor
        lam_pad = padding_np_array(lam, nPml, nPad)
        mu_pad  = padding_np_array(mu, nPml, nPad)
        rho_pad = padding_np_array(rho, nPml, nPad)
        th_lam_pad = torch.tensor(lam_pad, dtype=torch.float32, requires_grad=False)
        th_mu_pad  = torch.tensor(mu_pad,  dtype=torch.float32, requires_grad=False)
        th_rho_pad = torch.tensor(rho_pad, dtype=torch.float32, requires_grad=False)
    
        # write the para_file
        exp_name =  self.model.exp_name
        para_fname    = exp_name + '/para_file.json'
        survey_fname  = exp_name + '/survey_file.json'
        data_dir_name = exp_name + '/Data'
        paraGen(nz_pad, nx_pad, dz, dx, nt, dt, f0, nPml, nPad, para_fname, survey_fname, data_dir_name)
        surveyGen(src_z, src_x, rec_z, rec_x, survey_fname)
        print('The parameter file has been written to: ', para_fname)
        
        # run the forward propagation code
        fwi_ops.obscalc(th_lam_pad, th_mu_pad, th_rho_pad, th_stf, ngpu, shot_ids, para_fname)



    def apply_gradient(self, model_init, ngpu=1):
        ''' Gradient calculation

        Parameters:
        -----------
        gradient: gradient object
            The gradient object containing the gradient of the objective 
            function w.r.t. the model parameters.
        '''

        # get the model parameters
        nx = self.model.nx
        nz = self.model.nz
        dx = self.model.dx
        dz = self.model.dz
        nt = self.model.nt
        dt = self.model.dt
        nPml = self.model.nPml

        # set derived parameters
        nPad = int(32 - np.mod((nz+2*nPml), 32))
        nz_pad = nz + 2*nPml + nPad
        nx_pad = nx + 2*nPml

        # get the material parameters
        vp = model_init.vp
        vs = model_init.vs
        rho = model_init.rho
        lam = rho * (vp**2 - 2 * vs**2)
        mu = rho * vs**2

        # get the source parameters
        f0 = self.survey.f0
        src_x = self.survey.src_x   # refer to the index, not the coordinate
        src_z = self.survey.src_z   # refer to
        src_num = len(src_x)
        stf = source_ricker(f0, nt, dt)
        th_stf = torch.tensor(stf, dtype=torch.float32, requires_grad=False).repeat(src_num, 1)
        shot_ids = torch.tensor(np.arange(0, src_num), dtype=torch.int32)

        # get the receiver parameters
        rec_x = self.survey.rec_x  # refer to the index, not the coordinate
        rec_z = self.survey.rec_z  # refer to the index, not the coordinate

        # pad the material parameters and convert to torch tensor
        lam_pad = padding_np_array(lam, nPml, nPad)
        mu_pad  = padding_np_array(mu, nPml, nPad)
        rho_pad = padding_np_array(rho, nPml, nPad)
        th_lam_pad = torch.tensor(lam_pad, dtype=torch.float32, requires_grad=False)
        th_mu_pad  = torch.tensor(mu_pad,  dtype=torch.float32, requires_grad=False)
        th_rho_pad = torch.tensor(rho_pad, dtype=torch.float32, requires_grad=False)
    
        # write the para_file
        exp_name =  self.model.exp_name
        para_fname    = exp_name + '/para_file.json'
        survey_fname  = exp_name + '/survey_file.json'
        data_dir_name = exp_name + '/Data'
        paraGen(nz_pad, nx_pad, dz, dx, nt, dt, f0, nPml, nPad, para_fname, survey_fname, data_dir_name)
        surveyGen(src_z, src_x, rec_z, rec_x, survey_fname)
        print('The parameter file has been written to: ', para_fname)
        # run the forward propagation code
        outputs = fwi_ops.backward(th_lam_pad, th_mu_pad, th_rho_pad, th_stf, ngpu, shot_ids, para_fname)

        misfit    = outputs[0].detach().cpu().numpy()
        grad_lam  = outputs[1].detach().cpu().numpy()
        grad_mu   = outputs[2].detach().cpu().numpy()
        grad_rho0 = outputs[3].detach().cpu().numpy()
        grad_stf  = outputs[4].detach().cpu().numpy()

        grad_lam = grad_lam[nPml:-(nPad+nPml),nPml:-nPml]
        grad_mu  = grad_mu[nPml:-(nPad+nPml),nPml:-nPml]
        grad_rho0 = grad_rho0[nPml:-(nPad+nPml),nPml:-nPml]

        grad_vp  =  2 * rho * vp * grad_lam
        grad_vs  = -4 * rho * vs * grad_lam + 2 * rho * vs * grad_mu
        grad_rho = (vp**2 - 2 * vs**2) * grad_lam + vs **2 * grad_mu + grad_rho0

        return misfit, grad_vp, grad_vs, grad_rho, grad_stf
    
        # Finish TODO: 1. provide the model, source and receiver parameters to the forward propagation code
        # Finish TODO: 2. test the forward function, and benchmark the performance and accuracy
        # TODO: 3. implement the adjoint function, and perform the adjoint test
        # TODO: 4. simplify the forward and adjoint function, and make it more general for parameters
        # Finish TODO: 5. check gradient, and compare with the Python script version. Changing the source code to C++ could have some impact on the gradient calculation.


    # def apply_adjoint(self, survey, ngpu=1)