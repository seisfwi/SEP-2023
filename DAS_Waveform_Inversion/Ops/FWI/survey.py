import numpy as np

class Model(object):
    ''' The class of defining the model parameters
    '''

    def __init__(self, nx, nz, dx, dz, nt, dt, nPml, vp, vs, rho, exp_name):
        ''' Initialize the model parameters
        '''

        # get the model parameters
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.nt = nt
        self.dt = dt
        self.nPml = nPml
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.exp_name = exp_name


class Survey(object):
    ''' The class of defining the source parameters
    '''

    def __init__(self, f0, src_x, src_z, rec_x, rec_z):
        ''' Initialize the source parameters
        '''
        # get the source parameters
        self.f0 = f0
        self.src_x = src_x
        self.src_z = src_z

        self.rec_x = rec_x
        self.rec_z = rec_z
