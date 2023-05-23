#######################################################################
#  
# 2D elastic wave equation solver with time-domain staggered grid finite difference method 
# (4th order in space, 2nd order in time). All boundaries are simple absorbing 
# boundaries. No free surface! Simulation is parallelized with multiprocessing over shots.
# The source is set an explosive source added to the pressure field. The receivers
# are set as hydrophones (pressure) and geophones (particle velocity) and DAS 
# (avereged strain rate over a guagle length). The calculation of DAS is based on
#
# Eaid, M. V., Keating, S. D., & Innanen, K. A. (2020). Multiparameter seismic 
# elastic full-waveform inversion with combined geophone and shaped fiber-optic 
# cable data. Geophysics, 85(6), R537-R552.
# 
#
#  Author: Haipeng Li
#  Date  : 2023/04/25 
#  Email : haipeng@stanford.edu
#  Affiliation: SEP, Stanford University
#
#######################################################################


import os
from multiprocessing import Pool
import numpy as np
from numba import jit
import matplotlib.animation as animation
import matplotlib.pyplot as plt


class elasticSolver():

    def __init__(self, nx:int, nz:int, ndamp:int, dx:float, dz:float, dt:float, nt:int, f0:float, 
                 vp:np.ndarray, vs:np.ndarray, rho:np.ndarray, src_coord:np.ndarray, 
                 das_coord:np.ndarray, geo_coord:np.ndarray, das_sensitivity:np.ndarray):

        self.nx = nx + 2 * ndamp
        self.nz = nz + 2 * ndamp
        self.ndamp = ndamp
        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.nt = nt
        self.f0 = f0
        self.vp = np.pad(vp, ndamp, 'edge')
        self.vs = np.pad(vs, ndamp, 'edge')
        self.rho = np.pad(rho, ndamp, 'edge')
        self.src_coord = src_coord
        self.das_coord = das_coord
        self.geo_coord = geo_coord
        self.das_sensitivity = das_sensitivity

        # set derived parameters
        self.x = np.arange(0, nx * dx, dx)
        self.z = np.arange(0, nz * dz, dz)
        self.t = np.arange(0, nt * dt, dt)
        self.save_step = 10

        # Lame parameters
        self.mu = self.rho * self.vs**2
        self.lam = self.rho * self.vp**2 - 2 * self.mu

        # find the nearest grid for source and receiver positions
        self.src_grid = np.array([np.round(self.src_coord[:, 0]/dx).astype(int), np.round(self.src_coord[:, 1]/dz).astype(int)])
        self.das_grid = np.array([np.round(self.das_coord[:, 0]/dx).astype(int), np.round(self.das_coord[:, 1]/dz).astype(int)])
        self.geo_grid = np.array([np.round(self.geo_coord[:, 0]/dx).astype(int), np.round(self.geo_coord[:, 1]/dz).astype(int)])

        # set number of receivers and sources
        self.das_num = self.das_grid.shape[1]
        self.geo_num = self.geo_grid.shape[1]
        self.src_num = self.src_grid.shape[1]

        # set the damping parameters
        self.damp = np.ones((self.nx, self.nz))
        for i in range(ndamp):
            self.damp[   i,    :] *= np.sin(np.pi/2 * i/ndamp)**2
            self.damp[-i-1,    :] *= np.sin(np.pi/2 * i/ndamp)**2
            self.damp[   :,    i] *= np.sin(np.pi/2 * i/ndamp)**2
            self.damp[   :, -i-1] *= np.sin(np.pi/2 * i/ndamp)**2

        # extend the source and receivers
        self.src_grid += ndamp
        self.das_grid += ndamp
        self.geo_grid += ndamp

        # source wavelet
        f0 = self.f0
        t = self.t
        self.stf = (1.0-2.0*np.pi**2*f0**2*(t-1.2/f0)**2)*np.exp(-np.pi**2*f0**2*(t-1.2/f0)**2)

        # check the input parameters
        self.__check__()

        
    def set_model(self, vp, vs, rho):
        
        nx = self.nx - 2*self.ndamp
        nz = self.nz - 2*self.ndamp
        
        if vp.shape != (nx, nz) or vs.shape != (nx, nz) or rho.shape != (nx, nz):
            raise ValueError("wrong size")

        self.vp = np.pad(vp, self.ndamp, 'edge')
        self.vs = np.pad(vs, self.ndamp, 'edge')
        self.rho = np.pad(rho, self.ndamp, 'edge')
        
        # Lame parameters
        self.mu = self.rho * self.vs**2
        self.lam = self.rho * self.vp**2 - 2 * self.mu


    def __check__(self):
        ''' Check the input parameters
        '''

        # check the shape of vp, vs and rho
        if self.vp.shape  != (self.nx, self.nz) or \
           self.vs.shape  != (self.nx, self.nz) or \
           self.rho.shape != (self.nx, self.nz):
            raise ValueError("The shape of vp, vs and rho should be (nx, nz)")

        if self.src_coord.shape[1] != 2:
            raise ValueError("The shape of src_coord should be (nsrc, 2)")
        
        if self.das_coord.shape[1] != 2:
            raise ValueError("The shape of das_coord should be (nrec, 2)")

        if self.geo_coord.shape[1] != 2:
            raise ValueError("The shape of geo_coord should be (nrec, 2)")

        # check the source coordinates
        if self.src_grid[0, :].max() > self.nx or \
            self.src_grid[1, :].max() > self.nz or \
              self.src_grid[0, :].min() < 0 or \
                self.src_grid[1, :].min() < 0:
            raise ValueError("The source coordinates should be within the model")

        # check the das receiver coordinates
        if self.das_grid[0, :].max() > self.nx or \
              self.das_grid[1, :].max() > self.nz or \
                self.das_grid[0, :].min() < 0 or \
                    self.das_grid[1, :].min() < 0:
            raise ValueError("The das receiver coordinates should be within the model")

        # check the geo receiver coordinates
        if self.geo_grid[0, :].max() > self.nx or \
              self.geo_grid[1, :].max() > self.nz or \
                  self.geo_grid[0, :].min() < 0 or \
                      self.geo_grid[1, :].min() < 0:
            raise ValueError("The geo receiver coordinates should be within the model")

        if self.das_sensitivity.shape != (self.das_grid.shape[1], 6):
            raise ValueError("The shape of das_sensitivity should be (nchannel, 6) with exx, exy, exz, eyy, eyz, and ezz")


    def forward(self, save_wavefield = False):
        ''' Forward modeling
        '''

        # collect the results
        solus = []

        core_num = np.min([self.src_num, os.cpu_count()])

        # initialize the Pool object
        pool = Pool(core_num)

        # apply to all sources
        results = [pool.apply_async(self.forward_it, (isrc, save_wavefield, ))
                   for isrc in range(self.src_num)]

        # close the pool and wait for the work to finish
        pool.close()

        # collect the results
        for result in results:
            solus.append(result.get())

        # block at this line until all processes are done
        pool.join()

        return solus


    def forward_it(self, isrc, save_wavefield):
        ''' Forward modeling for a single shot
        '''

        # model parameters
        nx = self.nx
        nz = self.nz
        nt = self.nt
        dt = self.dt
        dx = self.dx
        dz = self.dz
        lam = self.lam
        mu = self.mu
        rho = self.rho
        stf = self.stf
        t = self.t
        ndamp = self.ndamp
        damp = self.damp
        save_step = self.save_step

        # source and receiver parameters
        das_num = self.das_num
        geo_num = self.geo_num
        src_grid = self.src_grid[:, isrc]
        das_grid = self.das_grid
        geo_grid = self.geo_grid

        # weight for DAS directional sensitivity
        das_sensitivity = self.das_sensitivity

        # initialize the wavefield
        vx  = np.zeros((nx, nz))
        vz  = np.zeros((nx, nz))
        sxx = np.zeros((nx, nz))
        szz = np.zeros((nx, nz))
        sxz = np.zeros((nx, nz))

        # initialize the seismograms
        geoPr  = np.zeros((geo_num, nt))
        geoVx  = np.zeros((geo_num, nt))
        geoVz  = np.zeros((geo_num, nt))
        dasExx = np.zeros((das_num, nt))
        dasEzz = np.zeros((das_num, nt))
        dasExz = np.zeros((das_num, nt))
        dasEtt = np.zeros((das_num, nt))

        # Initialize wavefield
        if save_wavefield:
            save_num  = nt // save_step + 1
            sxx_store = np.zeros((save_num, nx-2*ndamp, nz-2*ndamp))
            szz_store = np.zeros((save_num, nx-2*ndamp, nz-2*ndamp))
            vx_store = np.zeros((save_num, nx-2*ndamp, nz-2*ndamp))
            vz_store = np.zeros((save_num, nx-2*ndamp, nz-2*ndamp))


        # Time-stepping loop
        for it in range(nt):

            # Update velocities
            vx, vz, _, _, _, _ = update_velocity(vx, vz, sxx, szz, sxz, nx, nz, dx, dz, dt, rho)

            # Apply damping
            vx *= damp
            vz *= damp

            # Update stress
            sxx, szz, sxz,  _, _, _, _ = update_stress(vx, vz, sxx, szz, sxz, nx, nz, dx, dz, dt, lam, mu)

            # Apply damping
            sxx *= damp
            szz *= damp
            sxz *= damp

            # Add explosive source
            sxx[src_grid[0], src_grid[1]] += stf[it] * dt / 2.0
            szz[src_grid[0], src_grid[1]] += stf[it] * dt / 2.0

            # Record seismogram for GEO
            for i in range(geo_num):
                geoVx[i, it] = vx[geo_grid[0, i], geo_grid[1, i]]
                geoVz[i, it] = vz[geo_grid[0, i], geo_grid[1, i]]
                geoPr[i, it] = (sxx[geo_grid[0, i], geo_grid[1, i]] + szz[geo_grid[0, i], geo_grid[1, i]]) * 0.5
                
            # Record seismogram for DAS
            for i in range(das_num):
                dasExx[i, it] = (vx[das_grid[0, i], das_grid[1, i]] - vx[das_grid[0, i] - 1, das_grid[1, i]]) / dx
                dasEzz[i, it] = (vz[das_grid[0, i], das_grid[1, i]] - vz[das_grid[0, i], das_grid[1, i] - 1]) / dz
                dasExz[i, it] = 0.5 * ((vx[das_grid[0, i], das_grid[1, i] + 1] - vx[das_grid[0, i], das_grid[1, i]]) / dz + 
                                       (vz[das_grid[0, i] + 1, das_grid[1, i]] - vz[das_grid[0, i], das_grid[1, i]]) / dx)

                # average the Exx, Ezz, Exz to get Ett for DAS. Note here the gauge length is not divided, so the amplitude is not correct
                dasEtt[i, it] = das_sensitivity[i, 0] * dasExx[i, it] + das_sensitivity[i, 3] * dasEzz[i, it] + das_sensitivity[i, 1] * dasExz[i, it]


            if save_wavefield and it % save_step == 0:
                isnap = int(it / save_step)
                sxx_store[isnap, :, :] = sxx[ndamp:-ndamp, ndamp:-ndamp]
                szz_store[isnap, :, :] = szz[ndamp:-ndamp, ndamp:-ndamp]
                vx_store[isnap, :, :]  = vx[ndamp:-ndamp, ndamp:-ndamp]
                vz_store[isnap, :, :]  = vz[ndamp:-ndamp, ndamp:-ndamp]


        # Save the seismogram
        solu = {}
        solu['t']   = t
        solu['vx']  = geoVx
        solu['vz']  = geoVz
        solu['pr']  = geoPr
        solu['ett'] = dasEtt
        solu['exx'] = dasExx
        solu['ezz'] = dasEzz
        solu['exz'] = dasExz

        # Save the wavefield if needed
        if save_wavefield:
            solu['sxx_wavefield'] = sxx_store
            solu['szz_wavefield'] = szz_store
            solu['vx_wavefield'] = vx_store
            solu['vz_wavefield'] = vz_store

        return solu


### 2D elastic wave equation solver using the velocity-stress formulation ####

@jit(nopython=True)
def update_velocity(vx, vz, sxx, szz, sxz, nx, nz, dx, dz, dt, rho):
    ''' Update velocities 
    '''
    
    c1 = 9.0 / 8.0
    c2 = 1.0 / 24.0

    sxx_x_store = np.zeros_like(vx)
    sxz_x_store = np.zeros_like(vx)
    sxz_z_store = np.zeros_like(vx)
    szz_z_store = np.zeros_like(vx)

    for i in range(2, nx-2):
        for j in range(2, nz-2):

            # Effective density
            rhox = 0.5 * (rho[i, j] + rho[i+1, j])
            rhoz = 0.5 * (rho[i, j] + rho[i, j+1])

            # Update velocities
            szz_z = (c1 * (szz[i, j+1] - szz[i,j]) - c2 * (szz[i, j+2] - szz[i,j-1])) / dz
            sxz_x = (c1 * (sxz[i, j] - sxz[i-1,j]) - c2 * (sxz[i+1, j] - sxz[i-2,j])) / dx
            sxz_z = (c1 * (sxz[i, j] - sxz[i,j-1]) - c2 * (sxz[i, j+1] - sxz[i,j-2])) / dz
            sxx_x = (c1 * (sxx[i+1, j] - sxx[i,j]) - c2 * (sxx[i+2, j] - sxx[i-1,j])) / dx

            vx[i,j] +=  (sxz_z + sxx_x) * dt / rhoz
            vz[i,j] +=  (szz_z + sxz_x) * dt / rhox

            # Store for adjoint
            sxx_x_store[i,j] = sxx_x
            sxz_x_store[i,j] = sxz_x
            sxz_z_store[i,j] = sxz_z
            szz_z_store[i,j] = szz_z

    return vx, vz, sxx_x_store, sxz_x_store, sxz_z_store, szz_z_store


@jit(nopython=True)
def update_stress(vx, vz, sxx, szz, sxz, nx, nz, dx, dz, dt, lam, mu):
    ''' Update stresses
    '''

    c1 = 9.0 / 8.0
    c2 = 1.0 / 24.0

    vxx_store = np.zeros_like(vx)
    vzx_store = np.zeros_like(vx)
    vzz_store = np.zeros_like(vx)
    vxz_store = np.zeros_like(vx)

    for i in range(2, nx-2):
        for j in range(2, nz-2):

            # Effective shear modulus
            if mu[i, j] != 0.0 and mu[i+1, j] != 0.0 and mu[i, j+1] != 0.0 and mu[i+1, j+1] != 0.0:
                muxz = 4.0 / (1/mu[i,j] + 1/mu[i+1,j] + 1/mu[i,j+1] + 1/mu[i+1,j+1]) 
            else: 
                muxz = 0.0

            # Update stresses
            vzz = (c1 * (vz[i,j] - vz[i,j-1]) - c2 * (vz[i,j+1] - vz[i,j-2])) / dz
            vxx = (c1 * (vx[i,j] - vx[i-1,j]) - c2 * (vx[i+1,j] - vx[i-2,j])) / dx
            vxz = (c1 * (vx[i,j+1] - vx[i,j]) - c2 * (vx[i,j+2] - vx[i,j-1])) / dz
            vzx = (c1 * (vz[i+1,j] - vz[i,j]) - c2 * (vz[i+2,j] - vz[i-1,j])) / dx

            szz[i, j] += ((lam[i, j] + 2*mu[i, j])*vzz + lam[i, j]*vxx) * dt
            sxx[i, j] += (lam[i, j]*vzz + (lam[i, j] + 2*mu[i, j])*vxx) * dt
            sxz[i, j] += (vxz + vzx) * muxz * dt

            # Store for adjoint
            vxx_store[i,j] = vxx
            vzx_store[i,j] = vzx
            vzz_store[i,j] = vzz
            vxz_store[i,j] = vxz

    return sxx, szz, sxz, vxx_store, vzx_store, vzz_store, vxz_store


def dampProfile(nx, ny, ndamp):
    ''' Set up damping profile
    '''

    nx_damp = nx + 2*ndamp
    ny_damp = ny + 2*ndamp
    damp = np.ones((nx_damp, ny_damp))

    for i in range(ndamp):
        damp[i, :] *= np.sin(np.pi/2 * i/ndamp)**2
        damp[-i-1, :] *=  np.sin(np.pi/2 * i/ndamp)**2
        damp[:, i] *=  np.sin(np.pi/2 * i/ndamp)**2
        damp[:, -i-1] *= np.sin(np.pi/2 * i/ndamp)**2

    return damp



def plot_wavefield(wavefield):
    ''' Plot wavefield
    '''

    fig = plt.figure()
    ims = []

    for i in range(wavefield.shape[0]):
        caxis = wavefield.max() * 0.2
        im = plt.imshow(wavefield[i].T, vmin = -caxis, vmax = caxis, aspect = 1, cmap='seismic', animated=False)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat=True, repeat_delay=0)
    plt.close()

    return ani