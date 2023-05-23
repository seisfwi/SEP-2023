#######################################################################
#  
# Analytical displacement solution for the 3D/2D wave equation due to a moment tensor 
# source in a homogeneous, isotropic elastic medium based on Aki & Richards (2002).  
#  
#  Author: Haipeng Li
#  Date  : 2023/01/07 
#  Email : haipeng@stanford.edu
#  Affiliation: SEP, Stanford University
#
#######################################################################


# import modules and deal with the case when joblib and tqdm are not installed
try:
    from tqdm import tqdm
except:
    print('tqdm is not installed. Install it by "pip install tqdm"')

try:
    from joblib import Parallel, delayed
except:
    print('joblib is not installed. Install it by "pip install joblib"')

import os
import numpy as np


def AnalyticalSolution(vp, vs, rho, x, y, z, tmin, tmax, dt, 
                    f0, M0, M, dim = '3D', comp = 'displacement', 
                    strike = None, dip = None, rake = None, verbose = True):

    ''' Analytical solution for the 3D/2D wave equation due to a moment tensor 
        source in a homogeneous, isotropic elastic medium.

    Parameters
    ----------
    vp : float
        P-wave velocity in m/s
    vs : float
        S-wave velocity in m/s
    rho : float
        Density in kg/m^3
    x : float
        x-coordinate of the receiver location in m
    y : float
        y-coordinate of the receiver location in m. Set to zero for 2D problems.
    z : float
        z-coordinate of the receiver location in m
    tmin : float
        Minimum time in s for computing the solution
    tmax : float
        Maximum time in s for computing the solution
    dt : float
        Time step in s for computing the solution
    f0 : float
        Dominant frequency of the source time function in Hz
    M0 : float
        Moment magnitude of the source
    M : 3x3 array
        Moment tensor of the source
            [ Mxx Mxy Mxz ]
        M = [ Myx Myy Myz ]
            [ Mzx Mzy Mzz ]
    dim : str (optional)
        Dimension of the problem, '2D' or '3D'
    comp : str (optional)
        Component of the solution, 'displacement', 'velocity', 'acceleration', or 'strain'
    strike : float (optional)
        Strike of the source in degrees
    dip : float (optional)
        Dip of the source in degrees
    rake : float (optional)
        Rake of the source in degrees
    verbose : bool (optional)
        Print the input parameters

    Notes:
    ------
    If the strike, dip, and rake are provided, the moment tensor will be overwritten.

    Returns
    -------
    solu : dict containing the time axis and solutions for the specified component and dimension: 
        For 3D problems: 
            Displacement: solu['t'], solu['Ux'], solu['Uy'], solu['Uz']
            Velocity: solu['t'], solu['Vx'], solu['Vy'], solu['Vz']
            Acceleration: solu['t'], solu['Ax'], solu['Ay'], solu['Az']
            Strain: solu['t'], solu['Exx'], solu['Eyy'], solu['Ezz'], solu['Exy'], solu['Exz'], solu['Eyz']
        For 2D problems:
            Displacement: solu['t'], solu['Ux'], solu['Uz']
            Velocity: solu['t'], solu['Vx'], solu['Vz']
            Acceleration: solu['t'], solu['Ax'], solu['Az']
            Strain: solu['t'], solu['Exx'], solu['Ezz'], solu['Exz']

    Example:
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from analytical_solution import AnalyticalSolution
    >>> U = AnalyticalSolution(3000.0, 1500.0, 2500.0, 200.0, 200.0, 200.0, 0.0, 
        1.0, 0.001, 10.0, 1.0e+16, np.eye(3), dim='3D', comp = 'displacement', verbose = True)
    >>> fig = plt.figure(figsize=(8,6))
    >>> comps = ['Ux', 'Uy', 'Uz']
    >>> fig = plt.figure(figsize=(8, 6))
    >>> for i, comp in enumerate(comps):
    >>>     plt.subplot(len(comps), 1, i+1)
    >>>     plt.plot(U['t'], U[comp])
    >>>     plt.xlabel('Time (s)')
    >>>     plt.ylabel('Amplitude')
    >>>     plt.legend([comp])
    >>> plt.show()
    '''

    # check the input parameters
    if dim not in ['2D', '3D']:
        raise ValueError('The dimension of the problem must be 2D or 3D.')

    if comp not in ['displacement', 'velocity', 'acceleration', 'strain']:
        raise ValueError('The component must be displacement, velocity, acceleration, or strain.')

   # print the input parameters if verbose is True
    if verbose:
        print('------------------------------------------------------------')
        print(' Input parameters for computing the analytical solution:')
        print('     dim  = {}'.format(dim))
        print('     comp = {}'.format(comp))
        print('     vp   = {:.2f} m/s'.format(vp))
        print('     vs   = {:.2f} m/s'.format(vs))
        print('     rho  = {:.2f} kg/m^3'.format(rho))
        print('     x    = {:.2f} m'.format(x))
        print('     y    = {:.2f} m'.format(y))
        print('     z    = {:.2f} m'.format(z))
        print('     tmin = {:.2f} s'.format(tmin))
        print('     tmax = {:.2f} s'.format(tmax))
        print('     dt   = {:.2e} s'.format(dt))
        print('     f0   = {:.2f} Hz'.format(f0))
        print('     M0   = {:.2e} Nm'.format(M0))
        print('     M    = {}'.format(M[0,:]))
        print('            {}'.format(M[1,:]))
        print('            {}'.format(M[2,:]))
        print('------------------------------------------------------------')

    # set the source mechanism
    if strike is not None and dip is not None and rake is not None:
        M = MomentTensor(strike, dip, rake)
        print('set the source mechanism based on strike = %f, dip = %f, rake = %f' % (strike, dip, rake))
    
    # # reset the y coordinate and M[1,:] and M[:.1] to zero for 2D problems
    # if dim == '2D':
    #     y = 0.0
    #     M[1,:] = 0.0
    #     M[:,1] = 0.0

    # initialize the output dictionary
    t = np.arange(tmin, tmax + dt, dt)
    solu = {}
    solu['t'] = t
    

    if dim == '3D':
        if comp == 'displacement':
            U = AnalyticalDisplacement3D(vp, vs, rho, x, y, z, t, f0, M0, M)
            solu['Ux'] = U[0,:]
            solu['Uy'] = U[1,:]
            solu['Uz'] = U[2,:]
        
        elif comp == 'velocity':
            U = AnalyticalDisplacement3D(vp, vs, rho, x, y, z, t, f0, M0, M)
            solu['Vx'] = np.gradient(U[0,:], dt)
            solu['Vy'] = np.gradient(U[1,:], dt)
            solu['Vz'] = np.gradient(U[2,:], dt)
        
        elif comp == 'acceleration':
            U = AnalyticalDisplacement3D(vp, vs, rho, x, y, z, t, f0, M0, M)
            solu['Ax'] = np.gradient(np.gradient(U[0,:], dt), dt)
            solu['Ay'] = np.gradient(np.gradient(U[1,:], dt), dt)
            solu['Az'] = np.gradient(np.gradient(U[2,:], dt), dt)
        
        elif comp == 'strain':
            # use the central difference to compute the derivatives
            # dx, dy, dz = 1, 1, 1
            dx, dy, dz = 0.5, 0.5, 0.5
            x1, x2 = x - dx, x + dx
            y1, y2 = y - dy, y + dy
            z1, z2 = z - dz, z + dz

            # compute the displacements at the grid points for computing the derivatives
            Ux1 = AnalyticalDisplacement3D(vp, vs, rho, x1, y, z, t, f0, M0, M)
            Ux2 = AnalyticalDisplacement3D(vp, vs, rho, x2, y, z, t, f0, M0, M)
            Uy1 = AnalyticalDisplacement3D(vp, vs, rho, x, y1, z, t, f0, M0, M)
            Uy2 = AnalyticalDisplacement3D(vp, vs, rho, x, y2, z, t, f0, M0, M)
            Uz1 = AnalyticalDisplacement3D(vp, vs, rho, x, y, z1, t, f0, M0, M)
            Uz2 = AnalyticalDisplacement3D(vp, vs, rho, x, y, z2, t, f0, M0, M)

            # compute the Exx, Eyy, and Ezz components of the strain tensor based on 2nd-order central difference
            solu['Exx'] = (Ux2[0,:] - Ux1[0,:]) / (2*dx)
            solu['Eyy'] = (Uy2[1,:] - Uy1[1,:]) / (2*dy)
            solu['Ezz'] = (Uz2[2,:] - Uz1[2,:]) / (2*dz)

            # compute the Exy, Exz and Eyz components of the strain tensor based on 2nd-order central difference
            solu['Exy'] = 0.5 * ((Ux2[1,:] - Ux1[1,:]) / (2*dx) + (Uy2[0,:] - Uy1[0,:]) / (2*dy))
            solu['Exz'] = 0.5 * ((Ux2[2,:] - Ux1[2,:]) / (2*dx) + (Uz2[0,:] - Uz1[0,:]) / (2*dz))
            solu['Eyz'] = 0.5 * ((Uy2[2,:] - Uy1[2,:]) / (2*dy) + (Uz2[1,:] - Uz1[1,:]) / (2*dz))

    elif dim == '2D':
        if comp == 'displacement':
            U = AnalyticalDisplacement2D(vp, vs, rho, x, z, t, f0, M0, M)
            solu['Ux'] = U[0,:]
            solu['Uz'] = U[2,:]
        
        elif comp == 'velocity':
            U = AnalyticalDisplacement2D(vp, vs, rho, x, z, t, f0, M0, M)
            solu['Vx'] = np.gradient(U[0,:], dt)
            solu['Vz'] = np.gradient(U[2,:], dt)
        
        elif comp == 'acceleration':
            U = AnalyticalDisplacement2D(vp, vs, rho, x, z, t, f0, M0, M)
            solu['Ax'] = np.gradient(np.gradient(U[0,:], dt), dt)
            solu['Az'] = np.gradient(np.gradient(U[2,:], dt), dt)
        
        elif comp == 'strain':
            # set the grid size and the coordinates of the grid points
            dx, dz = 10, 10
            x1, x2 = x - dx, x + dx
            z1, z2 = z - dz, z + dz

            # compute the displacements at the grid points for computing the derivatives
            Ux1 = AnalyticalDisplacement2D(vp, vs, rho, x1, z, t, f0, M0, M)
            Ux2 = AnalyticalDisplacement2D(vp, vs, rho, x2, z, t, f0, M0, M)
            Uz1 = AnalyticalDisplacement2D(vp, vs, rho, x, z1, t, f0, M0, M)
            Uz2 = AnalyticalDisplacement2D(vp, vs, rho, x, z2, t, f0, M0, M)

            # compute the Exx, Eyy, and Ezz components of the strain tensor based on 2nd-order central difference
            solu['Exx'] = (Ux2[0,:] - Ux1[0,:]) / (2*dx)
            solu['Ezz'] = (Uz2[2,:] - Uz1[2,:]) / (2*dz)
            solu['Exz'] = 0.5 * ((Ux2[2,:] - Ux1[2,:]) / (2*dx) + (Uz2[0,:] - Uz1[0,:]) / (2*dz))

    return solu


def AnalyticalDisplacement3D(vp, vs, rho, x, y, z, t, f0, M0, M):
    ''' Analytical displacement solution (3D) for a moment tensor source in a 
        homogeneous, isotropic elastic medium.
    '''

    # set the unit vector of the receiver location
    coord = np.array([x, y, z])
    r0 = np.linalg.norm(coord)
    r  = coord/r0

    # Compute the radiation patterns: near (AN), intermediate (AIP,AIS), and far (AFP,AFS) fields
    AN = np.zeros(3)
    AIP = np.zeros(3)
    AIS = np.zeros(3)
    AFP = np.zeros(3)
    AFS = np.zeros(3)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                # near fields (AN)
                AN[i]  += (15.0*r[i]*r[j]*r[k] - 3.0*delta(j,k)*r[i] - 3.0*delta(i,k)*r[j] - 3.0*delta(i,j)*r[k]) * M[j, k]
                
                # intermediate fields (AIP,AIS)
                AIP[i] +=   (6.0*r[i]*r[j]*r[k] - delta(j,k)*r[i] - delta(i,k)*r[j] - delta(i,j)*r[k]) * M[j, k]
                AIS[i] += - (6.0*r[i]*r[j]*r[k] - delta(j,k)*r[i] - delta(i,k)*r[j] - 2.0*delta(i,j)*r[k]) * M[j, k]

                # far fields (AFP,AFS)
                AFP[i] +=   (r[i]*r[j]*r[k]) * M[j, k]
                AFS[i] += - (r[i]*r[j]*r[k] - delta(i, j)*r[k]) * M[j, k]

    # Compute the scalar coefficients for the displacement solution
    CN  = (1/(4 * np.pi * rho)) 
    CIP = (1/(4 * np.pi * rho * vp**2))
    CIS = (1/(4 * np.pi * rho * vs**2))
    CFP = (1/(4 * np.pi * rho * vp**3))
    CFS = (1/(4 * np.pi * rho * vs**3))

    # Compute the source time function
    stf0 = STF0(t, r0/vp, r0/vs, f0, M0)
    stf1 = STF(t - r0/vp, f0, M0)
    stf2 = STF(t - r0/vs, f0, M0)
    stf3 = STF1(t - r0/vp, f0, M0)
    stf4 = STF1(t - r0/vs, f0, M0)

    U = np.zeros((3, len(t)))
    # Compute the displacement solution
    for i in range(3):
        U[i] =  CN  * (1/r0**4) *  AN[i] * stf0 + \
                CIP * (1/r0**2) * AIP[i] * stf1 + \
                CIS * (1/r0**2) * AIS[i] * stf2 + \
                CFP * (1/r0   ) * AFP[i] * stf3 + \
                CFS * (1/r0   ) * AFS[i] * stf4

    return U


def AnalyticalDisplacement2D(vp, vs, rho, x, z, t, f0, M0, M):
    ''' Analytical displacement solution (2D) for a moment tensor source in a 
        homogeneous, isotropic elastic medium. The 2D displacement solution is 
        obtained by integrating the 3D displacement solution along the y-axis assuming a line source.
        
        U_2D(x,z,t) = Int_{y_min}^{y_max} U_3D(x,y,z,t) dy

    '''

    # time vector
    nt = len(t)
    tmax = t[-1]

    # set the spatial integration limits over the y-axis
    dy     = 1.0/(4.*f0) * vs / 5.0
    ny_min = int(- vp * tmax * 1.5/dy) - 1
    ny_max = int(  vp * tmax * 1.5/dy) + 1
    print(f"Calculating 2D solution by integrating 3D solution from {ny_max-ny_min} receivers ...")

    # perform the spatial integration over the y-axis
    U = np.zeros((3, nt))

    # parallel computation
    njobs = os.cpu_count() // 2
    njobs = min(njobs, ny_max - ny_min)

    results = Parallel(n_jobs = njobs)(delayed(AnalyticalDisplacement3D)(
        vp, vs, rho, x, - (rec - 1.0) * dy, z, t, f0, M0, M) 
        for rec in tqdm(range(ny_min, ny_max)))
    
    # collect the results
    for U_3D in results:
        U += U_3D * dy

    # for rec in tqdm(range(ny_min, ny_max)):
    #     # set the new y-coordinate for integration
    #     y_new = 0.0 - (rec - 1.0) * dy

    #     # compute the displacement solution at the new y-coordinate
    #     U_3D = AnalyticalDisplacement3D(vp, vs, rho, x, y_new, z, t, f0, M0, M)
        
    #     # integrate over the y-axis
    #     U += U_3D * dy


    return U


def delta(i, j):
    ''' Kronecker delta function.
    '''
    if i == j:
        return 1
    else:
        return 0


def STF(t, f0, M0):
    ''' Ricker Source time function.

    Parameters
    ----------
    t : array_like
        Time vector.
    f0 : float
        Dominant frequency of the source time function.
    M0 : float
        Moment magnitude of the source.

    Returns
    -------
    stf : array_like
        Source time function.
    
    Notes
    -----
    The default time delay is 1.2/f0.
    '''
    # set the time delay
    t0 = 1.2 / f0 

    # compute the source time function
    stf = np.zeros_like(t)
    for i in range(len(t)):
        stf[i] = (1.0-2.0*np.pi**2*f0**2*(t[i]-t0)**2)*np.exp(-np.pi**2*f0**2*(t[i]-t0)**2) 

    return M0 * stf


def STF1(t, f0, M0):
    ''' First derivative of the Ricker Source time function.

    Parameters
    ----------
    t : array_like
        Time vector.
    f0 : float
        Dominant frequency of the source time function.
    M0 : float
        Moment magnitude of the source.

    Returns
    -------
    stf : array_like
        First derivative of the Ricker Source time function.
    
    Notes
    -----
    The default time delay is 1.2/f0.
    '''
    # set the time delay
    t0 = 1.2 / f0 

    # compute the source time function
    stf = np.zeros_like(t)
    stf = (-2.0) * (np.pi * f0)**2 * (t - t0) * (3.0 - 2.0 *(np.pi * f0 * (t - t0))** 2) * np.exp(-(np.pi*f0*(t - t0))**2)

    return M0 * stf


def STF2(t, f0, M0):
    ''' Second derivative of the Ricker Source time function.

    Parameters
    ----------
    t : array_like
        Time vector.
    f0 : float
        Dominant frequency of the source time function.
    M0 : float
        Moment magnitude of the source.

    Returns
    -------
    stf : array_like
        Second derivative of the Ricker Source time function.
    
    Notes
    -----
    The default time delay is 1.2/f0.
    '''
    # set the time delay
    t0 = 1.2 / f0

    # compute the source time function
    stf = np.zeros_like(t)
    stf = (-6. * (np.pi * f0)**2 + 24. * (np.pi * f0)**4 * (t - t0) **2 - 8. * (np.pi * f0)**6 * (t - t0) **4) * np.exp(-(np.pi*f0*(t - t0))**2)

    return M0 * stf


def STF0(t, tmin, tmax, f0, M0):
    ''' The integral of the Ricker Source time function with t.

    Parameters
    ----------
    t : array_like
        Time vector.
    tmin : float
        The minimum time of the integration.
    tmax : float
        The maximum time of the integration.
    f0 : float
        Dominant frequency of the source time function.
    M0 : float
        Moment magnitude of the source.
    
    Returns
    -------
    stf : array_like
        The integral of the Ricker Source time function with t.

    Notes
    -----
    The default time delay is 1.2/f0.
    '''
    # set the time delay
    t0 = 1.2 / f0

    # compute the source time function
    dt = t[1] - t[0]
    stf = np.zeros_like(t)

    # set the range of integration
    tau = np.arange(tmin, tmax, dt)

    # compute the integral
    for i in range(len(t)):
        for itau in range(len(tau)):
            stf[i] += (1.0-2.0*np.pi**2*f0**2*(t[i] - tau[itau]-t0)**2)*np.exp(-np.pi**2*f0**2*(t[i] - tau[itau]-t0)**2) * tau[itau] * dt
    
    return M0 * stf 


def MomentTensor(strike, dip, rake):
    ''' Compute the components of a moment tensor for a fault mechanism

    Parameters
    ----------
    strike : float
        Strike of the fault plane (degrees)
    dip : float
        Dip of the fault plane (degrees)
    rake : float
        Rake of the fault plane (degrees)
    
    Returns
    ------- 
    CM_FD : array
        Moment tensor components in the fault plane coordinate system
    '''

    pi180 = np.pi/180
    CS = np.cos(strike * pi180)  # calculate each moment component
    SS = np.sin(strike * pi180)
    CDI = np.cos(dip * pi180)
    SDI = np.sin(dip * pi180)
    CR = np.cos(rake * pi180)
    SR = np.sin(rake * pi180)
    AS1 = CR * CS + SR * CDI * SS
    AS2 = CR * SS - SR * CDI * CS
    AS3 = -SR * SDI
    AN1 = -SDI * SS
    AN2 = SDI * CS
    AN3 = -CDI
    CM11 = 2. * AS1 * AN1  # change to the normal convention by eliminating the minus sign
    CM22 = 2. * AS2 * AN2
    CM33 = 2. * AS3 * AN3
    CM12 = (AS1 * AN2  +AS2 * AN1)
    CM13 = (AS1 * AN3 + AS3 * AN1)
    CM23 = (AS2 * AN3 + AS3 * AN2)

    CM_FD = np.zeros((3,3))
    CM_FD[0,0] = CM11
    CM_FD[0,1] = CM12
    CM_FD[0,2] = CM13
    CM_FD[1,0] = CM12
    CM_FD[1,1] = CM22
    CM_FD[1,2] = CM23
    CM_FD[2,0] = CM13
    CM_FD[2,1] = CM23
    CM_FD[2,2] = CM33

    return CM_FD
