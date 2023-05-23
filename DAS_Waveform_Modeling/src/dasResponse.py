#######################################################################
#  
# DAS response calculation using the quaduture method.
#  
#  Author: Haipeng Li
#  Date  : 2023/01/07 
#  Email : haipeng@stanford.edu
#  Affiliation: SEP, Stanford University
#
#######################################################################

import numpy as np
from analyticalSolution import AnalyticalSolution

def das_response(vp, vs, rho, GL, cable, nquad, srcx, srcy, srcz, tmin, tmax, dt, f0, M0, M):
    ''' Calculate the DAS response using the quaduture method.
    '''

    # Check the number of cable points
    npts = 21

    if cable.shape[0] != npts:
        raise ValueError('The number of cable points should be 21.')

    # Check the number of quadrature points
    if nquad == 1:
        points = np.array([10])                       # 1 points quaduture
        nq = 21
    elif nquad == 3:
        points = np.array([3, 10, 17])                # 3 points quaduture, every 7 points
        nq = 7
    elif nquad == 7:
        points = np.array([1, 4, 7, 10, 13, 16, 19])  # 5 points quaduture, every 5 points
        nq = 3
    elif nquad == 21:
        points = np.arange(0, 21, 1)                  # 21 points quaduture, every 1 points
        nq = 1
    else:
        raise ValueError('The number of quadrature points should be 1, 3, 7 or 21.')

    # Get the cable positions for the quadrature
    x = cable[points, 0] - srcx
    y = cable[points, 1] - srcy
    z = cable[points, 2] - srcz

    # Set up the time axis
    t = np.arange(tmin, tmax + dt, dt)
    nt = t.shape[0]

    # Set up the DAS response for each quadrature point
    das_point = np.zeros((npts, nt))

    for i in range(nquad):

        # Calculate analytical strain
        U = AnalyticalSolution(vp, vs, rho, x[i], y[i], z[i], tmin, tmax, dt, f0, M0, M, dim='3D', comp = 'strain', verbose=False)
        
        # Get DAS parameters
        n1 = points[i] - nq // 2
        n2 = points[i] + nq // 2 + 1

        # Set the DAS quadrature points using the surrounding cable points
        for j in range(n1, n2):
            das_point[j, :] = cable[j, 3] * U['Exx'] + \
                              cable[j, 4] * U['Exy'] + \
                              cable[j, 5] * U['Exz'] + \
                              cable[j, 6] * U['Eyy'] + \
                              cable[j, 7] * U['Eyz'] + \
                              cable[j, 8] * U['Ezz']

        # Trapezoidal integration
        s = np.linspace(-GL/2, GL/2, npts)
        das = np.trapz(das_point, s, axis = 0) / GL

    return das