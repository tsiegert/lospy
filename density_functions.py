import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan, arctan2, pi, exp, sqrt, cosh, fabs, zeros

from math import erf

import numba as nb
from numba import prange, float64, bool_

@nb.njit(float64[:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:],
                      float64,
                      float64,float64,float64,
                      float64,float64,float64,
                      float64,float64,float64,float64,
                      bool_),fastmath=True,parallel=True)
def Gaussian_Torus_Total(s: np.ndarray, b: np.ndarray, l: np.ndarray,
                         ds: float,
                         x0: float, y0: float, z0: float,
                         xT: float, yT: float, zT: float,
                         RT: float, Rt: float, sigmaT: float, sigmat: float,
                         luminosity_flag: bool):

    deg2rad = pi/180.

    # los vector
    x = x0 + s*cos(deg2rad*l)*cos(deg2rad*b)
    y = y0 + s*sin(deg2rad*l)*cos(deg2rad*b)
    z = z0 + s*sin(deg2rad*b)

    Rxy = sqrt((x-xT)**2 + (y-yT)**2)

    # complete torus: (here sigma_t = sigma_T)
    torus_slices = exp(-0.5 * ( (Rxy - RT)**2/sigmaT**2 + ((z-zT) - Rt)**2/sigmat**2 ) )

    if luminosity_flag == False:
    
        # for map integration
        torus_map_slices = torus_slices

    else:

        # for luminosity integration
        torus_map_slices = torus_slices * s**2

    # los
    val = np.sum(torus_map_slices*ds,axis=0)

    return(val)





@nb.njit(float64[:,:,:](float64[:,:,:],float64[:,:,:],
                      float64,float64,
                      float64,float64,
                      float64),fastmath=True,parallel=True)
def Gaussian_Ellipse(X: np.ndarray, Y: np.ndarray,
                     x0: float, y0: float,
                     sx: float, sy: float,
                     theta: float):
    
    a = 0.5*((np.cos(theta)/sx)**2 + (np.sin(theta)/sy)**2)
    b = 0.25*np.sin(2*theta)*(-1/sx**2 + 1/sy**2)
    c = 0.5*((np.sin(theta)/sx)**2 + (np.cos(theta)/sy)**2)
    
    return(np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2)))





@nb.njit(float64[:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:],
                      float64,
                      float64,float64,float64,
                      float64,float64,
                      float64,float64,float64,float64,
                      float64,float64,
                      float64,float64,
                      float64,float64,
                      float64,float64,
                      bool_),fastmath=True,parallel=True)
def Trojans(s: np.ndarray, b: np.ndarray, l: np.ndarray,
            ds: float,
            x0: float, y0: float, z0: float,
            RT: float, sigmaT: float,
            x4: float, y4: float, x5: float, y5: float,
            tx: float, ty: float,
            theta4: float, theta5: float,
            rhoz1: float, tz1: float,
            rhoz2: float, tz2: float,
            luminosity_flag: bool):

    deg2rad = pi/180.

    # los vector
    x = x0 + s*cos(deg2rad*l)*cos(deg2rad*b)
    y = y0 + s*sin(deg2rad*l)*cos(deg2rad*b)
    z = z0 + s*sin(deg2rad*b)

    # radial coordinate
    Rxyz = sqrt(x**2 + y**2 + z**2)

    # trojan slices: assuming suppresion at lagrange points using a asymmetric 2D Gaussian whose scale
    # relates to the trojan distribution of the planet

    # fitting with one Gaussian in z direction (gives wrong z distribution)
    #trojan_slices = np.exp( -0.5 * ( (Rxyz - RT)**2/sigmaT**2)) * \
    #    ( _Gaussian_Ellipse(x,y,x4,y4,tx,ty,theta4) + \
    #      _Gaussian_Ellipse(x,y,x5,y5,tx,ty,theta5)) * \
    #      np.exp( -0.5 * (z**2)/tz**2)

    # fitting with two Gaussians in z direction (better description of z distribution for Jovian Trojans)
    # maybe only valid for Jovian trojans
    trojan_slices = np.exp( -0.5 * ( (Rxyz - RT)**2/sigmaT**2)) * \
        ( Gaussian_Ellipse(x,y,x4,y4,tx,ty,theta4) + \
          Gaussian_Ellipse(x,y,x5,y5,tx,ty,theta5)) * \
          ( rhoz1 * np.exp( -0.5 * (z**2)/tz1**2) + \
            rhoz2 * np.exp( -0.5 * (z**2)/tz2**2) )

    if luminosity_flag == False:
    
        # for map integration
        trojan_map_slices = trojan_slices

    else:

        # for luminosity integration
        trojan_map_slices = trojan_slices * s**2

    # los
    val = np.sum(trojan_map_slices*ds,axis=0)

    return(val)




@nb.njit(float64[:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:],
                      float64,
                      float64,float64,float64,
                      float64,float64,float64,
                      float64,float64,
                      bool_),fastmath=True,parallel=True)
def Gaussian_Shell(s: np.ndarray, b: np.ndarray, l: np.ndarray,
                   ds: float,
                   x0: float, y0: float, z0: float,
                   xT: float, yT: float, zT: float,
                   RT: float, sigmaT: float,
                   luminosity_flag: bool):

    deg2rad = pi/180.

    # los vector
    x = x0 + s*cos(deg2rad*l)*cos(deg2rad*b)
    y = y0 + s*sin(deg2rad*l)*cos(deg2rad*b)
    z = z0 + s*sin(deg2rad*b)

    Rxyz = sqrt((x-xT)**2 + (y-yT)**2 + (z-zT)**2)

    # complete shell
    shell_slices = exp(-1/(2*sigmaT**2)*((Rxyz - RT)**2))

    if luminosity_flag == False:
    
        # for map integration
        shell_map_slices = shell_slices

    else:

        # for luminosity integration
        shell_map_slices = shell_slices * s**2
    
    # los
    val = np.sum(shell_map_slices*ds,axis=0)
    
    return(val)



@nb.njit(float64[:,:](float64[:,:],float64[:,:],
                      float64,float64,float64,
                      float64,float64,float64,
                      float64,float64),parallel=True)
def Solid_Torus(phi: np.ndarray,theta: np.ndarray,
                x0: float, y0: float, z0: float,
                xT: float, yT: float, zT: float,
                RT: float, Rt: float):
    Delta_x = x0-xT
    Delta_y = y0-yT
    Delta_z = z0-zT
    Delta_r = np.sqrt(Delta_x**2 + Delta_y**2 + Delta_z**2)
    p = Delta_x*np.cos(theta)*np.cos(phi) + Delta_y*np.cos(theta)*np.sin(phi) + Delta_z*np.sin(theta)
    q = np.cos(theta)*(Delta_x*np.cos(phi)+Delta_y*np.sin(phi))
    xi = np.sqrt(Delta_r**2 + RT**2 - Rt**2)
    nu = (4*RT**2*Delta_x**2 + 4*RT**2*Delta_y**2)**(0.25)
    A = 1.
    B = 4*p
    C = 4*p**2 + 2*xi**2 - 4*RT**2*np.cos(theta)**2
    D = 4*p*xi**2 - 8*RT**2*q
    E = (xi**4 - nu**4)

    val = np.zeros_like(phi)
    
    for i in prange(phi.shape[0]):
        for j in prange(phi.shape[1]):
        
            tmp = np.array([A,B[i,j],C[i,j],D[i,j],E], dtype=np.complex64)
        
            roots = np.flip(np.sort(np.real(np.roots(tmp))))
        
            val[i,j] = roots[0] - roots[1]
    
    return(val)




@nb.njit(float64[:,:](float64[:,:],float64[:,:],
                      float64,float64,float64,
                      float64,float64,float64,
                      float64,float64,
                      bool_),parallel=True)
def Spherical_Shell_Solid(phi: np.ndarray, theta: np.ndarray,
                          x0: float, y0: float, z0: float,
                          xT: float, yT: float, zT: float,
                          Ri: float, Ro: float,
                          luminosity_flag: bool):

    print(x0,y0,z0,xT,yT,zT)
    
    s0 = np.sqrt((x0-xT)**2 + (y0-yT)**2 + (z0-zT)**2)

    p0 = (x0-xT)*np.cos(theta)*np.cos(phi) + (y0-yT)*np.cos(theta)*np.sin(phi) + (z0-zT)*np.sin(theta)

    #print(Ri,Ro,s0)
    
    if (s0 > Ro):

        print('observer outside sphere')
        
        smin_i = -p0 - np.sqrt(p0**2 - s0**2 + Ri**2)
        smax_i = -p0 + np.sqrt(p0**2 - s0**2 + Ri**2)
    
        smin_o = -p0 - np.sqrt(p0**2 - s0**2 + Ro**2)
        smax_o = -p0 + np.sqrt(p0**2 - s0**2 + Ro**2)
    
        ds_o = smax_o - smin_o
        ds_i = smax_i - smin_i
    
        shape_ds_o = ds_o.shape
        ds_o = ds_o.ravel()
        ds_o[np.isnan(ds_o)] = 0
        ds_o[np.where((smax_o.ravel() < 0) | (smin_o.ravel() < 0))] = 0
        ds_o = ds_o.reshape(shape_ds_o)
    
        shape_ds_i = ds_i.shape
        ds_i = ds_i.ravel()
        ds_i[np.isnan(ds_i)] = 0
        ds_i[np.where((smax_i.ravel() < 0) | (smin_i.ravel() < 0))] = 0
        ds_i = ds_i.reshape(shape_ds_i)

    elif (s0 < Ri):

        print('observer inside sphere')
        
        #smin_i = 0
        smax_i = -p0 + np.sqrt(p0**2 - s0**2 + Ri**2)
        
        #smin_o = 0
        smax_o = -p0 + np.sqrt(p0**2 - s0**2 + Ro**2)
        
        ds_o = smax_o# - smin_o
        ds_i = smax_i# - smin_i
        
        shape_ds_o = ds_o.shape
        ds_o = ds_o.ravel()
        ds_o[np.isnan(ds_o)] = 0
        ds_o[np.where(smax_o.ravel() < 0)] = 0
        ds_o = ds_o.reshape(shape_ds_o)
        
        shape_ds_i = ds_i.shape
        ds_i = ds_i.ravel()
        ds_i[np.isnan(ds_i)] = 0
        ds_i[np.where(smax_i.ravel() < 0)] = 0
        ds_i = ds_i.reshape(shape_ds_i)
       
    elif ((s0 > Ri) & (s0  < Ro)) | ((s0 == 0.0) & (Ri == 0.0) & (s0 < Ro)):

        print('observer in shell')

        smin_i = -p0 - np.sqrt(p0**2 - s0**2 + Ri**2)
        smax_i = -p0 + np.sqrt(p0**2 - s0**2 + Ri**2)
    
        #smin_o = 0
        smax_o = -p0 + np.sqrt(p0**2 - s0**2 + Ro**2)
    
        ds_o = smax_o# - smin_o
        ds_i = smax_i - smin_i
    
        shape_ds_o = ds_o.shape
        ds_o = ds_o.ravel()
        ds_o[np.isnan(ds_o)] = 0
        ds_o[np.where(smax_o.ravel() < 0)] = 0
        ds_o = ds_o.reshape(shape_ds_o)
    
        shape_ds_i = ds_i.shape
        ds_i = ds_i.ravel()
        ds_i[np.isnan(ds_i)] = 0
        ds_i[np.where((smax_i.ravel() < 0) | (smin_i.ravel() < 0))] = 0
        ds_i = ds_i.reshape(shape_ds_i)
        
    else:
        
        print('case not included, yet')
        return(np.zeros(p0.shape))


    if luminosity_flag == False:

        # for map integration
        dF = (ds_o - ds_i)
        val = dF

    else:

        # for luminosity integration
        dL = 1/3*(ds_o**3 - ds_i**3)
        val = dL
    
    return(val)




@nb.njit(float64[:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:],
                      float64,
                      float64,float64,float64,
                      float64,float64,float64,
                      float64,float64,float64,float64,
                      float64,float64,
                      bool_),fastmath=True,parallel=True)
def NFW_profile_general(s: np.ndarray, b: np.ndarray, l: np.ndarray,
                        ds: float,
                        x0: float, y0: float, z0: float,
                        xT: float, yT: float, zT: float,
                        alpha: float, beta: float, gamma: float, R0: float,
                        rho_dm: float, n: float,
                        luminosity_flag: bool):

    deg2rad = pi/180.

    # los vector
    x = x0 + s*cos(deg2rad*l)*cos(deg2rad*b)
    y = y0 + s*sin(deg2rad*l)*cos(deg2rad*b)
    z = z0 + s*sin(deg2rad*b)

    Rxyz = sqrt((x-xT)**2 + (y-yT)**2 + (z-zT)**2)

    # halo slices
    halo_slices = rho_dm/(((Rxyz/R0)**gamma * (1 + (Rxyz/R0)**alpha)**((beta-gamma)/alpha)))
    
    # for map integration
    halo_n_slices = halo_slices**n

    
    if luminosity_flag == False:
    
        # for map integration
        halo_n_map_slices = halo_n_slices

    else:

        # for luminosity integration
        halo_n_map_slices = halo_n_slices * s**2
    
    # los
    val = np.sum(halo_n_map_slices*ds,axis=0)
    
    return(val)





@nb.njit(float64[:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:],
                      float64,
                      float64,float64,float64,
                      float64,float64,float64,
                      float64,float64,
                      float64,float64,float64,
                      float64,float64,float64,float64,
                      bool_),fastmath=True,parallel=True)
def Tilted_Ellipsoid_general(s: np.ndarray, b: np.ndarray, l: np.ndarray,
                             ds: float,
                             x0: float, y0: float, z0: float,
                             xT: float, yT: float, zT: float,
                             phi: float, theta: float,
                             barX: float, barY: float, barZ: float,
                             barPerp: float, barPara: float, barREnd: float, barHEnd: float,
                             luminosity_flag: bool):

    deg2rad = pi/180.

    phi = -phi*deg2rad
    
    # los vector
    x = xT - s*cos(deg2rad*l)*cos(deg2rad*b)
    y = yT - s*sin(deg2rad*l)*cos(deg2rad*b)
    z = zT - s*sin(deg2rad*b)

    # rotated variables (might nned to be x-xT, etc.)
    #xp = cos(phi)*cos(theta)*(x-xT) + sin(phi)*(y-yT) - cos(phi)*sin(theta)*(z-zT)
    #yp = -sin(phi)*cos(theta)*(x-xT) + cos(phi)*(y-yT) + sin(phi)*sin(theta)*(z-zT)
    #zp = sin(theta)*(x-xT) + cos(theta)*(z-zT)    

    # correct trafo?
    xp = cos(phi)*cos(theta)*x + sin(phi)*y - cos(phi)*sin(theta)*z
    yp = -sin(phi)*cos(theta)*x + cos(phi)*y + sin(phi)*sin(theta)*z
    zp = sin(theta)*x + cos(theta)*z

    
    # shapes
    rPerp = ((fabs(xp)/barX)**barPerp + (fabs(yp)/barY)**barPerp)**(1/barPerp)
    rs = (rPerp**barPara  + (fabs(zp)/barZ)**barPara)**(1/barPara)

    # init empty val array
    dims = s.shape
    bar_slices = zeros(dims)
    
    # need to loop over all entries?
    for i_s in range(dims[0]):
        for i_b in range(dims[1]):
            for i_l in range(dims[2]):
                
                
                # inside and outside
                if rs[i_s,i_b,i_l] <= barREnd:
                    bar_slices[i_s,i_b,i_l] = 1/cosh(rs[i_s,i_b,i_l])**2

                else:

                    bar_slices[i_s,i_b,i_l] = exp(-(rs[i_s,i_b,i_l]-barREnd)**2/barHEnd**2)/cosh(rs[i_s,i_b,i_l])**2

    # flux or luminosity
    if luminosity_flag == False:

        # for map integration
        bar_map_slices = bar_slices

    else:

        # for luminosity integration
        bar_map_slices = bar_slices * s**2

    # los
    val = np.sum(bar_map_slices*ds,axis=0)

    return(val)



@nb.njit(float64[:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:],
                      float64,
                      float64,float64,float64,
                      float64,float64,float64,
                      float64,float64,
                      float64,float64,
                      float64,float64,float64,
                      bool_),fastmath=True,parallel=True)
def _Nuclear_Stellar_Cluster_function(s: np.ndarray, b: np.ndarray, l: np.ndarray,
                                      ds: float,
                                      x0: float, y0: float, z0: float,
                                      xT: float, yT: float, zT: float,
                                      phi: float, theta: float,
                                      rho0: float, rho1: float,
                                      R0: float, Rin: float, Rout: float,
                                      luminosity_flag: bool):

    deg2rad = pi/180.

    phi = -phi*deg2rad
    
    # los vector
    x = xT - s*cos(deg2rad*l)*cos(deg2rad*b)
    y = yT - s*sin(deg2rad*l)*cos(deg2rad*b)
    z = zT - s*sin(deg2rad*b)

    # correct trafo?
    xp = cos(phi)*cos(theta)*x + sin(phi)*y - cos(phi)*sin(theta)*z
    yp = -sin(phi)*cos(theta)*x + cos(phi)*y + sin(phi)*sin(theta)*z
    zp = sin(theta)*x + cos(theta)*z

    Rp = sqrt(xp**2+yp**2+zp**2)

    # init empty val array
    dims = s.shape
    nsc_slices = zeros(dims)

    # need to loop over all entries
    for i_s in range(dims[0]):
        for i_b in range(dims[1]):
            for i_l in range(dims[2]):
                
                # inside and outside
                if Rp[i_s,i_b,i_l] <= Rin:
                    nsc_slices[i_s,i_b,i_l] = rho0/(1+(Rp[i_s,i_b,i_l]/R0)**2)

                elif (Rp[i_s,i_b,i_l] > Rin) & (Rp[i_s,i_b,i_l] <= Rout):
                    nsc_slices[i_s,i_b,i_l] = rho1/(1+(Rp[i_s,i_b,i_l]/R0)**3)

                else:
                    nsc_slices[i_s,i_b,i_l] = 0.

    
    # flux or luminosity
    if luminosity_flag == False:

        # for map integration
        nsc_map_slices = nsc_slices

    else:

        # for luminosity integration
        nsc_map_slices = nsc_slices * s**2

    # los
    val = np.sum(nsc_map_slices*ds,axis=0)

    return(val)



@nb.njit(float64[:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:],
                      float64,
                      float64,float64,float64,
                      float64,float64,float64,
                      float64,float64,
                      float64,float64,
                      float64,float64,float64,
                      bool_),fastmath=True,parallel=True)
def Nuclear_Stellar_Cluster_function(s: np.ndarray, b: np.ndarray, l: np.ndarray,
                                     ds: float,
                                     x0: float, y0: float, z0: float,
                                     xT: float, yT: float, zT: float,
                                     phi: float, theta: float,
                                     rho0: float, rho1: float,
                                     R0: float, Rin: float, Rout: float,
                                     luminosity_flag: bool):

    deg2rad = pi/180.

    phi = -phi*deg2rad
    
    # los vector
    x = xT - s*cos(deg2rad*l)*cos(deg2rad*b)
    y = yT - s*sin(deg2rad*l)*cos(deg2rad*b)
    z = zT - s*sin(deg2rad*b)

    # correct trafo?
    xp = cos(phi)*cos(theta)*x + sin(phi)*y - cos(phi)*sin(theta)*z
    yp = -sin(phi)*cos(theta)*x + cos(phi)*y + sin(phi)*sin(theta)*z
    zp = sin(theta)*x + cos(theta)*z

    Rp = sqrt(xp**2+yp**2+zp**2)

    # init empty val array
    dims = s.shape
    nsc_slices = zeros(dims)

    # need to loop over all entries
    for i_s in range(dims[0]):
        for i_b in range(dims[1]):
            for i_l in range(dims[2]):
                
                # inside and outside
                if Rp[i_s,i_b,i_l] <= Rin:
                    nsc_slices[i_s,i_b,i_l] = rho0/(1+(Rp[i_s,i_b,i_l]/R0)**2)

                elif (Rp[i_s,i_b,i_l] > Rin) & (Rp[i_s,i_b,i_l] <= Rout):
                    nsc_slices[i_s,i_b,i_l] = rho1/(1+(Rp[i_s,i_b,i_l]/R0)**3)

                else:
                    nsc_slices[i_s,i_b,i_l] = 0.

    
    # flux or luminosity
    if luminosity_flag == False:

        # for map integration
        nsc_map_slices = nsc_slices

    else:

        # for luminosity integration
        nsc_map_slices = nsc_slices * s**2

    # los
    val = np.sum(nsc_map_slices*ds,axis=0)

    return(val)



@nb.njit(float64[:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:],
                      float64,
                      float64,float64,float64,
                      float64,float64,float64,
                      float64,float64,
                      float64,float64,float64,
                      float64,float64,float64,float64,
                      float64,
                      float64, float64,
                      bool_),fastmath=True,parallel=True)
def Nuclear_Stellar_Disk_function(s: np.ndarray, b: np.ndarray, l: np.ndarray,
                                  ds: float,
                                  x0: float, y0: float, z0: float,
                                  xT: float, yT: float, zT: float,
                                  phi: float, theta: float,
                                  rho0: float, rho1: float, rho2: float,
                                  R0: float, Rin: float, Rout: float, Rmax: float,
                                  zscl: float,
                                  Rscut: float, zcut: float,
                                  luminosity_flag: bool):

    deg2rad = pi/180.

    phi = -phi*deg2rad
    
    # los vector
    x = xT - s*cos(deg2rad*l)*cos(deg2rad*b)
    y = yT - s*sin(deg2rad*l)*cos(deg2rad*b)
    z = zT - s*sin(deg2rad*b)

    # correct trafo?
    xp = cos(phi)*cos(theta)*x + sin(phi)*y - cos(phi)*sin(theta)*z
    yp = -sin(phi)*cos(theta)*x + cos(phi)*y + sin(phi)*sin(theta)*z
    zp = sin(theta)*x + cos(theta)*z

    Rp = sqrt(xp**2+yp**2)
    Rs = sqrt(Rp**2+zp**2)

    # init empty val array
    dims = s.shape
    nsd_slices = zeros(dims)

    # need to loop over all entries
    for i_s in range(dims[0]):
        for i_b in range(dims[1]):
            for i_l in range(dims[2]):

                if fabs(zp[i_s,i_b,i_l]) <= zcut:
                
                    # inside, middle, and outside
                    if Rp[i_s,i_b,i_l] <= Rin:
                        nsd_slices[i_s,i_b,i_l] = rho0*(Rp[i_s,i_b,i_l]/R0)**(-0.1)*exp(-fabs(zp[i_s,i_b,i_l]/zscl))*exp(-(Rs[i_s,i_b,i_l]-Rscut)**2/Rscut**2)
                        
                    elif (Rp[i_s,i_b,i_l] > Rin) & (Rp[i_s,i_b,i_l] <= Rout):
                        nsd_slices[i_s,i_b,i_l] = rho1*(Rp[i_s,i_b,i_l]/R0)**(-3.5)*exp(-fabs(zp[i_s,i_b,i_l]/zscl))*exp(-(Rs[i_s,i_b,i_l]-Rscut)**2/Rscut**2)

                    elif (Rp[i_s,i_b,i_l] > Rout) & (Rp[i_s,i_b,i_l] <= Rmax):
                        nsd_slices[i_s,i_b,i_l] = rho2*(Rp[i_s,i_b,i_l]/R0)**(-10.0)*exp(-fabs(zp[i_s,i_b,i_l]/zscl))*exp(-(Rs[i_s,i_b,i_l]-Rscut)**2/Rscut**2)
                        
                    else:
                        nsd_slices[i_s,i_b,i_l] = 0.

                else:
                    nsd_slices[i_s,i_b,i_l] = 0.
    
    # flux or luminosity
    if luminosity_flag == False:

        # for map integration
        nsd_map_slices = nsd_slices

    else:

        # for luminosity integration
        nsd_map_slices = nsd_slices * s**2

    # los
    val = np.sum(nsd_map_slices*ds,axis=0)

    return(val)


