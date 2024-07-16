import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan, arctan2, pi, exp, sqrt, cosh, fabs, deg2rad, rad2deg

from math import erf

from numba import jit, njit, prange

from scipy.interpolate import griddata
from scipy.interpolate import interp2d as interpol2d
from scipy.integrate import quad

import matplotlib.pyplot as plt
from matplotlib import colors

#import time
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c

import pandas as pd

from functools import partial

#from tqdm.autonotebook import tqdm

from astroquery.jplhorizons import Horizons

from density_functions import *
#from luminosity_functions import *
#from special_objects import *

from num_los_setup import num_los_setup

from utils import *

from healpy.newvisufunc import projview, newprojplot


# conversions
d2r = pi/180
r2d = 180/pi

# default values
default_time = '2000-01-01 00:00:00.00'
default_observer = 399 # Earth
default_pixelsize = 3. # degrees

# default frame
default_phi_min = -180. # full sky
default_phi_max = +180.
default_theta_min = -90.
default_theta_max = +90.
default_frame = 'E'

# default integration steps in relative units
default_smin = 0
default_smax = 10
default_n_los_steps = 400

class los(object):
    """
    Generic to-be-line-of-sight-integrated object.

    Working as parent class.
    """

    def __init__(self,**kwargs):

        self._time = default_time if not 'time' in kwargs else kwargs.pop('time')
        self._observer = default_observer if not 'observer' in kwargs else kwargs.pop('observer')
        self._pixelsize = default_pixelsize if not 'pixelsize' in kwargs else kwargs.pop('pixelsize')

        self._phi_min = default_phi_min if not 'phi_min' in kwargs else kwargs.pop('phi_min')
        self._phi_max = default_phi_max if not 'phi_max' in kwargs else kwargs.pop('phi_max')
        self._theta_min = default_theta_min if not 'theta_min' in kwargs else kwargs.pop('theta_min')
        self._theta_max = default_theta_max if not 'theta_max' in kwargs else kwargs.pop('theta_max')

        self._smin = default_smin if not 'smin' in kwargs else kwargs.pop('smin')
        self._smax = default_smax if not 'smax' in kwargs else kwargs.pop('smax')
        self._n_los_steps = default_n_los_steps if not 'n_los_steps' in kwargs else kwargs.pop('n_los_steps')
    
        
        self._kwargs = kwargs

        self.define_sky(phi_min=self._phi_min,
                        phi_max=self._phi_max,
                        theta_min=self._theta_min,
                        theta_max=self._theta_max)

        self._frame = default_frame if not 'frame' in kwargs else kwargs.pop('frame')

        if (self._frame == 'G') and (self._observer != 10):
            print('Warning: Frame is Galactic. Will set observer to Sun position!')
            self._observer = 10

        
    def define_sky(self,phi_min=-180,phi_max=180,theta_min=-90,theta_max=90):
        """
        Define image space of where the los-integrated object lives in.
        """
        # definition of image space
        # minmax range (similar to above but now also including s)
        if ("phi_min" in self._kwargs) and ("phi_max" in self._kwargs) and ("theta_min" in self._kwargs) and ("theta_max" in self._kwargs):
            self._phi_min, self._phi_max = kwargs.pop("phi_min"), kwargs.pop("phi_max")
            self._theta_min, self._theta_max = kwargs.pop("theta_min"), kwargs.pop("theta_max")
        else:
            self._phi_min, self._phi_max = phi_min, phi_max
            self._theta_min, self._theta_max = theta_min, theta_max
            
        # definition of pixel size and number of bins
        self._n_phi = int((self._phi_max-self._phi_min)/self._pixelsize)
        self._n_theta = int((self._theta_max-self._theta_min)/self._pixelsize)

        self._theta_arrg = deg2rad(np.linspace(self._theta_min,self._theta_max,self._n_theta+1))
        self._phi_arrg = deg2rad(np.linspace(self._phi_min,self._phi_max,self._n_phi+1))
        self._theta_arr = (self._theta_arrg[1:]+self._theta_arrg[0:-1])/2
        self._phi_arr = (self._phi_arrg[1:]+self._phi_arrg[0:-1])/2

        # define 2D meshgrid for image coordinates
        self._PHI_ARRg, self._THETA_ARRg = np.meshgrid(self._phi_arrg,self._theta_arrg)
        self._PHI_ARR, self._THETA_ARR = np.meshgrid(self._phi_arr,self._theta_arr)

        # jacobian (integral measure on a sphere, exact for this pixel definition, should be 4pi)
        self._dOmega = deg2rad((np.sin(self._THETA_ARR + deg2rad(self._pixelsize/2)) - np.sin(self._THETA_ARR - deg2rad(self._pixelsize/2))) * self._pixelsize)


    
        

    def density_function(self):
        """
        Is used with the specific source.
        """
        pass


    def luminosity_function(self):
        """
        Is used with the specific source.
        """
        pass


    def los_source(self):
        """
        Do the los integration appropriate for the source.
        """
        # calculate Earth coordinates for time
        vecs = calculate_body_coordinates(self._observer,self._time)

        # extract cartesian coordiantes
        _x, _y, _z = vecs['x'].data[0], vecs['y'].data[0], vecs['z'].data[0]

        # do the actual integration with function in child class
        result = self.density_function(_x,_y,_z,**self._kwargs)

        self._image = result/(4*np.pi)
        self._image_integrated = self._image*self._dOmega
        

    def luminosity_source(self):
        """
        Do the los integration appropriate for the source weighted with s^2 for luminosity.
        """
        # calculate Earth coordinates for time
        vecs = calculate_body_coordinates(self._observer,self._time)

        # extract cartesian coordiantes
        _x, _y, _z = vecs['x'].data[0], vecs['y'].data[0], vecs['z'].data[0]

        # do the actual integration with function in child class
        result = self.luminosity_function(_x,_y,_z,**self._kwargs)

        self._luminosity_image = result
        self._luminosity_image_integrated = self._luminosity_image*self._dOmega
        self._luminosity = np.sum(self._luminosity_image_integrated)
        
        
    def plot_source(self,projection=None,luminosity_flag=False,integrated=False,**kwargs):
        """
        Plot the los-integrated object with or without projection.
        """
        plt.figure(figsize=(10.24,7.68))
        plt.subplot(projection=projection)

        if projection == None:
            r2d = 180/pi
            r2d_l = 1
            cbar_orientation = 'vertical'

        else:
            r2d = 1
            r2d_l = 180/pi
            cbar_orientation = 'horizontal'

            
        #o = np.round(np.log10(self._image.max()))

        if luminosity_flag == False:

            if integrated == False:
        
                plot_image = self._image

            else:

                plot_image = self._image_integrated

        else:

            if integrated == False:
            
                plot_image = self._luminosity_image

            else:

                plot_image = self._luminosity_image_integrated
            
        
        plt.pcolormesh(self._PHI_ARRg*r2d,
                       self._THETA_ARRg*r2d,
                       np.flip(plot_image,axis=1),
                       #plot_image,
                       **kwargs)

        cbar = plt.colorbar(orientation='horizontal')
        cbar.ax.tick_params(labelsize=20)
        #if self._spectrum != {}:
        #    cbar.set_label(label='Flux ({0}-{1} MeV) '.format(self._emin,self._emax)+r'$[10^{{ {0} }}$'.format(np.int(o))+r'$\,\mathrm{ph\,cm^{-2}\,s^{-1}\,sr^{-1}}$]',fontsize=20)
        #else:
        cbar.set_label(label='Flux [arbitrary units]',fontsize=20)

        if projection != None:
        
            plt.xticks(np.array([120,60,0,-60,-120])/r2d_l,
                       labels=[r'$-120^{\circ}$'+'\n',
                               r'$-60^{\circ}$'+'\n',
                               r'$0^{\circ}$'+'\n',
                               r'$+60^{\circ}$'+'\n',
                               r'$+120^{\circ}$'+'\n'],
                       color='darkgray',fontsize=20)
            plt.yticks(np.array([-60,-30,0,30,60])/r2d_l,fontsize=20)

            #plt.xlim(self._phi_max,self._phi_min)                       


        else:
            plt.xlim(self._phi_max/r2d_l,self._phi_min/r2d_l)

            
        if self._frame == 'G':
            xlabel = 'Galactic Longitude [deg]'
            ylabel = 'Galactic Latitude [deg]'
        else:
            xlabel = 'Ecliptic Longitude [deg]'
            ylabel = 'Ecliptic Latitude [deg]'

        plt.xlabel(xlabel,fontsize=20)
        plt.ylabel(ylabel,fontsize=20)
        plt.grid()

        plt.title('Date: {0}'.format(self._time),fontsize=20)
        plt.tight_layout()



   
        

    def _image_projector(self):
        """
        Create healpix image projector from cartesian maps.
        """
        
        # calculate image with 4 times higher resolution by spline interpolation
        self._fine_image, self._fine_theta, self._fine_phi = fine_grid_interp(self._theta_arr,
                                                                              self._phi_arr,
                                                                              self._image)

        self._nside = get_nside_from_image_pixel_number(self._image)
        
        self._healpix_image = array2healpix(self._fine_image,
                                            self._fine_phi,
                                            self._fine_theta,
                                            nside=self._nside,
                                            max_iter=100)

        self._projector = hp.projector.CartesianProj(coord=[self._frame],
                                                     xsize=self._n_phi*4,
                                                     ysize=self._n_theta*4)        



    def _plot_healpix_image(self,coord=[default_frame]):

        projview(
            self._healpix_image,
            coord=coord,
            graticule=True,
            graticule_labels=True,
            unit="Flux [arbitrary units]",
            xlabel="longitude",
            ylabel="latitude",
            cb_orientation="vertical",
            #min=-0.05,
            #max=0.05,
            latitude_grid_spacing=30,
            projection_type="mollweide",
            title="",
        );
        
        #hp.mollview(self._healpix_image,coord=coord,flip='astro')
        #hp.graticule(color='white',alpha=0.2)

        
        
    def _trafo_image(self,frames):
        """

        TS: NOT WORKING PROERPYL AT THE MOMENT
        
        Projecting images between coordinate frames:
        Example:
        Keyword frames = ['E','G'] converts image from ecliptic to galactic.
        'E': ecliptic
        'G': galactic
        'C': celestial
        """

        if not hasattr(self,'_projector'):

            self.image_projetor()

        

        self._fine_image_converted = self._projector.projmap(change_coord(self._healpix_image,
                                                                          coord=frames),
                                                             vec2pix_func=partial(hp.vec2pix,
                                                                                  self._nside))

        print('fine_image_converted.shape',self._fine_image_converted.shape)
        
        #set to previous resolution:
        self._image, _, _ = fine_grid_interp(self._fine_theta,
                                             self._fine_phi,
                                             self._fine_image_converted,
                                             scl=0.25)

        
    def trafo_image_EC2GAL(self,healpix_trafo=True):
        """
        Transformation of Ecliptic frame to Galactic frame by re-gridding the image.
        Might lose some sensitivity here (use healpix trafo by default: a bit slower but fewer artefacts).
        """
        
        if self._frame == 'E':
            
            #print('Changing frame to: Galactic')

            #if healpix_trafo == True:

            #    if self.image_projector == None:

            #        self.image_projetor()

            #    else:

            #        fine_image_galactic = was.projmap(change_coord(img,coord=['E']),vec2pix_func=partial(hp.vec2pix, nside))
                    
                
                
            #else:
            
            # trafo to gal. coordinates
            self.lon,self.lat = trafo_ec2gal(self._PHI_ARR,
                                             self._THETA_ARR,
                                             deg=False)

            # interpolate ecliptic torus to new irregular grid
            image_gal = griddata((self.lon.ravel(), self.lat.ravel()),            # new coordinates are not regular any more
                                 self._image.ravel(),                              # the image stays the same, just gets re-arranged
                                 (self._PHI_ARR.ravel(), self._THETA_ARR.ravel()),  # the representation we want to be the same (regular grid)
                                 method='nearest')                                 # nearest neighbour interpolation avoids 'edge effects'

            image_gal = image_gal.reshape(self._PHI_ARR.shape) # rebuild the image to be a regular 2D array
                

            self._frame = 'G'
            self._image = image_gal

        else:
            print('Frame is already Galactic!')
        

    def trafo_image_GAL2EC(self):
        """
        Transformation of Galactic frame to Ecliptic frame by re-gridding the image.
        Might lose some sensitivity here (should replace with healpix trafo in future).
        """

        if self._frame == 'G':
            #print('Changing frame to: Ecliptic')
        
            # trafo to gal. coordinates
            self.lon,self.lat = trafo_gal2ec(self._PHI_ARR,
                                             self._THETA_ARR,
                                             deg=False)

            # interpolate ecliptic torus to new irregular grid
            image_ec = griddata((self.lon.ravel(), self.lat.ravel()),            # new coordinates are not regular any more
                                self._image.ravel(),                              # the image stays the same, just gets re-arranged
                                (self._PHI_ARR.ravel(), self._THETA_ARR.ravel()),  # the representation we want to be the same (regular grid)
                                method='nearest')                                 # nearest neighbour interpolation avoids 'edge effects'

            image_ec = image_ec.reshape(self._PHI_ARR.shape) # rebuild the image to be a regular 2D array
            
            self._frame = 'E'
            self._image = image_ec

        else:
            print('Frame is already Ecliptic!')




    


            
        

    def calc_flux(self):
        try:
            self._flux = np.sum(self._image*self._dOmega)
        except AttributeError:
            print('Image not yet calculated, use los_source() first.')

    @property
    def dOmega(self):
        return np.sum(self._dOmega)
        
    @property        
    def flux(self):
        try:
            if not hasattr(self, '_flux'):
                self.calc_flux()
                return self._flux
            else:
                return self._flux
        except:
            pass

    @property
    def luminosity(self):
        try:
            if not hasattr(self, '_luminosity'):
                self.luminosity_source()
                return self._luminosity
            else:
                return self._luminosity
        except:
            pass


    @property
    def luminosity_image(self):
        try:
            if not hasattr(self, '_luminosity'):
                self.luminosity_source()
                return self._luminosity_image
            else:
                return self._luminosity_image
        except:
            pass

    @property
    def luminosity_image_integrated(self):
        try:
            if not hasattr(self, '_luminosity'):
                self.luminosity_source()
                return self._luminosity_image_integrated
            else:
                return self._luminosity_image_integrated
        except:
            pass

        
        

    @property
    def time(self):
        return self._time




    @property
    def coordinates(self):
        return(self._PHI_ARR,self._THETA_ARR,self._PHI_ARRg,self._THETA_ARRg,self._dOmega)



    
    def flux_in_circular_region(self,phi_center,theta_center,radius):
        self._center = phi_center,theta_center
        self._radius = radius
        distances = angular_distance(self._center[0],
                                     self._center[1],
                                     self._PHI_ARR*180/np.pi,
                                     self._THETA_ARR*180/np.pi)

        idx = np.where(distances <= self._radius)

        self._flux_in_region = np.sum(self._image[idx[0],idx[1]]*self._dOmega[idx[0],idx[1]])

        return self._flux_in_region

    
        
    def flux_in_rectangular_region(self,phi_center,theta_center,phi_width,theta_width):
        self._center = phi_center,theta_center
        self._widths = phi_width,theta_width
        distances = angular_distance(self._center[0],
                                     self._center[1],
                                     self._PHI_ARR*180/np.pi,
                                     self._THETA_ARR*180/np.pi)

        idx = np.where((self._PHI_ARR*180/np.pi >= (self._center[0]-self._widths[0]/2)) & \
                       (self._PHI_ARR*180/np.pi <= (self._center[0]+self._widths[0]/2)) & \
                       (self._THETA_ARR*180/np.pi >= (self._center[1]-self._widths[1]/2)) & \
                       (self._THETA_ARR*180/np.pi >= (self._center[1]-self._widths[1]/2)))
                       

        self._flux_in_region = np.sum(self._image[idx[0],idx[1]]*self._dOmega[idx[0],idx[1]])

        return self._flux_in_region

    

    

    def normalise_by_flux(self,flux):

        self.calc_flux()

        flux_norm = flux/self._flux

        #flux_to_luminosity = self._flux/self._luminosity
        
        self._flux *= flux_norm
        self._image *= flux_norm
        self._image_integrated = self._image*self._dOmega

        self._old_luminosity = self._luminosity
        self._luminosity = flux_norm * self._luminosity
        self._luminosity_image *= self._luminosity / self._old_luminosity
        self._luminosity_image_integrated = self._luminosity_image*self._dOmega
        #self._luminosity = np.sum(self._luminosity_image_integrated)


        
    def normalise_by_luminosity(self,luminosity):

        self.calc_flux()

        luminosity_norm = luminosity/self._luminosity

        self._old_flux = self._flux
        self._flux = luminosity_norm * self._flux
        self._image *= self._flux / self._old_flux
        self._image_integrated = self._image*self._dOmega
        
        self._luminosity *= luminosity_norm
        self._luminosity_image *= luminosity_norm
        self._luminosity_image_integrated = self._luminosity_image*self._dOmega
    

        #F1 = L1

        #F2 = L2

        #F2 = F1 * L2 / L1

    
    
#############################################
#############################################
#### SOURCE FUNCTIONS #######################
#############################################
#############################################

# TS: not skilled enough to place them in different file
    
class MBA_Gaussian(los):
    """
    Main Belt Asteroids modelled as Gaussian torus with density rhoT1 = 19.27 objects/AU^3,
    large radius RT = 2.6090 AU, radial width sigmaT = 0.394 AU, vertical radius Rt = 0.268 AU,
    and vertical width sigmat = 0.2609 AU, centred around the Sun.
    """
    
    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'MBA_Gaussian'

        self.nls = num_los_setup(smin=0,smax=10,n_los_steps=400,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)

        # MBA smooth parameters
        # (single Gaussian torus fit to 3D density distribution)
        # size units in AU, density rhoT objects per AU3
        xT, yT, zT = 0., 0., 0.

        rhoT1 =     68.45e3      # +/-     0.05
        RT1 =       2.6185       # +/-    0.0006
        sigmaT1 =   461.63e-3    # +/-  0.4e-3
        Rt1 =       2.09e-3	 # +/-   0.32e-3
        sigmat1 =   374.71e-3	 # +/-  0.4e-3

        self._params = [xT, yT, zT, RT1, Rt1, sigmaT1, sigmat1]
        self._amplitude = rhoT1
        

    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):

        torus = Gaussian_Torus_Total(self.nls._grid_s,
                                     self.nls._grid_b,
                                     self.nls._grid_l,
                                     self.nls._ds,
                                     _x,_y,_z,
                                     *self._params,
                                     luminosity_flag)

                
        return self._amplitude*torus

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)




class MBA_Hungaria(los):
    """
    Hungaria population of the Main Belt Asteroids modelled as Gaussian torus.
    See XY for details.
    """

    def __init__(self,**kwargs):

        super().__init__(**kwargs)

        self.name = 'MBA_Hungaria'

        self.nls = num_los_setup(smin=0,smax=10,n_los_steps=400,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)


        # Inner MBA parameters (1 gaussian torus)
        xT, yT, zT = 0., 0., 0.

        rho0 = 6.15e3
        RT = 1.8451
        sigmaT = 124.5e-3
        Rt = 0.0048
        sigmat = 0.5016	

        self._params = [xT, yT, zT, RT, Rt, sigmaT, sigmat]
        self._amplitude = rho0        

        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):

        torus = Gaussian_Torus_Total(self.nls._grid_s,
                                     self.nls._grid_b,
                                     self.nls._grid_l,
                                     self.nls._ds,
                                     _x,_y,_z,
                                     *self._params,
                                     luminosity_flag)

                
        return self._amplitude*torus

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)


class MBA_Outer(los):
    """
    Outer astereoid population of the Main Belt Asteroids modelled as Gaussian torus.
    Hildas asteroids are ignored.
    See XY for details.
    """

    def __init__(self,**kwargs):

        super().__init__(**kwargs)

        self.name = 'MBA_Outer'

        self.nls = num_los_setup(smin=0,smax=10,n_los_steps=400,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)


        # Outer MBA parameters (1 gaussian torus)
        xT, yT, zT = 0., 0., 0.

        rho0 = 1.265e3
        RT = 3.2747
        sigmaT = 0.4502
        Rt = 0.0027
        sigmat = 0.5416

        self._params = [xT, yT, zT, RT, Rt, sigmaT, sigmat]
        self._amplitude = rho0        

        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):

        torus = Gaussian_Torus_Total(self.nls._grid_s,
                                     self.nls._grid_b,
                                     self.nls._grid_l,
                                     self.nls._ds,
                                     _x,_y,_z,
                                     *self._params,
                                     luminosity_flag)

                
        return self._amplitude*torus

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)


    
class Main_Belt_Asteroids(los):
    """
    Main Belt Asteroids modelled as five Gaussian tori with relative importance:
    Three main tori that merge into a stretched shape.
    Two tori that mimic accumulations above and below the ecliptic plane.
    See XY for details.
    """

    def __init__(self,**kwargs):

        super().__init__(**kwargs)

        self.name = 'Main_Belt_Asteroids'

        self.nls = num_los_setup(smin=0,smax=10,n_los_steps=400,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)

        # Total MBA model with 5 tori:
        # 2 tori for Hungaria objects (closest to Earth)
        # 3 tori to define shape that extends with radius
        # size units in AU, rhoT for comparison to other objects
        xT, yT, zT = 0., 0., 0.

        # Whatever: this fits best
        #torus 1
        rhoT1 = 	89.53e3	 # +/- 0.20e3
        RT1 = 	        2.3332	 # +/- 0.0010
        sigmaT1 = 	311.9e-3 # +/- 0.6e-3
        Rt1 = 	        -1.56e-3 # +/- 0.26e-3
        sigmat1 = 	159.16e-3# +/- 0.33e-3	
        
        # torus 2
        rhoT2 = 	45.24e3	# +/- 0.15e3
        RT2 = 	        2.8142	# +/- 0.0009
        sigmaT2 = 	358.2e-3# +/- 0.6e-3
        Rt2 = 	        6.10e-3 # +/- 0.33e-3
        sigmat2 = 	390.8e-3# +/- 0.7e-3
        
        # torus 3
        rhoT3 = 	1.762e3	# +/- 0.034e3	
        RT3 = 	        3.043	# +/- 0.004	
        sigmaT3 = 	0.5775	# +/- 0.0020	
        Rt3 = 	        -0.0117	# +/- 0.0022	
        sigmat3 =       0.7231	# +/- 0.0024
        
        # torus 4
        rhoT4 = 	7.67e3	# +/- 0.06e3						
        RT4 = 	        1.8400	# +/- 0.0014						
        sigmaT4 =       0.1373	# +/- 0.0010						
        Rt4 = 	        -0.5591	# +/- 0.0017	
        sigmat4 =       0.1511	# +/- 0.0013
        
        # torus 5
        rhoT5 = 	7.64e3	# +/- 0.06e3	
        RT5 = 	        1.8257	# +/- 0.0014	
        sigmaT5 = 	132.4e-3# +/- 1.0e-3	
        Rt5 = 	        0.5752	# +/- 0.0016	
        sigmat5 = 	0.1528	# +/- 0.0012
        

        self._params_1 = [xT, yT, zT, RT1, Rt1, sigmaT1, sigmat1]
        self._amplitude_1 = rhoT1

        self._params_2 = [xT, yT, zT, RT2, Rt2, sigmaT2, sigmat2]
        self._amplitude_2 = rhoT2

        self._params_3 = [xT, yT, zT, RT3, Rt3, sigmaT3, sigmat3]
        self._amplitude_3 = rhoT3

        self._params_4 = [xT, yT, zT, RT4, Rt4, sigmaT4, sigmat4]
        self._amplitude_4 = rhoT4

        self._params_5 = [xT, yT, zT, RT5, Rt5, sigmaT5, sigmat5]
        self._amplitude_5 = rhoT5

        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):

        torus_1 = Gaussian_Torus_Total(self.nls._grid_s,
                                       self.nls._grid_b,
                                       self.nls._grid_l,
                                       self.nls._ds,
                                       _x,_y,_z,
                                       *self._params_1,
                                       luminosity_flag)

        torus_2 = Gaussian_Torus_Total(self.nls._grid_s,
                                       self.nls._grid_b,
                                       self.nls._grid_l,
                                       self.nls._ds,
                                       _x,_y,_z,
                                       *self._params_2,
                                       luminosity_flag)

        torus_3 = Gaussian_Torus_Total(self.nls._grid_s,
                                       self.nls._grid_b,
                                       self.nls._grid_l,
                                       self.nls._ds,
                                       _x,_y,_z,
                                       *self._params_3,
                                       luminosity_flag)

        torus_4 = Gaussian_Torus_Total(self.nls._grid_s,
                                       self.nls._grid_b,
                                       self.nls._grid_l,
                                       self.nls._ds,
                                       _x,_y,_z,
                                       *self._params_4,
                                       luminosity_flag)

        torus_5 = Gaussian_Torus_Total(self.nls._grid_s,
                                       self.nls._grid_b,
                                       self.nls._grid_l,
                                       self.nls._ds,
                                       _x,_y,_z,
                                       *self._params_5,
                                       luminosity_flag)

                
        return self._amplitude_1*torus_1 + self._amplitude_2*torus_2 + self._amplitude_3*torus_3 + self._amplitude_4*torus_4 + self._amplitude_5*torus_5

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)


    

class Kuiper_Belt(los):
    """
    Kuiper Belt asteroids modelled as single torus.
    This is a rough estimate as many objects and therefore the distributions are unknown.
    See XY for details.
    """

    def __init__(self,**kwargs):

        super().__init__(**kwargs)

        self.name = 'Kuiper_Belt'

        self.nls = num_los_setup(smin=0,smax=60,n_los_steps=600,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)


        # Kuiper belt smooth parameters
        xT, yT, zT = 0., 0., 0.
        
        rhoT1   = 3.73e-3
        RT1     = 11
        sigmaT1 = 27.9	
        Rt1     = -1.3
        sigmat1 = 11.62

        rhoT2   = 0.0304
        RT2     = 35.73	
        sigmaT2 = 4.70	
        Rt2     = -2.47	
        sigmat2 = 8.77	

        rhoT3   = 0.1150	
        RT3     = 42.54	
        sigmaT3 = 3.61	
        Rt3     = 0.33	
        sigmat3 = 2.68

        self._params_1 = [xT, yT, zT, RT1, Rt1, sigmaT1, sigmat1]
        self._amplitude_1 = rhoT1

        self._params_2 = [xT, yT, zT, RT2, Rt2, sigmaT2, sigmat2]
        self._amplitude_2 = rhoT2

        self._params_3 = [xT, yT, zT, RT3, Rt3, sigmaT3, sigmat3]
        self._amplitude_3 = rhoT3

        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):

        torus_1 = Gaussian_Torus_Total(self.nls._grid_s,
                                       self.nls._grid_b,
                                       self.nls._grid_l,
                                       self.nls._ds,
                                       _x,_y,_z,
                                       *self._params_1,
                                       luminosity_flag)

        torus_2 = Gaussian_Torus_Total(self.nls._grid_s,
                                       self.nls._grid_b,
                                       self.nls._grid_l,
                                       self.nls._ds,
                                       _x,_y,_z,
                                       *self._params_2,
                                       luminosity_flag)

        torus_3 = Gaussian_Torus_Total(self.nls._grid_s,
                                       self.nls._grid_b,
                                       self.nls._grid_l,
                                       self.nls._ds,
                                       _x,_y,_z,
                                       *self._params_3,
                                       luminosity_flag)

        return self._amplitude_1*torus_1 + self._amplitude_2*torus_2 + self._amplitude_3*torus_3

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)



class Spherical_Shell_Gaussian(los):
    """
    Spherical shell with Gaussian width .
    Input parameters:
    xT: x-coordinate of centre of shell
    yT: y-coordinate of centre of shell
    zT: z-coordinate of centre of shell
    RT: radius of shell
    sigmaT: radial width of shell
    rho0: normalisation (so far unitless)
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'Spherical_Shell_Gaussian'

        self.nls = num_los_setup(smin=self._smin,
                                 smax=self._smax,
                                 n_los_steps=self._n_los_steps,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)

        RT = kwargs.pop('RT')
        sigmaT = kwargs.pop('sigmaT')
        
        xT = kwargs.pop('xT')
        yT = kwargs.pop('yT')
        zT = kwargs.pop('zT')

        rho0 = kwargs.pop('rho0')

        self._params = [xT,yT,zT,RT,sigmaT]
        self._amplitude = rho0
        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):
        
        shell = Gaussian_Shell(self.nls._grid_s,
                               self.nls._grid_b,
                               self.nls._grid_l,
                               self.nls._ds,
                               _x,_y,_z,
                               *self._params,
                               luminosity_flag)
        
        return self._amplitude*shell

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)
    
    
    

class Gaussian_Torus(los):
    """
    Gaussian torus.
    Input parameters:
    xT: x-coordinate of centre of torus
    yT: y-coordinate of centre of torus
    zT: z-coordinate of centre of torus
    RT: large radius of torus
    sigmaT: radial width of torus
    rt: small radius of torus
    sigmat: vertical width of torus
    rho0: normalisation (so far unitless)
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'Gaussian_Torus'

        self.nls = num_los_setup(smin=self._smin,
                                 smax=self._smax,
                                 n_los_steps=self._n_los_steps,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)

    def density_function(self,_x,_y,_z,**kwargs):
        
        #xT, yT, zT = 0., 0., 0.

        RT = kwargs.pop('RT')
        Rt = kwargs.pop('Rt')
        sigmaT = kwargs.pop('sigmaT')
        sigmat = kwargs.pop('sigmat')
        
        xT = kwargs.pop('xT')
        yT = kwargs.pop('yT')
        zT = kwargs.pop('zT')

        
        
        torus = Gaussian_Torus_Total(self.nls._grid_s,
                                     self.nls._grid_b,
                                     self.nls._grid_l,
                                     self.nls._ds,
                                     _x,_y,_z,
                                     xT,yT,zT,
                                     RT,Rt,sigmaT,sigmat)

        rho0 = kwargs.pop('rho0')
        
        return rho0*torus







class Jovian_Trojans(los):
    """
    Jovian trojans modelled as combination of Gaussian tori and Gaussian hotspots..
    See XY for details.
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'Jovian_Trojans'

        self.nls = num_los_setup(smin=0,
                                 smax=10,
                                 n_los_steps=400,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)


        # Jovian trojan parameters trojan_stuff_sph2() fit
        # valid for 2 z-direction Gaussian (better description of z-distribution)
        xT, yT, zT = 0., 0., 0.

        # fit parameters from migrad
        K = 	        1.00  # +/-	0.01
        RT = 	        4.9894 # +/-	0.05					
        sigmaT = 	0.278 # +/-	0.0027					
        tx = 	        0.974 # +/-	0.009					
        ty = 	        1.613 # +/-	0.018					
        theta4 =       -0.487 # +/-	-0.0014					
        theta5 = 	5.115 # +/-	0.05					
        tz1 = 	        0.488 # +/-	0.005
        tz2 = 	        1.120 # +/-	0.014
        rhoz1 = 	793   # +/-	0.05
        rhoz2 =	        686   # +/-	0.035	
    
        # For =  calculation of # +/- Lagrange points
        Mj = 0.3178 # kilo Earth masses
        Ms = 332.950   # kilo Earth masses
    
        # time for which this has been determined:
        time0 = '2021-11-21 00:00:00.000'

        # calculate Jupiter coordinates for time0
        vecs0 = calculate_body_coordinates('5',time0)
        # extract cartesian coordiantes for time 0
        _xj0, _yj0, _zj0 = vecs0['x'].data[0], vecs0['y'].data[0], vecs0['z'].data[0]
        # calculate rotation of Gaussian ellipses at time by difference in Jupiter position wrt Sun
    
        # calculate Jupiter coordinates for time
        vecs = calculate_body_coordinates('5',self._time)
        # extract cartesian coordiantes
        _xj, _yj, _zj = vecs['x'].data[0], vecs['y'].data[0], vecs['z'].data[0]    
        # actual distance from Sun
        _rj = vecs['range'].data[0]
        # Lagrange points
        x4,y4,x5,y5 = calculate_Lagrange_points(Ms,Mj,_rj,_xj,_yj)

        # 2D positions
        pos1 = np.array([_xj0,_yj0])
        pos2 = np.array([_xj,_yj])

        # rotation angle
        alpha = np.arccos(np.dot(pos1,pos2)/np.linalg.norm(pos1)/np.linalg.norm(pos2))

        #print('alpha = ',alpha*180/np.pi)

        self.params = [RT,sigmaT,x4,y4,x5,y5,tx,ty,theta4-alpha,theta5-alpha,rhoz1,tz1,rhoz2,tz2]
        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):
            
        # trojans los with rotated thetas
        trojans = Trojans(self.nls._grid_s,
                          self.nls._grid_b,
                          self.nls._grid_l,
                          self.nls._ds,
                          _x,_y,_z,
                          *self.params,
                          luminosity_flag)

        return trojans

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)

    


class Neptunian_Trojans(los):
    """
    Neptunian trojans modelled as combination of Gaussian tori and Gaussian hotspots...
    Scaled version of Jupiter trojans as asteroid content in Neptunian trojans is unknown.
    See XY for details.
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'Neptunian_Trojans'

        self.nls = num_los_setup(smin=0,
                                 smax=100,
                                 n_los_steps=1000,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)

        xT, yT, zT = 0., 0., 0.

        # Neptunian trojan parameters scaled from Jupiter trojans
        # radial components
        RJ = 5.204
        RN = 30.047
        r_NJ = RN/RJ

        # inclination comparison for maximum vertical components
        iJ_median = 11.58
        iN_median = 17.80
        i_NJ = np.tan(np.deg2rad(iN_median))/np.tan(np.deg2rad(iJ_median))
    
        # Jupiter parameters scaled
        K = 	        1.00
        RT = 	        4.9894 * r_NJ
        sigmaT = 	0.278 * r_NJ
        tx = 	        0.974 * r_NJ / np.sqrt(2)
        ty = 	        1.613 * r_NJ / np.sqrt(2)
        theta4 =       -0.487
        theta5 = 	5.115
        tz1 = 	        0.488 * r_NJ * i_NJ / np.sqrt(3)
        tz2 = 	        1.120 * r_NJ * i_NJ / np.sqrt(3)
        rhoz1 = 	793
        rhoz2 =	        793

        # calculate Neptune coordinates for time
        vecs = calculate_body_coordinates('Neptune Barycenter',self._time)

        # extract cartesian coordiantes
        _xn, _yn, _zn = vecs['x'].data[0], vecs['y'].data[0], vecs['z'].data[0]    

        # actual distance from Sun
        # RT = np.sqrt(_xn**2 + _yn**2 + _zn**2)

        # Calculation of Lagrange points (no approximation with 60 deg)
        Mn = 0.01715   # kilo Earth masses
        Ms = 332.950   # kilo Earth masses

        # time for which this has been determined:
        time0 = '2021-11-21 00:00:00.000'

        # calculate Neptune coordinates for time0
        vecs0 = calculate_body_coordinates('8',time0)
        # extract cartesian coordiantes for time 0
        _xn0, _yn0, _zn0 = vecs0['x'].data[0], vecs0['y'].data[0], vecs0['z'].data[0]
        # calculate rotation of Gaussian ellipses at time by difference in Neptune position wrt Sun
    
        # calculate Neptune coordinates for time
        vecs = calculate_body_coordinates('8',self._time)
        # extract cartesian coordiantes
        _xn, _yn, _zn = vecs['x'].data[0], vecs['y'].data[0], vecs['z'].data[0]    
        # actual distance from Sun
        _rn = vecs['range'].data[0]
        # Lagrange points
        x4,y4,x5,y5 = calculate_Lagrange_points(Ms,Mn,_rn,_xn,_yn)

        # 2D positions
        pos1 = np.array([_xn0,_yn0])
        pos2 = np.array([_xn,_yn])

        # rotation angle
        alpha = np.arccos(np.dot(pos1,pos2)/np.linalg.norm(pos1)/np.linalg.norm(pos2))

        #print('alpha = ',alpha*180/np.pi)

        self.params = [RT,sigmaT,x4,y4,x5,y5,tx,ty,theta4-alpha,theta5-alpha,rhoz1,tz1,rhoz2,tz2]
        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):
            
        # trojans los with rotated thetas
        trojans = Trojans(self.nls._grid_s,
                          self.nls._grid_b,
                          self.nls._grid_l,
                          self.nls._ds,
                          _x,_y,_z,
                          *self.params,
                          luminosity_flag)

        return trojans

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)

    

class Torus_Solid(los):
    """
    Homogeneously filled torus (solid torus).
    Input parameters:
    xT: x-coordinate of centre of torus
    yT: y-coordinate of centre of torus
    zT: z-coordinate of centre of torus
    RT: large radius of torus
    rt: small radius of torus
    rho0: normalisation (so far unitless)
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'Solid_Torus'

        self.nls = None
        #num_los_setup(smin=self._smin,
        #                         smax=self._smax,
        #                         n_los_steps=self._n_los_steps,
        #                         lmin=self._phi_min,
        #                         lmax=self._phi_max,
        #                         bmin=self._theta_min,
        #                         bmax=self._theta_max,
        #                         pixelsize=self._pixelsize)

    def density_function(self,_x,_y,_z,**kwargs):
        
        #xT, yT, zT = 0., 0., 0.

        RT = kwargs.pop('RT')
        Rt = kwargs.pop('Rt')
        
        xT = kwargs.pop('xT')
        yT = kwargs.pop('yT')
        zT = kwargs.pop('zT')

        
        
        torus = Solid_Torus(self._PHI_ARR,
                            self._THETA_ARR,
                            _x,_y,_z,
                            xT,yT,zT,
                            RT,Rt)

        rho0 = kwargs.pop('rho0')
        
        return rho0*torus



class Spherical_Shell(los):
    """
    Homogeneously filled spherical shell.
    Input parameters:
    xT: x-coordinate of centre of the shell
    yT: y-coordinate of centre of the shell
    zT: z-coordinate of centre of the shell
    Ri: inner radius
    Ro: outer radius
    rho0: normalisation (so far unitless)

    For Ri = 0, it is a homogeneously filled sphere.
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'Spherical_Shell'

        self.nls = None

    def density_function(self,_x,_y,_z,**kwargs):
        
        Ri = kwargs.pop('Ri')
        Ro = kwargs.pop('Ro')
        
        xT = kwargs.pop('xT')
        yT = kwargs.pop('yT')
        zT = kwargs.pop('zT')
        
        shell = Spherical_Shell_Solid(self._PHI_ARR,
                                      self._THETA_ARR,
                                      _x,_y,_z,
                                      xT,yT,zT,
                                      Ri,Ro)

        rho0 = kwargs.pop('rho0')
        
        return rho0*shell








class Oort_Cloud(los):
    """
    Oort Cloud modelled as spherical Gaussian shell.
    This is a guesstimate based on literature values.
    See XY for details.
    """

    def __init__(self,**kwargs):

        super().__init__(**kwargs)

        self.name = 'Oort_Cloud'

        self.nls = num_los_setup(smin=8000,smax=72000,n_los_steps=400,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)

        # Oort Cloud paramters
        rhoT = 1.
        RT = 40000.
        sigmaT = 8000.
        xT, yT, zT = 0., 0., 0.

        self._params = [xT,yT,zT,RT,sigmaT]
        self._amplitude = rhoT
        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):

        shell = Gaussian_Shell(self.nls._grid_s,
                               self.nls._grid_b,
                               self.nls._grid_l,
                               self.nls._ds,
                               _x,_y,_z,
                               *self._params,
                               luminosity_flag)
    
        return self._amplitude*shell

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)








class Double_Powerlaw_Halo_Profile(los):
    """
    General double power law halo profile, e.g. for dark matter.
    Input parameters:
    xT: x-coordinate of centre of shell
    yT: y-coordinate of centre of shell
    zT: z-coordinate of centre of shell
    alpha:
    beta:
    gamma:
    R0: scale radius of profile
    rho_dm: normalisation (local DM density, for example)
    n: 1 for decay, 2 for annihilation
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'Double_Powerlaw_Halo_Profile'

        self.nls = num_los_setup(smin=self._smin,
                                 smax=self._smax,
                                 n_los_steps=self._n_los_steps,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)

        xT = kwargs.pop('xT')
        yT = kwargs.pop('yT')
        zT = kwargs.pop('zT')

        rho_dm = kwargs.pop('rho_dm')
        R0 = kwargs.pop('R0')
        
        alpha = kwargs.pop('alpha')
        beta = kwargs.pop('beta')
        gamma = kwargs.pop('gamma')
        
        n = kwargs.pop('n')
        
        self._params = [xT,yT,zT,alpha,beta,gamma,R0,rho_dm,n]
        self._amplitude = rho_dm
        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):
        
        halo = NFW_profile_general(self.nls._grid_s,
                                   self.nls._grid_b,
                                   self.nls._grid_l,
                                   self.nls._ds,
                                   _x,_y,_z,
                                   *self._params,
                                   luminosity_flag)
        
        return halo

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)
    

    @property
    def J_factor(self):
        if self._params[-1] == 2:
            self._J_factor = (self.flux*u.Msun**2/u.pc**6*u.kpc*c.c**4).to(u.GeV**2/u.cm**5)/self.dOmega*4*np.pi
            return self._J_factor
        else:
            print('n = {0}, cannot calculate J-factor. Try D-factor?'.format(self._params[-1]))

            
    @property
    def D_factor(self):
        if self._params[-1] == 1:
            self._D_factor = (self.flux*u.Msun/u.pc**3*u.kpc*c.c**2).to(u.GeV**1/u.cm**2)/self.dOmega*4*np.pi
            return self._D_factor
        else:
            print('n = {0}, cannot calculate D-factor. Try J-factor?'.format(self._params[-1]))


    


class NFW_Halo_Milky_Way(los):
    """
    NFW halo of the Milky Way according to paper XY
    n: 1 for decay, 2 for annihilation
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'NFW_Halo_Milky_Way'

        self.nls = num_los_setup(smin=0,
                                 smax=50,
                                 n_los_steps=1000,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)

        # halo parameters
        rho_dm = 0.01069 # Msun/pc3

        R0 = 20. # kpc
                
        alpha = 1
        beta = 3
        gamma = 1

        # Galactic centre
        xT = 8.179 # kpc
        yT = 0
        zT = 0.019 # kpc

        # annihilation or decay
        n = kwargs.pop('n')
        
        self._params = [xT,yT,zT,alpha,beta,gamma,R0,rho_dm,n]
        self._amplitude = rho_dm
        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):
        
        halo = NFW_profile_general(self.nls._grid_s,
                                   self.nls._grid_b,
                                   self.nls._grid_l,
                                   self.nls._ds,
                                   _x,_y,_z,
                                   *self._params,
                                   luminosity_flag)
        
        return halo

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)
    

    @property
    def J_factor(self):
        if self._params[-1] == 2:
            self._J_factor = (self.flux*u.Msun**2/u.pc**6*u.kpc*c.c**4).to(u.GeV**2/u.cm**5)/self.dOmega*4*np.pi
            return self._J_factor
        else:
            print('n = {0}, cannot calculate J-factor. Try D-factor?'.format(self._params[-1]))

            
    @property
    def D_factor(self):
        if self._params[-1] == 1:
            self._D_factor = (self.flux*u.Msun/u.pc**3*u.kpc*c.c**2).to(u.GeV**1/u.cm**2)/self.dOmega*4*np.pi
            return self._D_factor
        else:
            print('n = {0}, cannot calculate D-factor. Try J-factor?'.format(self._params[-1]))

    
                          




class Freudenreich98_Bulge(los):
    """
    Freudenreich 1998 bulge.
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'Freudenreich98_Bulge'

        self.nls = num_los_setup(smin=0,
                                 smax=20,
                                 n_los_steps=400,
                                 lmin=self._phi_min,
                                 lmax=self._phi_max,
                                 bmin=self._theta_min,
                                 bmax=self._theta_max,
                                 pixelsize=self._pixelsize)


        phi = 13.97
        theta = 0
        barX = 1.696
        barY = 0.6426
        barZ = 0.4425
        barPerp = 1.574
        barPara = 3.501
        barREnd = 3.128
        barHEnd = 0.461
        
        # Galactic centre
        xT = 8.179 # kpc
        yT = 0
        zT = 0.019 # kpc

        #
        self._params = [xT,yT,zT,phi,theta,barX,barY,barZ,barPerp,barPara,barREnd,barHEnd]
        
    def density_function(self,_x,_y,_z,luminosity_flag=False,**kwargs):
        
        bar = Tilted_Ellipsoid_general(self.nls._grid_s,
                                       self.nls._grid_b,
                                       self.nls._grid_l,
                                       self.nls._ds,
                                       _x,_y,_z,
                                       *self._params,
                                       luminosity_flag)
        
        return bar

    def luminosity_function(self,_x,_y,_z,luminosity_flag=True,**kwargs):

        return self.density_function(_x,_y,_z,luminosity_flag=luminosity_flag,**kwargs)
    



    



class Empty(los):
    """
    Nothing (just for coordinates, for example)
    """

    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        
        self.name = 'Nothing'

        self.nls = None

    def density_function(self,_x,_y,_z,**kwargs):

        pass
