from astroquery.jplhorizons import Horizons

from numpy import cos, sin, tan, arccos, arcsin, arctan, arctan2, pi, exp, sqrt, cosh, fabs, deg2rad, rad2deg, array
from numpy import arange, linspace, floor, where, float64, ones, count_nonzero

from scipy.interpolate import RectSphereBivariateSpline, RectBivariateSpline

import healpy as hp


def calculate_body_coordinates(body_id,time):

    # getting ephemerides of a specific body (e.g. planet) as observed from the Sun to calculate cartesian coordinates
    obj = Horizons(id=body_id,id_type='majorbody', # 399 = Earth
                   location='@10',               # Sun
                   epochs={'start': time,            # times to include
                           'stop': time + '0001',    #
                           'step':'1d'})

    # get object vectors
    vecs = obj.vectors()

    return vecs


def calculate_body_coordinates_from_other_body(body_id,other_body_id,time):

    # getting ephemerides of a specific body (e.g. planet) as observed from the Sun to calculate cartesian coordinates
    obj = Horizons(id=body_id,id_type='majorbody', # 399 = Earth
                   location=other_body_id,                 # 10 = Sun
                   epochs={'start': time,            # times to include
                           'stop': time + '0001',    #
                           'step':'1d'})

    # get object vectors
    vecs = obj.vectors()
    ephs = obj.ephemerides()
    
    return(vecs,ephs)



def trafo_ec2gal(phi,theta,deg=False):
    """
    Transformation of ecliptic coordinates to galactic coordinates.
    """
    d2r = pi/180
    alpha = 60.188*d2r
    beta = 96.377*d2r
    
    lat = arcsin(-sin(alpha)*cos(phi)*cos(theta) + cos(alpha)*sin(theta))
    
    lon = arctan2(cos(alpha)*sin(beta)*cos(phi)*cos(theta) + sin(alpha)*sin(beta)*sin(theta) + cos(beta)*sin(phi)*cos(theta),
                  cos(alpha)*cos(beta)*cos(phi)*cos(theta) + sin(alpha)*cos(beta)*sin(theta) - sin(beta)*sin(phi)*cos(theta))
    
    if deg == True:
        return lon/d2r,lat/d2r
    else:
        return lon,lat


def trafo_gal2ec(lon,lat,deg=False):
    """
    Transformation of galactic coordinates to ecliptic coordinates.
    """
    d2r = pi/180
    alpha = 60.188*d2r
    beta = 96.377*d2r
    
    theta = arcsin(sin(alpha)*cos(beta)*cos(lon)*cos(lat) + sin(alpha)*sin(beta)*sin(lon)*cos(lat) + cos(alpha)*sin(lat))
    
    phi = arctan2(-sin(beta)*cos(lon)*cos(lat) + cos(beta)*sin(lon)*cos(lat),
                  cos(alpha)*cos(beta)*cos(lon)*cos(lat) + cos(alpha)*sin(beta)*sin(lon)*cos(lat) - sin(alpha)*sin(lat))
    
    if deg == True:
        return phi/d2r,theta/d2r
    else:
        return phi,theta


# misc. functions required for the stuff to work
def polar2cart(ra,dec):
    """
    Coordinate transformation of ra/dec (lon/lat) [phi/theta] polar/spherical coordinates
    into cartesian coordinates
    :param: ra   angle in deg
    :param: dec  angle in deg
    """
    x = cos(np.deg2rad(ra)) * cos(np.deg2rad(dec))
    y = sin(np.deg2rad(ra)) * cos(np.deg2rad(dec))
    z = sin(np.deg2rad(dec))

    return array([x,y,z])


def cart2polar(vector):
    """
    Coordinate transformation of cartesian x/y/z values into spherical (deg)
    :param: vector   vector of x/y/z values
    """
    ra = arctan2(vector[1],vector[0])
    dec = arcsin(vector[2])

    return r2d*ra, r2d*dec



def calculate_Earth_coordinates(time):
    
    # getting ephemerides of Earth as observed from the Sun to calculate cartesian coordinates
    obj = Horizons(id='399',id_type='majorbody', # 399 = Earth
                   location='@10',               # Sun
                   epochs={'start': time,            # times to include
                           'stop': time + '0001',    # 
                           'step':'1d'})

    # get object vectors
    vecs = obj.vectors()

    return vecs




def calculate_Lagrange_points(M1,M2,R,
                              x2,y2):

    reduced_mass = (M1-M2)/(M1+M2)
    
    # unrotated frame
    x4 = R/2*reduced_mass
    y4 = sqrt(3)/2*R

    # not needed because rotated frame depends on object 2
    #x5 = RT/2*reduced_mass
    #y5 = -sqrt(3)/2*RT

    alpha = arctan2(y4,x4)

    # rotated Lagrange points
    x4_r = x2*cos(alpha) + y2*sin(alpha)
    y4_r = -x2*sin(alpha) + y2*cos(alpha)

    x5_r = x2*cos(-alpha) + y2*sin(-alpha)
    y5_r = -x2*sin(-alpha) + y2*cos(-alpha)

    return(x4_r,y4_r,x5_r,y5_r)





def fine_grid_interp(theta,phi,image,scl=4):

    #print(theta.shape)
    #print(phi.shape)
    #print(image.shape)
    
    fine_interp = RectBivariateSpline(theta,
                                      phi,
                                      image,
                                      kx=3,ky=3)
    fine_theta = linspace(theta[0],theta[-1],int(theta.shape[0]*scl))
    fine_phi = linspace(phi[0],phi[-1],int(phi.shape[0]*scl))
    fine_image = fine_interp(fine_theta,fine_phi)

    #print(fine_theta.shape)
    #print(fine_phi.shape)
    #print(fine_image.shape)
    
    return(fine_image,fine_theta,fine_phi)
                



def array2healpix(image,phi,theta, nside=16, max_iter=3, **kwargs):
    """
    Return a healpix ring-ordered map corresponding to a lat-lon map image array.
    Code taken from stackexchange (don't remember where, thank you, person!)
    """

    # Keep track of the number of unseen pixels
    unseen = 1
    ntries = 0
    while unseen > 0:

        pix = hp.ang2pix(nside,(theta+pi/2).reshape(len(theta),1),phi)
        healpix_map = (
            ones(hp.nside2npix(nside), dtype=float64) * hp.UNSEEN
        )
        healpix_map[pix] = image

        # Count the unseen pixels
        unseen = count_nonzero(healpix_map == hp.UNSEEN)

        # Did we do this too many times?
        ntries += 1
        if ntries > max_iter:
            raise ValueError(
                "Maximum number of iterations exceeded. Either decreaser `nside` or increase `max_iter`."
            )

    return(healpix_map)





def change_coord(m, coord):
    """ Change coordinates of a HEALPIX map

    Parameters
    ----------
    m : map or array of maps
      map(s) to be rotated
    coord : sequence of two character
      First character is the coordinate system of m, second character
      is the coordinate system of the output map. As in HEALPIX, allowed
      coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

    Example
    -------
    The following rotate m from galactic to equatorial coordinates.
    Notice that m can contain both temperature and polarization.
    >>>> change_coord(m, ['G', 'C'])

    TS: Code taken from stackexchange (don't remember where, thank you, person!)
    """
    # Basic HEALPix parameters
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, arange(npix))

    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))

    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return(m[..., new_pix])



def get_nside_from_image_pixel_number(image,scl=4):

    npix = image.shape[0]*image.shape[1]

    nside_fractional = sqrt(npix / 12 / scl)

    twobase = 2**arange(24) # maximum 16 Megapixels

    compared_values = floor(nside_fractional / twobase)

    #find where this is equal to one
    idx = where(compared_values == 1)[0][0]

    nside = twobase[idx]
    
    return(nside)



def GreatCircle(l1,b1,l2,b2,deg=True):

    if deg == True:
        l1,b1,l2,b2 = deg2rad(l1),deg2rad(b1),deg2rad(l2),deg2rad(b2)

    return sin(b1)*sin(b2) + cos(b1)*cos(b2)*cos(l1-l2)


def angular_distance(l1,b1,l2,b2,deg=True):
    """
    Calculate angular distance on a sphere from longitude/latitude pairs to other using Great circles
    """
    gc = GreatCircle(l1,b1,l2,b2,deg=deg)

    if gc.size == 1:
        if gc > 1:
            gc = 1.
    else:
        gc[where(gc > 1)] = 1.

    return rad2deg(arccos(gc))
