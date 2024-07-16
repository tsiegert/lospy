import numpy as np

class num_los_setup():

    def __init__(self,
                 pixelsize=3.,
                 lmin=-180.,
                 lmax=+180.,
                 bmin=-90.,
                 bmax=+90.,
                 smin=0.,
                 smax=10.,
                 n_los_steps=400):

        self._pixelsize = pixelsize
        
        # definition of image space
        # minmax range (similar to above but now also including s)
        self._lmin, self._lmax = lmin, lmax
        self._bmin, self._bmax = bmin, bmax

        self._n_l = int((self._lmax-self._lmin)/self._pixelsize)
        self._n_b = int((self._bmax-self._bmin)/self._pixelsize)


        # integration boundaries for los
        self._smin, self._smax = smin, smax

        # use resolutin useful for case
        # driven here empirical
        self._n_los_steps = n_los_steps

        # define lon, lat, and los arrays, with and without boundaries (sky dimensions)
        self._s = np.linspace(self._smin,self._smax,self._n_los_steps)
        self._ds = np.diff(self._s)[0] # los element for regularly-spaced bins, might be adapted later
        self._bg = np.linspace(self._bmin,self._bmax,self._n_b+1)
        self._lg = np.linspace(self._lmin,self._lmax,self._n_l+1)
        self._b = (self._bg[1:]+self._bg[0:-1])/2
        self._l = (self._lg[1:]+self._lg[0:-1])/2

        # 3D grid for los
        self._grid_s, self._grid_b, self._grid_l = np.meshgrid(self._s,self._b,self._l,indexing="ij")
        

    @property
    def s(self):
        return self._s

    @property
    def l(self):
        return self._l

    @property
    def b(self):
        return self._b

    @property
    def n_s(self):
        return self._n_los_steps

    @property
    def n_l(self):
        return self._n_l

    @property
    def n_b(self):
        return self._n_b

    @property
    def ds(self):
        return self._ds

    @property
    def smin(self):
        return self._smin

    @property
    def smax(self):
        return self._smax
