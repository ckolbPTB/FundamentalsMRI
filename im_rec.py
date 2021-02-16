##
## Copyright (C) 2021  Christoph Kolbitsch  <christoph.kolbitsch@ptb.de>
##
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.


import numpy as np


def k2i(kdat):
    if len(kdat.shape) == 1: # Carry out 1D FFT: k-space -> image space
       im = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kdat,(0)),
                         (kdat.shape[0],),(0,), norm=None), (0,)))
    else: # Carry  out 2D FFT: k-space -> image space
       im = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kdat,(0,1)),
                         (kdat.shape[0], kdat.shape[1]),(0,1), norm=None), (0,1)))

    return(im)
    
    
def i2k(im):
   if len(im.shape) == 1: # Carry out 1D FFT: image space -> k-space
       kdat = (np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im,(0)),
                      (im.shape[0],),(0,), norm=None), (0,)))
   else: # Carry out 2D FFT: image space -> k-space
       kdat = (np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im,(0,1)),
                      (im.shape[0], im.shape[1]),(0,1), norm=None), (0,1)))

   return(kdat)
   
   
def gauss_filt(dim, sigma):
    if sigma > 1 or sigma <= 0:
        raise KeyError('Sigma is normalised and should be between 0 and 1')
        
    # Create meshgrid based on image dimensions
    gridx, gridy = np.meshgrid(np.linspace(-1., 1., dim[0]), np.linspace(-1., 1., dim[1]))
    gridx = np.array(gridx)
    gridy = np.array(gridy)
    
    # Calculate 2D-Gaussian function
    filt_weights = np.exp(-(gridx**2 + gridy**2)/(2*sigma**2))
    
    return(filt_weights)
    
    
def get_lowres(kdat, low_res_factor):
    if low_res_factor > 1 or low_res_factor <= 0:
        raise KeyError('The factor should be between 0 and 1')
        
    low_res_factor = np.sqrt(low_res_factor)
    
    # Calculate low-res data extend
    kdim = np.round(np.array(kdat.shape) * np.array(low_res_factor))
    
    # Ensure dimensions are even
    kdim = kdim + kdim % 2
    
    # Calculate offset
    koff = np.round((np.array(kdat.shape) - kdim)/2)
    
    # Calculate extend of low-res k-space center
    kx0 = int(koff[0])
    kx1 = int(koff[0]+kdim[0])
    ky0 = int(koff[1])
    ky1 = int(koff[1]+kdim[1])
    
    kdat_lowres = np.zeros(kdat.shape, dtype=kdat.dtype)
    kdat_lowres[kx0:kx1, ky0:ky1] = kdat[kx0:kx1, ky0:ky1]
    
    return(kdat_lowres)
    
    
def create_us_mask(kdim, us_factor):
    if len(us_factor) != 2:
        raise KeyError('The undersampling factors should be a vector of length 2')
    if us_factor[0] < 1 or us_factor[1] < 1:
        raise KeyError('The undersampling factors should be larger than 1')
        
    # Ensure the undersampling factors are integers
    us_factor_kx = int(us_factor[0])
    us_factor_ky = int(us_factor[1])
    
    # Create a binary mask of the undersampling pattern
    us_mask = np.zeros(kdim)
    
    # Undersampling pattern
    us_mask[0::us_factor_kx, 0::us_factor_ky] = 1
    
    return(us_mask)    
    
   
# The following is adapted from:
# https://github.com/pierrepaleo/localtomo
#
def phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
	"""
	 phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)
	
	Create a Shepp-Logan or modified Shepp-Logan phantom.

	A phantom is a known object (either real or purely mathematical) 
	that is used for testing image reconstruction algorithms.  The 
	Shepp-Logan phantom is a popular mathematical model of a cranial
	slice, made up of a set of ellipses.  This allows rigorous 
	testing of computed tomography (CT) algorithms as it can be 
	analytically transformed with the radon transform (see the 
	function `radon').
	
	Inputs
	------
	n : The edge length of the square image to be produced.
	
	p_type : The type of phantom to produce. Either 
	  "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
	  if `ellipses' is also specified.
	
	ellipses : Custom set of ellipses to use.  These should be in 
	  the form
	  	[[I, a, b, x0, y0, phi],
	  	 [I, a, b, x0, y0, phi],
	  	 ...]
	  where each row defines an ellipse.
	  I : Additive intensity of the ellipse.
	  a : Length of the major axis.
	  b : Length of the minor axis.
	  x0 : Horizontal offset of the centre of the ellipse.
	  y0 : Vertical offset of the centre of the ellipse.
	  phi : Counterclockwise rotation of the ellipse in degrees,
	        measured as the angle between the horizontal axis and 
	        the ellipse major axis.
	  The image bounding box in the algorithm is [-1, -1], [1, 1], 
	  so the values of a, b, x0, y0 should all be specified with
	  respect to this box.
	
	Output
	------
	P : A phantom image.
	
	Usage example
	-------------
	  import matplotlib.pyplot as pl
	  P = phantom ()
	  pl.imshow (P)
	
	References
	----------
	Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue 
	from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
	Feb. 1974, p. 232.
	
	Toft, P.; "The Radon Transform - Theory and Implementation", 
	Ph.D. thesis, Department of Mathematical Modelling, Technical 
	University of Denmark, June 1996.
	
	"""
	
	if (ellipses is None):
		ellipses = _select_phantom (p_type)
	elif (np.size (ellipses, 1) != 6):
		raise AssertionError ("Wrong number of columns in user phantom")
	
	# Blank image
	p = np.zeros ((n, n))

	# Create the pixel grid
	ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]

	for ellip in ellipses:
		I   = ellip [0]
		a2  = ellip [1]**2
		b2  = ellip [2]**2
		x0  = ellip [3]
		y0  = ellip [4]
		phi = ellip [5] * np.pi / 180  # Rotation angle in radians
		
		# Create the offset x and y values for the grid
		x = xgrid - x0
		y = ygrid - y0
		
		cos_p = np.cos (phi) 
		sin_p = np.sin (phi)
		
		# Find the pixels within the ellipse
		locs = (((x * cos_p + y * sin_p)**2) / a2 
              + ((y * cos_p - x * sin_p)**2) / b2) <= 1
		
		# Add the ellipse intensity to those pixels
		p [locs] += I

	return(p)


def _select_phantom (name):
	if (name.lower () == 'shepp-logan'):
		e = _shepp_logan ()
	elif (name.lower () == 'modified shepp-logan'):
		e = _mod_shepp_logan ()
	else:
		raise ValueError ("Unknown phantom type: %s" % name)
	
	return(e)


def _shepp_logan ():
	#  Standard head phantom, taken from Shepp & Logan
	return [[   2,   .69,   .92,    0,      0,   0],
	        [-.98, .6624, .8740,    0, -.0184,   0],
	        [-.02, .1100, .3100,  .22,      0, -18],
	        [-.02, .1600, .4100, -.22,      0,  18],
	        [ .01, .2100, .2500,    0,    .35,   0],
	        [ .01, .0460, .0460,    0,     .1,   0],
	        [ .02, .0460, .0460,    0,    -.1,   0],
	        [ .01, .0460, .0230, -.08,  -.605,   0],
	        [ .01, .0230, .0230,    0,  -.606,   0],
	        [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan ():
	#  Modified version of Shepp & Logan's head phantom, 
	#  adjusted to improve contrast.  Taken from Toft.
	return [[   1,   .69,   .92,    0,      0,   0],
	        [-.80, .6624, .8740,    0, -.0184,   0],
	        [-.20, .1100, .3100,  .22,      0, -18],
	        [-.20, .1600, .4100, -.22,      0,  18],
	        [ .10, .2100, .2500,    0,    .35,   0],
	        [ .10, .0460, .0460,    0,     .1,   0],
	        [ .10, .0460, .0460,    0,    -.1,   0],
	        [ .10, .0460, .0230, -.08,  -.605,   0],
	        [ .10, .0230, .0230,    0,  -.606,   0],
	        [ .10, .0230, .0460,  .06,  -.605,   0]]
