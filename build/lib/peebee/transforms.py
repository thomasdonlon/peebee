"""
This submodule is home to coordinate transformations that help improve the general user experience. 
"""

import numpy as np
from functools import wraps 
from .glob import fix_arrays, r_sun
from astropy.coordinates import SkyCoord
import astropy.units as u

#stolen from mwahpy
@fix_arrays
def gal_to_cart(l, b, r, left_handed=True, rad=False, sun_pos=(r_sun, 0., 0.)):
	"""
	Transform from a heliocentric Galactic coordinate system to a Galactocentric Cartesian coordinate system.

	:l: Galactic longitude (deg)
	:b: Galactic latitude (deg)
	:d: Heliocentric distance (kpc)

	"""

	if not rad:
		l = l*np.pi/180
		b = b*np.pi/180

	x = r*np.cos(l)*np.cos(b)
	if left_handed:
		x *= -1
	x += sun_pos[0] #this is broken TODO

	y = r*np.sin(l)*np.cos(b) + sun_pos[1]
	z = r*np.sin(b) + sun_pos[2]

	return x, y, z

#stolen from mwahpy
@fix_arrays
def cart_to_gal(x, y, z, left_handed=True, sun_pos=(r_sun, 0., 0.)):
	"""
	Transform from a heliocentric Galactic coordinate system to a Galactocentric Cartesian coordinate system.

	:l: Galactic longitude (deg)
	:b: Galactic latitude (deg)
	:d: Heliocentric distance (kpc)

	"""

	if left_handed:
		r = ((x-sun_pos[0])**2 + (y-sun_pos[1])**2 + (z-sun_pos[2])**2)**0.5
		l = np.arctan2((y-sun_pos[1]),-1*(x-sun_pos[0]))*180/np.pi
	else:
		r = ((x+sun_pos[0])**2 + (y-sun_pos[1])**2 + (z-sun_pos[2])**2)**0.5
		l = np.arctan2((y-sun_pos[1]),(x+sun_pos[0]))*180/np.pi
	b = np.arcsin((z-sun_pos[2])/r)*180/np.pi

	return l, b, r

#===============================================================================
# DECORATORS
#===============================================================================

#-----------------------------------------------
# conversion of inputs to desired frame
# (need one for -> gal and -> cart. Shouldn't need the others, but easy to add in the future if we end up needing it)
# should come AFTER fix_arrays (I think)
#-----------------------------------------------

#allows the first 3 inputs to be in different coordinate systems
# based on frame = 'cart' or = 'gal'
def convert_to_frame(fr):
	""":meta private:"""
	#fr: the frame ('cart' or 'gal' are currently supported)

	def internal_decorator(func): #this is silly but required I think

		@wraps(func)
		def wrapper(*args, **kwargs):

			#check whether frame kwarg was passed into func
			#frame = kwargs.get('frame') might be the "correct" way of doing this
			try:
				frame = kwargs['frame']
			except KeyError: #not passed
				return func(*args, **kwargs)

			#check whether sun_pos kwarg was passed into func
			try:
				sun_pos = kwargs['sun_pos']
			except KeyError: #not passed
				sun_pos = (r_sun, 0., 0.) #peebee default

			#the "correct" way of checking if func is a method is broken so let's do it the "naive" way
			offset = 0
			try:
				test = np.array(args[0], dtype=float)
			except TypeError: #args[0] is __self__ (probably)
				offset = 1

			if frame == fr: #everything is already correct
				return func(*args, **kwargs)

			elif frame == 'icrs' or frame == 'ecl': #d will stay the same

				if frame == 'ecl': #horrid to have to type this out every time; 'ecl' is a shorthand
					frame = 'barycentricmeanecliptic' #no idea if this is the "right" ecliptic frame to use, there are a dozen of them in astropy
													  #until someone complains, it's correct
				
				sc = SkyCoord(args[offset+0]*u.deg, args[offset+1]*u.deg, frame=frame)
				coords = sc.galactic
				l = coords.l.value
				b = coords.b.value
				d = args[offset+2]

			elif frame == 'cart': #fr is not cart
				l, b, d = cart_to_gal(args[offset+0], args[offset+1], args[offset+2], sun_pos=sun_pos)

			elif frame == 'gal': #fr is not gal
				l, b, d = args[offset+0], args[offset+1], args[offset+2]

			else:
				raise ValueError("Obtained unsupported value for 'frame' keyword. Currently supported options are ['cart', 'gal', 'icrs', 'ecl'].")

			#sorry I know this is gross
			if fr == 'gal':
				tmp_args = [l, b, d] + list(args[offset+3:])
				if offset:
					args = [args[0]] + tmp_args
				else:
					args = tmp_args
			elif fr == 'cart':
				x, y, z = gal_to_cart(l, b, d, sun_pos=sun_pos)
				tmp_args = [x, y, z] + list(args[offset+3:])
				if offset:
					args = [args[0]] + tmp_args
				else:
					args = tmp_args
			else:
				raise ValueError("Developer error! Unsupported value for 'frame' keyword in @convert_to_frame decorator. Currently supported options are ['cart', 'gal'].")

			return func(*args, **kwargs)
		return wrapper
	return internal_decorator
