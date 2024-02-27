"""
Text here for Sphinx (I think)
"""

import numpy as np
from .glob import fix_arrays, r_sun

#stolen from mwahpy
@fix_arrays
def gal_to_cart(l, b, r, left_handed=True, rad=False):
	"""
	Transform from a heliocentric Galactic coordinate system to a Galactocentric Cartesian coordinate system.

	:l: Galactic longitude (deg)
	:b: Galactic latitude (deg)
	:d: Heliocentric distance (kpc)
	"""

	if not rad:
		l = l*np.pi/180
		b = b*np.pi/180

	x = r*np.cos(l)*np.cos(b) + r_sun
	if left_handed:
		x *= -1

	y = r*np.sin(l)*np.cos(b)
	z = r*np.sin(b)

	return x, y, z

#stolen from mwahpy
@fix_arrays
def cart_to_gal(x, y, z, left_handed=True):
	"""
	Transform from a heliocentric Galactic coordinate system to a Galactocentric Cartesian coordinate system.

	:l: Galactic longitude (deg)
	:b: Galactic latitude (deg)
	:d: Heliocentric distance (kpc)
	"""

	if left_handed:
		r = ((x-r_sun)**2 + y**2 + z**2)**0.5
		l = np.arctan2(y,-1*(x-r_sun))*180/np.pi
	else:
		r = ((x+r_sun)**2 + y**2 + z**2)**0.5
		l = np.arctan2(y,(x+r_sun))*180/np.pi
	b = np.arcsin(z/r)*180/np.pi

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
	#fr: the frame ('cart' or 'gal' are currently supported)

	def internal_decorator(func): #this is silly but required I think

		def wrapper(*args, **kwargs):

			#check whether frame kwarg was passed into func
			#frame = kwargs.get('frame') might be the "correct" way of doing this
			try:
				frame = kwargs['frame']
			except KeyError: #not passed
				return func(*args, **kwargs)

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

			elif frame == 'cart': #fr is not cart
				l, b, d = cart_to_gal(args[offset+0], args[offset+1], args[offset+2])

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
				x, y, z = gal_to_cart(l, b, d)
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