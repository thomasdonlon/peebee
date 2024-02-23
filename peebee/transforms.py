"""
Text here for Sphinx (I think)
"""

import numpy
from .decorators import fix_arrays

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

    if left_handed:
        #left-handed
        x = -1*r_sun - r*np.cos(l)*np.cos(b)
    else:
        #right-handed
        x = r*np.cos(l)*np.cos(b) + r_sun

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
        r = ((x+r_sun)**2 + y**2 + z**2)**0.5
        l = np.arctan2(y,-1*(x-8))*180/np.pi
    else:
        r = ((x-r_sun)**2 + y**2 + z**2)**0.5
        l = np.arctan2(y,(x+8))*180/np.pi
    b = np.arcsin(z/r)*180/np.pi

    return l, b, r