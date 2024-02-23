"""
Helper file for peebee, holds the decorators for functions. 
These are used to help automate things like array-like vs float handling in function inputs, alternate between Cartesian and Galactic frames, etc. 
This doesn't need to show up in Sphinx.
"""

#TODO: Needs an automatic astropy converison decorator (check if quantities are being used, and if they are convert to the correct units)
#      Otherwise assume kpc, s, Msun, etc. 

import astropy.units as u
from astropy.coordinates import SkyCoord

#===============================================================================
# DECORATORS
#===============================================================================

#this generically allows functions to take in either arrays or single values
# and turns everything into (numpy) arrays behind the scenes
def fix_arrays(func, *args, **kwargs):
    def wrapper(*args, **kwargs):

        use_array = True
        if not(isinstance(args[0], np.ndarray)): #check whether the first input is an array (assume all inputs are symmetric)
            use_array = False
            args = tuple([np.array([args[i]]) for i in range(len(args))]) #if they aren't, convert the args to arrays

        ret = func(*args, **kwargs) #catches the output from the function

        if not use_array: #convert them back if necessary
            if not(isinstance(ret, tuple)): #check whether we can actually iterate on the returned values (i.e. whether the func returns a single value or multiple)
                ret = ret[0]
            else:
                ret = tuple([ret[i][0] for i in range(len(ret))])

        return ret
    return wrapper

#-----------------------------------------------
# conversion of inputs to desired frame
# (need one for -> gal and -> cart. Shouldn't need the others, but easy to add in the future if we end up needing it)
# should come AFTER fix_arrays (I think)
#-----------------------------------------------

#allows the first 3 inputs to be in different coordinate systems
# based on frame = 'cart' or = 'gal'
def convert_to_frame(func, fr, *args, **kwargs):
    #fr: the frame ('cart' or 'gal' are currently supported)
    def wrapper(*args, **kwargs):

        #check whether frame kwarg was passed into func
        try:
            frame = kwargs['frame']
        except KeyError: #not passed
            return func(*args, **kwargs)

        if frame == fr: #everything is already correct
            return func(*args, **kwargs)

        elif frame == 'icrs' or frame == 'ecl': #d will stay the same

            if frame == 'ecl': #horrid to have to type this out every time; 'ecl' is a shorthand
                frame = 'barycentricmeanecliptic' #no idea if this is the "right" ecliptic frame to use, there are a dozen of them in astropy
                                                  #until someone complains, it's correct
            
            sc = SkyCoord(args[0]*u.deg, args[1]*u.deg, frame=frame)
            coords = sc.galactic
            l = coords.l.value
            b = coords.b.value

        elif frame == 'cart': #fr is not cart
            l, b, d = transforms.cart_to_gal(args[0], args[1], args[2])

        elif frame == 'gal': #fr is not gal
            l, b, d = args[0], args[1], args[2]

        else:
            raise ValueError("Obtained unsupported value for 'frame' keyword. Currently supported options are ['cart', 'gal', 'icrs', 'ecl'].")

        if fr == 'gal':
            args = [l, b, d] + args[3:]
        elif fr == 'cart':
            x, y, z = transforms.gal_to_cart(l, b, d)
            args = [x, y, z] + args[3:]
        else:
            raise ValueError("Developer error! Unsupported value for 'frame' keyword in @convert_to_frame decorator. Currently supported options are ['cart', 'gal'].")

        return func(*args, **kwargs)
    return wrapper
