"""
Helper file for peebee, holds the general utilities for functions elsewhere in peebee, plus constants that we want to reference across the package. 
This doesn't need to show up in Sphinx.
"""

#TODO: Needs an automatic astropy converison decorator (check if quantities are being used, and if they are convert to the correct units)
#      Otherwise assume kpc, s, Msun, etc. 

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
#import inspect

#===============================================================================
# CONSTANTS
#===============================================================================

r_sun = 8.0 #kpc
kpcs2tommsyr = 9.731e29 
kpctocm = 3.086e21 # convert kpc to cm

#===============================================================================
# DECORATORS
#===============================================================================

#this generically allows functions to take in either arrays or single values
# and turns everything into (numpy) arrays behind the scenes
def fix_arrays(func):
    def wrapper(*args, **kwargs):

        use_array = True

        #this *should* work but python3 is difficult with how it exactly passes (bound) methods
        #check if func is method and correct for that (can't turn __self__ into an array)
        # check_arg = 0
        # if inspect.ismethod(func): #first arg is self
        #     check_arg = 1

        #so let's do it the "naive" way
        check_arg = 0
        try:
            test = np.array(args[0], dtype=float)
        except TypeError: #args[0] is __self__ (probably)
            check_arg = 1

        if not(isinstance(args[check_arg], np.ndarray)): #check whether the first input is an array (assume all inputs are symmetric)
            use_array = False
            tmp_args = [np.array([args[i+check_arg]], dtype=float) for i in range(len(args)-check_arg)] #if they aren't, convert the args to arrays
            if check_arg: #sloppy but it works
                tmp_args = [args[0]] + tmp_args
            args = tuple(tmp_args)

        ret = func(*args, **kwargs) #catches the output from the function

        if not use_array: #convert them back if necessary
            if not(isinstance(ret, tuple)): #check whether we can actually iterate on the returned values (i.e. whether the func returns a single value or multiple)
                ret = ret[0]
            else:
                ret = tuple([ret[i][0] for i in range(len(ret))])

        return ret
    return wrapper

