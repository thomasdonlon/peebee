PEEBEE
========================================

Copyright Tom Donlon, 2024 UAH

github user: thomasdonlon

Requires Python v.>3.6.0

-----------------------------------------

A python package for the intersection between pulsar timing, accelerations, and Galactic structure.
Currently, this package mainly includes convenience functions for various things I have needed to do 
in pulsar acceleration research. This package will also form the foundation for the pulsars.uah.edu backend. 

CONTENTS
========================================

A non-exhaustive list of contents of the package is given below:

 - Ability to compute accelerations (including line-of-sight accelerations) for user-defined potentials 
 (including support for gala and galpy potentials)
 - Convenience functions for computing Pdots, including the Shklovskii effect, Relativistic orbital decay, Galactic accelerations, etc. 

INSTALLATION
========================================

FOR USERS:

1. Open your terminal, and run

> python3 -m pip install peebee

FOR DEVELOPERS:

1. Clone the peebee github repository

2. Make the desired changes in the source code

3. Navigate to the directory where you cloned the repo, and then run

> python3 setup.py develop --user

(note, you will probably need to uninstall any previous versions of peebee you had on your machine before running this)

4. To test your changes, insert import statements for the subpackages that you want to use in your .py files as you normally would:

> import peebee.{peebee subpackage}

> ...

5. Once you are done making changes to the source code, put in a pull request to master

6. Navigate to the directory where you cloned the repo, and then run

> python3 setup.py develop --uninstall

> pip3 install peebee

Your changes will not be available in the main peebee build until a new release comes out.

TODO
========================================

MAJOR:
 - Add support for astropy units

MINOR:
 - Agama potentials?

ISSUES
========================================

- Currently, the most up-to-date versions of astropy (v6.1.0) and gala (v1.8.1) result in an error being produced when computing accelerations using Gala potentials. This can be resolved by downgrading astropy to v5.3.0. 
