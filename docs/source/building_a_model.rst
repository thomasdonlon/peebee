Building a Potential/Acceleration Model
=========================================

This tutorial shows how to construct gravitational potential models using peebee's native components, combine them into realistic galaxies, and integrate with external libraries like Gala and Galpy.

Overview
--------

Peebee provides a library of analytic potential models that can be combined to represent realistic galactic systems. Each model implements the core interface for calculating accelerations and can be composed with others using simple arithmetic operators.

Individual Model Components
---------------------------

Let's start by exploring peebee's built-in models and their typical use cases.

Dark Matter Halos
~~~~~~~~~~~~~~~~~~

The Navarro-Frenk-White (NFW) profile is the standard model for dark matter halos:

.. code-block:: python

    import numpy as np
    from peebee.models import NFW
    
    # Create NFW halo with typical Milky Way parameters
    halo = NFW(m_vir=1.2e12, r_s=18.0)  # virial mass, scale radius
    
    print(f"Halo model: {halo.name}")
    print(f"Parameters: {halo.params}")
    print(f"Number of parameters: {halo.nparams}")
    
    # Test acceleration calculation at solar position
    r_sun = 8.178  # kpc
    alos_sun = halo.alos(0.0, 0.0, r_sun, frame='cart')
    print(f"Halo acceleration at Sun: {alos_sun:.3f} mm/s/yr")

Other halo models available:

.. code-block:: python

    from peebee.models import Hernquist, SphericalFlatRC
    
    # Hernquist profile (often used for bulges)
    hernquist = Hernquist(m=2e10, a=0.7)  # mass, scale radius
    
    # Spherical model with flat rotation curve
    flat_halo = SphericalFlatRC(v_0=220.0, r_0=8.0)  # circular velocity, scale radius
    
    # Test at same position
    l, b, d = 45.0, 10.0, 1.5  # Galactic coordinates
    
    alos_nfw = halo.alos(l, b, d)
    alos_hern = hernquist.alos(l, b, d) 
    alos_flat = flat_halo.alos(l, b, d)
    
    print(f"Acceleration comparison at l={l}°, b={b}°, d={d} kpc:")
    print(f"  NFW:       {alos_nfw:+7.3f} mm/s/yr")
    print(f"  Hernquist: {alos_hern:+7.3f} mm/s/yr")
    print(f"  Flat RC:   {alos_flat:+7.3f} mm/s/yr")

Galactic Disks
~~~~~~~~~~~~~~~

Several disk models are available for representing stellar and gas disks:

.. code-block:: python

    from peebee.models import MiyamotoNagaiDisk, ExponentialDisk
    
    # Miyamoto-Nagai disk (most common)
    mn_disk = MiyamotoNagaiDisk(m_tot=5e10, a=3.0, b=0.3)  # mass, radial scale, vertical scale
    
    # Exponential disk profile
    exp_disk = ExponentialDisk(m_tot=4e10, h_R=2.8, h_z=0.3)  # mass, radial scale height, vertical scale height
    
    print(f"Disk models available:")
    print(f"  MN disk: {mn_disk.name} ({mn_disk.nparams} params)")
    print(f"  Exp disk: {exp_disk.name} ({exp_disk.nparams} params)")
    
    # Test accelerations at different heights above disk
    z_heights = [0.0, 0.5, 1.0, 2.0]  # kpc
    print(f"\nDisk acceleration vs height (at R = 8 kpc):")
    for z in z_heights:
        alos_mn = mn_disk.alos(0.0, 0.0, 8.0 + z, frame='cart')
        alos_exp = exp_disk.alos(0.0, 0.0, 8.0 + z, frame='cart') 
        print(f"  z={z:.1f} kpc: MN = {alos_mn:+7.3f}, Exp = {alos_exp:+7.3f} mm/s/yr")

Point Masses and Simple Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Useful for central black holes and testing:

.. code-block:: python

    from peebee.models import PointMass
    
    # Sagittarius A* supermassive black hole
    sgrA_star = PointMass(m=4.15e6)  # solar masses
    
    # Acceleration at different distances from Galactic center
    distances = [0.001, 0.01, 0.1, 1.0]  # kpc (1 pc to 1 kpc)
    print(f"Sgr A* acceleration vs distance:")
    for d in distances:
        alos_bh = sgrA_star.alos(0.0, 0.0, d, frame='cart')
        print(f"  {d*1000:6.1f} pc: {alos_bh:+10.3f} mm/s/yr")

Creating Composite Models
-------------------------

Real galaxies require multiple components. Peebee makes composition intuitive:

Basic Two-Component Galaxy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from peebee.models import NFW, MiyamotoNagaiDisk
    
    # Define individual components
    dark_halo = NFW(m_vir=1.2e12, r_s=18.0)
    stellar_disk = MiyamotoNagaiDisk(m_tot=4.5e10, a=2.8, b=0.28)
    
    # Combine using the + operator
    simple_galaxy = dark_halo + stellar_disk
    
    print(f"Composite model: {simple_galaxy.name}")
    print(f"Total parameters: {simple_galaxy.nparams}")
    
    # Parameter names are automatically prefixed to avoid conflicts
    print(f"Parameter names:")
    for name in simple_galaxy.param_names:
        print(f"  {name}")

The composite model automatically handles:

- Parameter name prefixing (``NFW.m_vir``, ``Miyamoto-Nagai Disk.m_tot``)
- Combined acceleration calculations
- Parameter access and modification

Working with Composite Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # View all current parameter values
    print("Current parameter values:")
    for name, value in simple_galaxy.params.items():
        print(f"  {name}: {value}")
    
    # Update specific parameters
    simple_galaxy.set_params({
        'NFW.m_vir': 1.0e12,
        'NFW.r_s': 20.0,
        'Miyamoto-Nagai Disk.m_tot': 5.0e10,
        'Miyamoto-Nagai Disk.a': 3.2
    })
    
    # Test the updated model
    test_positions = [(30, 10, 1.5), (120, -20, 2.0), (240, 45, 0.8)]
    
    print(f"\nAccelerations with updated parameters:")
    for l, b, d in test_positions:
        alos = simple_galaxy.alos(l, b, d)
        print(f"  l={l:3.0f}°, b={b:3.0f}°, d={d:.1f} kpc: {alos:+7.3f} mm/s/yr")

Complex Multi-Component Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For realistic modeling, you might need many components:

.. code-block:: python

    from peebee.models import NFW, MiyamotoNagaiDisk, PointMass, Hernquist
    
    # Build a complete galactic model
    dark_halo = NFW(m_vir=1.2e12, r_s=18.0)
    stellar_disk = MiyamotoNagaiDisk(m_tot=4.5e10, a=2.8, b=0.28) 
    gas_disk = MiyamotoNagaiDisk(m_tot=1.2e10, a=3.5, b=0.085)
    bulge = Hernquist(m=2.0e10, a=0.7)
    central_bh = PointMass(m=4.15e6)
    
    # Combine all components
    realistic_mw = dark_halo + stellar_disk + gas_disk + bulge + central_bh
    
    print(f"Complete model: {realistic_mw.name}")
    print(f"Components: {len(realistic_mw.name.split(' + '))}")
    print(f"Total parameters: {realistic_mw.nparams}")

Component Analysis
~~~~~~~~~~~~~~~~~~

You can analyze contributions from individual components:

.. code-block:: python

    # Test position in the solar neighborhood
    l, b, d = 45.0, 0.0, 1.0
    
    # Calculate individual contributions
    alos_halo = dark_halo.alos(l, b, d)
    alos_sdisk = stellar_disk.alos(l, b, d)
    alos_gdisk = gas_disk.alos(l, b, d)
    alos_bulge = bulge.alos(l, b, d)
    alos_bh = central_bh.alos(l, b, d)
    alos_total = realistic_mw.alos(l, b, d)
    
    print(f"Component breakdown at l={l}°, b={b}°, d={d} kpc:")
    print(f"  Dark halo:    {alos_halo:+8.3f} mm/s/yr ({alos_halo/alos_total*100:5.1f}%)")
    print(f"  Stellar disk: {alos_sdisk:+8.3f} mm/s/yr ({alos_sdisk/alos_total*100:5.1f}%)")
    print(f"  Gas disk:     {alos_gdisk:+8.3f} mm/s/yr ({alos_gdisk/alos_total*100:5.1f}%)")
    print(f"  Bulge:        {alos_bulge:+8.3f} mm/s/yr ({alos_bulge/alos_total*100:5.1f}%)")
    print(f"  Central BH:   {alos_bh:+8.3f} mm/s/yr ({alos_bh/alos_total*100:5.1f}%)")
    print(f"  Total:        {alos_total:+8.3f} mm/s/yr")

Parameter Space Management
--------------------------

For optimization and analysis, proper parameter management is crucial.

Parameter Scaling
~~~~~~~~~~~~~~~~~

Many parameters span orders of magnitude. Use logarithmic scaling for masses:

.. code-block:: python

    # Enable logarithmic scaling for mass parameters
    realistic_mw.toggle_log_params([
        'NFW.m_vir', 
        'Miyamoto-Nagai Disk.m_tot',
        'Miyamoto-Nagai Disk 2.m_tot',  # gas disk (second MN disk)
        'Hernquist.m',
        'Point Mass.m'
    ])
    
    print("Parameters with log scaling:")
    for name in realistic_mw.param_names:
        is_log = name in realistic_mw.log_params
        value = realistic_mw.params[name]
        if is_log:
            print(f"  {name}: {value:.3f} (log₁₀)")
        else:
            print(f"  {name}: {value:.3f} (linear)")

Parameter Bounds
~~~~~~~~~~~~~~~~

Define reasonable ranges for parameters:

.. code-block:: python

    # Define typical parameter bounds for optimization
    parameter_bounds = {
        # Halo parameters (log-scaled masses)
        'NFW.m_vir': (11.0, 12.5),     # 10¹¹ to 10¹² M☉
        'NFW.r_s': (10.0, 30.0),       # kpc
        
        # Stellar disk parameters  
        'Miyamoto-Nagai Disk.m_tot': (10.5, 11.0),  # 10¹⁰.⁵ to 10¹¹ M☉
        'Miyamoto-Nagai Disk.a': (2.0, 4.0),        # kpc
        'Miyamoto-Nagai Disk.b': (0.1, 0.5),        # kpc
        
        # Gas disk parameters
        'Miyamoto-Nagai Disk 2.m_tot': (9.5, 10.5), # 10⁹.⁵ to 10¹⁰.⁵ M☉
        'Miyamoto-Nagai Disk 2.a': (2.5, 5.0),      # kpc
        'Miyamoto-Nagai Disk 2.b': (0.05, 0.2),     # kpc
        
        # Bulge parameters
        'Hernquist.m': (9.5, 10.5),    # 10⁹.⁵ to 10¹⁰.⁵ M☉
        'Hernquist.a': (0.3, 1.0),     # kpc
    }
    
    print("Suggested parameter bounds for fitting:")
    for name, (low, high) in parameter_bounds.items():
        current = realistic_mw.params.get(name, "N/A")
        print(f"  {name}: [{low:.1f}, {high:.1f}] (current: {current})")

External Library Integration
----------------------------

Peebee can wrap potentials from Gala and Galpy for optimization.

Gala Integration
~~~~~~~~~~~~~~~~

Use state-of-the-art models from the Gala library:

.. code-block:: python

    try:
        from gala.potential import MilkyWayPotential2022, NFWPotential
        from peebee.models import GalaPotential
        
        # Wrap the full MilkyWayPotential2022
        mw2022 = GalaPotential(MilkyWayPotential2022())
        
        print(f"Gala model: {mw2022.name}")
        print(f"Parameters: {len(mw2022.params)} total")
        
        # You can also wrap individual components
        mw_components = MilkyWayPotential2022()
        
        gala_halo = GalaPotential(mw_components['halo'])
        gala_disk = GalaPotential(mw_components['disk'])
        gala_bulge = GalaPotential(mw_components['bulge'])
        
        # Combine Gala components with native peebee models
        hybrid_model = gala_halo + stellar_disk  # Gala halo + peebee disk
        
        print(f"Hybrid model: {hybrid_model.name}")
        
        # Test calculations
        l, b, d = 30.0, 15.0, 1.8
        alos_gala = mw2022.alos(l, b, d)
        alos_hybrid = hybrid_model.alos(l, b, d)
        
        print(f"Acceleration comparison:")
        print(f"  Full Gala MW2022: {alos_gala:+7.3f} mm/s/yr")
        print(f"  Hybrid model:     {alos_hybrid:+7.3f} mm/s/yr")
        
    except ImportError:
        print("Gala not installed - skipping Gala examples")

Galpy Integration
~~~~~~~~~~~~~~~~~

Similarly for Galpy models:

.. code-block:: python

    try:
        from galpy.potential import MWPotential2014, NFWPotential, MiyamotoNagaiPotential
        from peebee.models import GalpyPotential
        
        # Wrap Galpy's MWPotential2014
        galpy_mw = GalpyPotential(MWPotential2014)
        
        print(f"Galpy model: {galpy_mw.name}")
        
        # Individual Galpy components
        galpy_nfw = GalpyPotential(NFWPotential(amp=1.0, a=16.0))
        galpy_disk = GalpyPotential(MiyamotoNagaiPotential(amp=0.6, a=3.0, b=0.28))
        
        # Test at solar position
        alos_galpy = galpy_mw.alos(0.0, 0.0, 8.0, frame='cart')
        print(f"Galpy MW2014 at solar position: {alos_galpy:+7.3f} mm/s/yr")
        
    except ImportError:
        print("Galpy not installed - skipping Galpy examples")

Model Validation and Testing
----------------------------

Before using models for science, validate them:

Symmetry Tests
~~~~~~~~~~~~~~

.. code-block:: python

    # Test spherical symmetry for NFW halo
    halo = NFW(m_vir=1e12, r_s=20.0)
    
    r = 10.0  # kpc
    test_positions = [
        (r, 0, 0),    # x-axis
        (0, r, 0),    # y-axis  
        (0, 0, r),    # z-axis
        (r/np.sqrt(2), r/np.sqrt(2), 0)  # diagonal
    ]
    
    print("Spherical symmetry test for NFW halo:")
    for i, (x, y, z) in enumerate(test_positions):
        alos = halo.alos(x, y, z, frame='cart')
        print(f"  Position {i+1}: ({x:4.1f}, {y:4.1f}, {z:4.1f}) → {alos:+8.3f} mm/s/yr")

Comparison with Literature
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compare with known results at solar position
    # Literature value: ~220 km/s circular velocity at R=8 kpc
    
    R_sun = 8.178  # kpc
    
    # Create a model matching literature parameters
    lit_halo = NFW(m_vir=1.0e12, r_s=16.0)
    lit_disk = MiyamotoNagaiDisk(m_tot=6.5e10, a=3.0, b=0.28)
    lit_model = lit_halo + lit_disk
    
    # Calculate acceleration at solar radius
    alos_lit = lit_model.alos(0.0, 0.0, R_sun, frame='cart')
    
    # Convert to circular velocity for comparison
    # alos = v²/R for circular motion
    v_circ = np.sqrt(abs(alos_lit) * R_sun * 1.023e-6)  # conversion factor
    
    print(f"Model validation:")
    print(f"  Acceleration at R={R_sun} kpc: {alos_lit:+7.3f} mm/s/yr")
    print(f"  Implied circular velocity: {v_circ:.0f} km/s")
    print(f"  Literature expectation: ~220 km/s")

Best Practices
--------------

When building models for scientific analysis:

1. **Start simple**: Begin with basic two-component models before adding complexity
2. **Use physical priors**: Set parameter bounds based on observational constraints
3. **Test thoroughly**: Validate models against known results and symmetries  
4. **Document parameters**: Keep track of parameter choices and their sources
5. **Consider degeneracies**: Some parameters may be correlated (e.g., mass and scale radius)

Next Steps
----------

Now that you can build custom models:

- :doc:`sampling_from_a_model`: Generate synthetic data to test your models
- :doc:`inference`: Fit your models to real observational data
- :doc:`noise_models`: Handle realistic uncertainties in your analysis