Calculating Accelerations
==========================

This tutorial demonstrates how to quickly calculate line-of-sight accelerations using peebee. We'll start with Gala's modern Milky Way potential and show the basic usage patterns for acceleration calculations.

Overview
--------

Peebee's core functionality is computing line-of-sight (LOS) accelerations from gravitational potential models. These accelerations can be observed in pulsar timing arrays and provide constraints on Galactic structure and dynamics.

Quick Start with Gala's Milky Way Potential
--------------------------------------------

The easiest way to get started is using Gala's comprehensive ``MilkyWayPotential2022``, which includes realistic disk, halo, bulge, and bar components.

Basic Setup
~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from peebee.models import GalaPotential
    from gala.potential import MilkyWayPotential2022
    
    # Wrap Gala's modern Milky Way potential
    mw_potential = GalaPotential(MilkyWayPotential2022())
    
    print(f"Model components: {mw_potential.name}")
    print(f"Parameters: {list(mw_potential.params.keys())}")

Computing Line-of-Sight Accelerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary method for calculating accelerations is ``alos()``, which computes the line-of-sight component of gravitational acceleration:

.. code-block:: python

    # Define pulsar positions in Galactic coordinates
    l = np.array([30.0, 45.0, 60.0])    # Galactic longitude (degrees)
    b = np.array([10.0, -5.0, 20.0])   # Galactic latitude (degrees) 
    d = np.array([1.5, 2.0, 0.8])     # Heliocentric distance (kpc)
    
    # Calculate line-of-sight accelerations
    alos_values = mw_potential.alos(l, b, d)
    
    print("Pulsar accelerations:")
    for i in range(len(l)):
        print(f"  l={l[i]:5.1f}°, b={b[i]:5.1f}°, d={d[i]:.1f} kpc: "
              f"alos = {alos_values[i]:+7.3f} mm/s/yr")

Understanding Coordinate Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Peebee supports multiple coordinate frames. The ``alos()`` method automatically handles coordinate transformations:

.. code-block:: python

    # Galactic coordinates (default)
    alos_gal = mw_potential.alos(30.0, 10.0, 1.5, frame='gal')
    
    # Cartesian Galactocentric coordinates
    x, y, z = 6.5, 1.0, 0.2  # kpc
    alos_cart = mw_potential.alos(x, y, z, frame='cart')
    
    # Both should give similar results for the same physical position
    print(f"Galactic frame:    alos = {alos_gal:.3f} mm/s/yr")
    print(f"Cartesian frame:   alos = {alos_cart:.3f} mm/s/yr")

Working with Real Data
----------------------

Here's how to process typical pulsar timing data:

.. code-block:: python

    import pandas as pd
    
    # Example: loading pulsar data (replace with your actual data file)
    # Typical format: pulsar name, l, b, distance, observed acceleration, uncertainty
    pulsar_data = {
        'name': ['J1713+0747', 'J1909-3744', 'J0437-4715'],
        'l': [16.33, 359.93, 253.40],
        'b': [24.98, -29.87, -42.28], 
        'd': [1.15, 1.14, 0.156],
        'alos_obs': [1.23, -0.87, 2.45],
        'alos_err': [0.15, 0.12, 0.18]
    }
    df = pd.DataFrame(pulsar_data)
    
    # Calculate predicted accelerations
    alos_pred = mw_potential.alos(df['l'], df['b'], df['d'])
    
    # Compare observations with predictions
    print("\nComparison with observations:")
    print("Pulsar        | Observed  | Predicted | Residual  | Significance")
    print("-" * 65)
    for i, row in df.iterrows():
        residual = row['alos_obs'] - alos_pred[i]
        significance = residual / row['alos_err']
        print(f"{row['name']:12s} | {row['alos_obs']:+8.2f} | {alos_pred[i]:+8.2f} | "
              f"{residual:+8.2f} | {significance:+8.2f}σ")

Understanding Units and Conventions
-----------------------------------

Peebee uses specific units and coordinate conventions:

Units
~~~~~

- **Distances**: kiloparsecs (kpc) 
- **Accelerations**: mm/s/yr (millimeters per second per year)
- **Coordinates**: degrees for angular coordinates (l, b)

.. code-block:: python

    # Unit conversion examples
    alos_mmsyr = 1.23  # mm/s/yr
    
    # Convert to other common units
    alos_cms2 = alos_mmsyr * 3.169e-20  # cm/s²
    alos_mssyr = alos_mmsyr * 1e-3       # m/s/yr
    
    print(f"Acceleration: {alos_mmsyr:.3f} mm/s/yr = {alos_cms2:.2e} cm/s²")

Solar Position
~~~~~~~~~~~~~~

By default, peebee assumes the Sun is at (8.178, 0.0, 0.0) kpc in Galactocentric coordinates. You can modify this using coordinate transformation functions:

.. code-block:: python

    from peebee.transforms import gal_to_cart
    
    # Custom solar position
    custom_sun_pos = (8.2, 0.0, 0.025)  # Recent Gaia-based values
    
    # Convert Galactic to Cartesian with custom solar position
    x, y, z = gal_to_cart(l, b, d, sun_pos=custom_sun_pos)
    alos_custom = mw_potential.alos(x, y, z, frame='cart')
    alos_default = mw_potential.alos(l, b, d)
    
    print(f"Default sun position: {alos_default[0]:.3f} mm/s/yr")
    print(f"Custom sun position:  {alos_custom[0]:.3f} mm/s/yr")
    print(f"Difference: {alos_custom[0] - alos_default[0]:.3f} mm/s/yr")

Array Processing and Performance
--------------------------------

Peebee efficiently handles arrays of positions for bulk calculations:

.. code-block:: python

    # Generate random sky positions for timing
    np.random.seed(42)
    n_pulsars = 1000
    
    # Random positions on sky within 3 kpc
    l_rand = np.random.uniform(0, 360, n_pulsars)
    b_rand = np.random.uniform(-90, 90, n_pulsars) 
    d_rand = np.random.uniform(0.1, 3.0, n_pulsars)
    
    # Time the calculation
    import time
    start_time = time.time()
    alos_bulk = mw_potential.alos(l_rand, b_rand, d_rand)
    elapsed = time.time() - start_time
    
    print(f"Calculated {n_pulsars} accelerations in {elapsed:.3f} seconds")
    print(f"Rate: {n_pulsars/elapsed:.0f} calculations/second")
    
    # Statistical summary
    print(f"\nAcceleration statistics:")
    print(f"  Mean: {np.mean(alos_bulk):+7.3f} mm/s/yr")
    print(f"  Std:  {np.std(alos_bulk):7.3f} mm/s/yr") 
    print(f"  Range: [{np.min(alos_bulk):+6.2f}, {np.max(alos_bulk):+6.2f}] mm/s/yr")

Working with Individual Components
----------------------------------

You can access individual components of the Gala potential:

.. code-block:: python

    # Access individual potential components
    mw2022 = MilkyWayPotential2022()
    
    # Create models for individual components
    disk_model = GalaPotential(mw2022['disk'])
    halo_model = GalaPotential(mw2022['halo']) 
    bulge_model = GalaPotential(mw2022['bulge'])
    
    # Test position
    l, b, d = 45.0, 10.0, 1.5
    
    # Component accelerations
    alos_disk = disk_model.alos(l, b, d)
    alos_halo = halo_model.alos(l, b, d)
    alos_bulge = bulge_model.alos(l, b, d)
    alos_total = mw_potential.alos(l, b, d)
    
    print(f"Component accelerations at l={l}°, b={b}°, d={d} kpc:")
    print(f"  Disk:   {alos_disk:+7.3f} mm/s/yr")
    print(f"  Halo:   {alos_halo:+7.3f} mm/s/yr")
    print(f"  Bulge:  {alos_bulge:+7.3f} mm/s/yr")
    print(f"  Total:  {alos_total:+7.3f} mm/s/yr")
    print(f"  Sum:    {alos_disk + alos_halo + alos_bulge:+7.3f} mm/s/yr")

Pulsar Timing Physics Context
-----------------------------

While this tutorial focuses on acceleration calculations, it's helpful to understand how these relate to pulsar timing observations.

The Shklovskii Effect
~~~~~~~~~~~~~~~~~~~~~

Proper motion of pulsars induces an apparent change in pulse period:

.. code-block:: python

    from peebee import accels
    
    # Example pulsar parameters
    period = 0.033  # seconds (J0437-4715)
    proper_motion = 121.4  # mas/yr
    distance = 0.156  # kpc
    
    # Calculate Shklovskii contribution to period derivative
    pdot_shk = accels.pdot_shk(period, proper_motion, distance)
    
    # Convert to line-of-sight acceleration contribution
    alos_shk = accels.alos_spin_gal(distance, period, pdot_shk, proper_motion)
    
    print(f"Shklovskii effect:")
    print(f"  Period derivative: {pdot_shk:.2e} s/s")
    print(f"  LOS acceleration:  {alos_shk:.3f} mm/s/yr")

This shows how proper motion creates apparent accelerations that must be accounted for when using pulsar timing data to constrain Galactic potentials.

Next Steps
----------

Now that you can calculate accelerations from existing models, learn how to:

- :doc:`building_a_model`: Create custom potential models and combine components
- :doc:`sampling_from_a_model`: Generate synthetic data for testing and validation
- :doc:`inference`: Fit potential models to observed acceleration data
- :doc:`noise_models`: Handle realistic noise and uncertainties in your analysis

