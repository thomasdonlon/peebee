Sampling From a Model
=====================

This tutorial demonstrates how to generate synthetic pulsar acceleration data from gravitational models. This is essential for testing analysis pipelines, validating fitting procedures, and understanding systematic effects.

Overview
--------

Synthetic data generation allows you to:

- Test optimization algorithms with known "true" parameters
- Estimate parameter uncertainties and biases
- Design observational strategies
- Validate analysis pipelines before applying to real data

Peebee's ``sampling`` module provides tools for generating mock pulsar positions and calculating their corresponding accelerations.

Basic Synthetic Data Generation
-------------------------------

Let's start with the simplest approach: uniformly distributed pulsars in a volume.

Uniform Volume Sampling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from peebee import sampling, models
    from peebee.models import NFW, MiyamotoNagaiDisk
    
    # Create a test model
    halo = NFW(m_vir=1e12, r_s=20.0)
    disk = MiyamotoNagaiDisk(m_tot=5e10, a=3.0, b=0.3)
    test_model = halo + disk
    
    # Define sampling region (Cartesian coordinates in kpc)
    bounds = np.array([
        [4.0, 12.0],   # x: 4-12 kpc (avoid Galactic center, extend beyond Sun)
        [-4.0, 4.0],   # y: ±4 kpc
        [-2.0, 2.0]    # z: ±2 kpc (near Galactic plane)
    ])
    
    # Generate 50 uniformly distributed mock pulsars
    np.random.seed(42)  # for reproducibility
    n_pulsars = 50
    
    x_mock, y_mock, z_mock, alos_mock = sampling.sample_alos_sources_uniform(
        test_model, n_pulsars, bounds
    )
    
    print(f"Generated {n_pulsars} mock pulsars")
    print(f"Position range:")
    print(f"  x: [{np.min(x_mock):.2f}, {np.max(x_mock):.2f}] kpc")
    print(f"  y: [{np.min(y_mock):.2f}, {np.max(y_mock):.2f}] kpc") 
    print(f"  z: [{np.min(z_mock):.2f}, {np.max(z_mock):.2f}] kpc")
    print(f"Acceleration range: [{np.min(alos_mock):.2f}, {np.max(alos_mock):.2f}] mm/s/yr")

Convert to Observational Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert Cartesian coordinates to the Galactic coordinates used in observations:

.. code-block:: python

    from peebee.transforms import cart_to_gal
    
    # Convert to Galactic coordinates
    l_mock, b_mock, d_mock = cart_to_gal(x_mock, y_mock, z_mock)
    
    # Display first 10 pulsars
    print("First 10 mock pulsars:")
    print("   l      b      d     alos")
    print("  (°)    (°)   (kpc) (mm/s/yr)")
    print("-" * 35)
    for i in range(10):
        print(f"{l_mock[i]:6.1f} {b_mock[i]:6.1f} {d_mock[i]:5.2f} {alos_mock[i]:8.2f}")

Sky Distribution Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~

For more realistic distributions, sample uniformly on the sky:

.. code-block:: python

    # Generate pulsars uniformly distributed on the sky within 3 kpc
    np.random.seed(123)
    n_sky = 100
    max_distance = 3.0  # kpc
    
    l_sky, b_sky, d_sky, alos_sky = sampling.sample_sky_uniform(
        test_model, n_sky, max_distance
    )
    
    print(f"Sky uniform sample of {n_sky} pulsars:")
    print(f"  Longitude range: [{np.min(l_sky):.1f}, {np.max(l_sky):.1f}]°")
    print(f"  Latitude range:  [{np.min(b_sky):.1f}, {np.max(b_sky):.1f}]°") 
    print(f"  Distance range:  [{np.min(d_sky):.2f}, {np.max(d_sky):.2f}] kpc")

Realistic Pulsar Distributions
------------------------------

Real pulsars follow more complex spatial distributions that reflect their birthplaces and subsequent evolution.

Power-Law Radial Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many pulsar populations show power-law radial profiles:

.. code-block:: python

    # Power-law distribution: n(r) ∝ r^(-γ)
    # Typical values: γ = 1-2 for pulsar populations
    
    # Center of distribution
    center_of_mass = np.array([8.0, 0.0, 0.0])  # near solar position
    scale_radius = 5.0  # kpc
    gamma = 1.5  # power-law index
    
    n_rpl = 75
    l_rpl, b_rpl, d_rpl, alos_rpl = sampling.sample_alos_sources_RPL(
        test_model, n_rpl, center_of_mass, scale_radius, gamma
    )
    
    print(f"Power-law distribution ({n_rpl} pulsars, γ={gamma}):")
    
    # Calculate distances from center
    from peebee.transforms import gal_to_cart
    x_rpl, y_rpl, z_rpl = gal_to_cart(l_rpl, b_rpl, d_rpl)
    distances_from_center = np.sqrt((x_rpl - center_of_mass[0])**2 + 
                                   (y_rpl - center_of_mass[1])**2 + 
                                   (z_rpl - center_of_mass[2])**2)
    
    print(f"  Distance from center: [{np.min(distances_from_center):.2f}, {np.max(distances_from_center):.2f}] kpc")
    print(f"  Median distance: {np.median(distances_from_center):.2f} kpc")

Adding Observational Noise
---------------------------

Real observations include uncertainties. Add realistic noise to synthetic data:

Gaussian Noise
~~~~~~~~~~~~~~

.. code-block:: python

    # Add Gaussian noise with typical pulsar timing uncertainties
    
    # Typical acceleration uncertainties: 0.1-1.0 mm/s/yr
    # For this example, use 10% relative uncertainty
    relative_uncertainty = 0.1  # 10%
    
    alos_noisy = sampling.perturb_value(
        alos_mock, 
        relative_uncertainty, 
        noise_model='gaussian',
        relative_err=True
    )
    
    # Calculate actual uncertainties used
    uncertainties = relative_uncertainty * np.abs(alos_mock)
    
    print("Added Gaussian noise:")
    print("Original vs Noisy accelerations (first 10):")
    print("  Original    Noisy   Uncertainty  Pull")
    print("  (mm/s/yr)  (mm/s/yr)  (mm/s/yr)    (σ)")
    print("-" * 45)
    for i in range(10):
        pull = (alos_noisy[i] - alos_mock[i]) / uncertainties[i]
        print(f"  {alos_mock[i]:8.2f} {alos_noisy[i]:8.2f} {uncertainties[i]:8.2f} {pull:8.2f}")

Different Noise Models
~~~~~~~~~~~~~~~~~~~~~~

Test different noise characteristics:

.. code-block:: python

    # Compare different noise models
    base_uncertainty = 0.2  # mm/s/yr
    
    # Gaussian noise
    alos_gauss = sampling.perturb_value(alos_mock, base_uncertainty, 'gaussian')
    
    # Lorentzian (heavy-tailed) noise - more outliers  
    alos_lorentz = sampling.perturb_value(alos_mock, base_uncertainty, 'lorentzian')
    
    # Uniform noise
    alos_uniform = sampling.perturb_value(alos_mock, base_uncertainty, 'uniform')
    
    # Compare noise characteristics
    print("Noise model comparison:")
    print("Model      RMS    Min     Max   Outliers(>3σ)")
    for name, values in [('Gaussian', alos_gauss), ('Lorentzian', alos_lorentz), ('Uniform', alos_uniform)]:
        residuals = values - alos_mock
        rms = np.std(residuals)
        min_res, max_res = np.min(residuals), np.max(residuals)
        outliers = np.sum(np.abs(residuals) > 3*base_uncertainty)
        print(f"{name:10s} {rms:5.2f} {min_res:7.2f} {max_res:7.2f} {outliers:8d}")

Data Validation and Quality Cuts
---------------------------------

Apply realistic selection criteria to synthetic data:

Basic Quality Cuts
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Apply typical observational cuts
    
    # 1. Distance cuts (reliable parallax measurements)
    distance_mask = (d_mock > 0.1) & (d_mock < 3.0)  # 0.1-3 kpc
    
    # 2. Signal-to-noise cuts
    snr = np.abs(alos_noisy) / uncertainties
    snr_mask = snr > 2.0  # require 2σ detection
    
    # 3. Exclude low Galactic latitudes (high extinction)
    latitude_mask = np.abs(b_mock) > 5.0  # |b| > 5°
    
    # Combine all cuts
    quality_mask = distance_mask & snr_mask & latitude_mask
    
    print(f"Quality cuts applied:")
    print(f"  Original sample:     {len(alos_mock)} pulsars")
    print(f"  Distance cut:        {np.sum(distance_mask)} pass")
    print(f"  S/N cut:            {np.sum(snr_mask)} pass")
    print(f"  Latitude cut:       {np.sum(latitude_mask)} pass")
    print(f"  All cuts:           {np.sum(quality_mask)} pass")
    print(f"  Selection efficiency: {np.sum(quality_mask)/len(alos_mock)*100:.1f}%")

Parameter Recovery Testing
--------------------------

Use synthetic data to validate analysis pipelines:

.. code-block:: python

    # Test if we can recover the input model parameters
    from peebee import optimize
    from peebee.noise import GaussianNoise
    
    # Extract clean sample
    l_clean = l_mock[quality_mask]
    b_clean = b_mock[quality_mask] 
    d_clean = d_mock[quality_mask]
    alos_clean = alos_noisy[quality_mask]
    alos_err_clean = uncertainties[quality_mask]
    
    # Set up fitter with synthetic data
    fitter = optimize.Fitter()
    fitter.set_model(test_model)  # same model used to generate data
    fitter.set_data(l_clean, b_clean, d_clean, alos_clean, alos_err_clean)
    fitter.set_noise_model(GaussianNoise(sigma=1.0))
    
    # Store original parameters for comparison
    original_params = test_model.params.copy()
    
    # Perturb starting values to test convergence
    perturbed_params = original_params.copy()
    perturbed_params['NFW.m_vir'] *= 1.2  # 20% high
    perturbed_params['NFW.r_s'] *= 0.8    # 20% low
    perturbed_params['Miyamoto-Nagai Disk.m_tot'] *= 1.1  # 10% high
    
    test_model.set_params(perturbed_params)
    
    # Set parameter bounds
    fitter.configure_params({
        'NFW.m_vir': (5e11, 2e12),
        'NFW.r_s': (10.0, 30.0),
        'Miyamoto-Nagai Disk.m_tot': (2e10, 8e10),
        'Miyamoto-Nagai Disk.a': (1.0, 5.0),
    })
    
    print(f"Running parameter recovery test with {len(alos_clean)} clean pulsars...")
    results = fitter.optimize(method='gradient_descent')
    
    print("Parameter recovery test:")
    print("Parameter                    True      Start     Fitted    Recovery")
    print("-" * 70)
    for param in ['NFW.m_vir', 'NFW.r_s', 'Miyamoto-Nagai Disk.m_tot', 'Miyamoto-Nagai Disk.a']:
        true_val = original_params[param]
        start_val = perturbed_params[param]
        fit_val = results.best_fit_params[param]
        recovery = (fit_val - true_val) / true_val * 100
        print(f"{param:25s} {true_val:9.2e} {start_val:9.2e} {fit_val:9.2e} {recovery:7.1f}%")
    
    print(f"\nFit quality: χ² = {results.reduced_chi2:.2f}")

Best Practices
--------------

When creating synthetic datasets:

1. **Match observation characteristics**: Use realistic spatial distributions, uncertainties, and selection effects
2. **Include systematic effects**: Distance uncertainties, calibration errors, model systematics  
3. **Test multiple realizations**: Generate many synthetic datasets to understand statistical variations
4. **Validate pipelines**: Use synthetic data to test analysis methods before applying to real data
5. **Document assumptions**: Keep track of the model parameters and observational assumptions used

Next Steps
----------

With synthetic data in hand, you can:

- :doc:`inference`: Fit models to your synthetic datasets to test parameter recovery
- :doc:`noise_models`: Explore robust fitting techniques for datasets with outliers
- Apply the same analysis pipeline to real pulsar timing data