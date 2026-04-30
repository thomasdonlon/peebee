Optimizing a Potential Model to Acceleration Data
==================================================

This tutorial covers how to fit gravitational potential models to pulsar acceleration data using peebee's optimization framework. We'll explore different optimization algorithms, interpret results, and handle real observational challenges.

Overview
--------

Parameter inference in peebee follows a standard workflow:

1. **Set up the model**: Define the gravitational potential to fit
2. **Load the data**: Provide pulsar positions, accelerations, and uncertainties  
3. **Configure optimization**: Set parameter bounds, choose algorithms, add noise models
4. **Run fitting**: Execute the optimization and obtain best-fit parameters
5. **Interpret results**: Analyze goodness-of-fit, uncertainties, and systematic effects

The ``optimize`` module provides the ``Fitter`` class that coordinates this process.

Basic Optimization Workflow
---------------------------

Let's start with a complete example using synthetic data.

Setting Up the Fitter
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from peebee import optimize, models, sampling
    from peebee.models import NFW, MiyamotoNagaiDisk
    from peebee.noise import GaussianNoise
    
    # Create test model and generate synthetic data
    true_halo = NFW(m_vir=1.2e12, r_s=18.0)
    true_disk = MiyamotoNagaiDisk(m_tot=5e10, a=3.0, b=0.3)
    true_model = true_halo + true_disk
    
    # Generate mock observations
    np.random.seed(42)
    bounds = np.array([[4.0, 12.0], [-4.0, 4.0], [-1.0, 1.0]])
    x_obs, y_obs, z_obs, alos_true = sampling.sample_alos_sources_uniform(
        true_model, 40, bounds
    )
    
    # Add realistic noise
    alos_err = 0.1 * np.abs(alos_true)  # 10% uncertainties
    alos_obs = alos_true + np.random.normal(0, alos_err)
    
    print(f"Generated {len(alos_obs)} mock observations")
    print(f"Acceleration range: [{np.min(alos_obs):.2f}, {np.max(alos_obs):.2f}] mm/s/yr")
    print(f"Mean uncertainty: {np.mean(alos_err):.3f} mm/s/yr")

The Basic Fitter Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Step 1: Create fitter instance
    fitter = optimize.Fitter()
    
    # Step 2: Set up the model to fit (use same structure as true model)
    fit_model = NFW(m_vir=8e11, r_s=15.0) + MiyamotoNagaiDisk(m_tot=4e10, a=2.5, b=0.25)
    fitter.set_model(fit_model)
    
    # Step 3: Load observational data
    fitter.set_data(x_obs, y_obs, z_obs, alos_obs, alos_err, frame='cart')
    
    # Step 4: Configure noise model
    fitter.set_noise_model(GaussianNoise(sigma=1.0))
    
    # Step 5: Set parameter bounds
    fitter.configure_params({
        'NFW.m_vir': (5e11, 2e12),           # virial mass range
        'NFW.r_s': (5.0, 40.0),              # scale radius range  
        'Miyamoto-Nagai Disk.m_tot': (1e10, 1e11),  # disk mass range
        'Miyamoto-Nagai Disk.a': (1.0, 5.0), # disk scale length
        'Miyamoto-Nagai Disk.b': (0.1, 1.0), # disk scale height
        'noise.sigma': (0.1, 10.0)           # noise parameter
    })
    
    print("Configured parameters:")
    for param, bounds in fitter.param_bounds.items():
        current = fit_model.params.get(param, fitter.noise_model.params.get(param.split('.')[-1], 'N/A'))
        print(f"  {param}: [{bounds[0]:.2e}, {bounds[1]:.2e}] (current: {current:.2e})")

Running the Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Run optimization with gradient descent
    print("Running optimization...")
    results = fitter.optimize(method='gradient_descent')
    
    print(f"\nOptimization completed!")
    print(f"Reduced χ²: {results.reduced_chi2:.3f}")
    print(f"AIC: {results.aic:.1f}")
    
    print("\nBest-fit parameters:")
    for param, value in results.best_fit_params.items():
        if param in fitter.param_bounds:  # only show fitted parameters
            true_val = true_model.params.get(param, 'N/A')
            if isinstance(true_val, (int, float)):
                recovery = (value - true_val) / true_val * 100
                print(f"  {param}: {value:.2e} (true: {true_val:.2e}, recovery: {recovery:+5.1f}%)")
            else:
                print(f"  {param}: {value:.2e}")

Different Optimization Algorithms
---------------------------------

Peebee supports several optimization methods for different use cases.

Gradient Descent
~~~~~~~~~~~~~~~~

Fast local optimization, good when you have reasonable starting values:

.. code-block:: python

    # Gradient-based optimization (scipy's minimize)
    results_gd = fitter.optimize(method='gradient_descent')
    print(f"Gradient descent: χ² = {results_gd.reduced_chi2:.3f}, "
          f"time = {results_gd.optimization_time:.2f}s")

Differential Evolution
~~~~~~~~~~~~~~~~~~~~~~

Global optimization, robust to initial conditions but slower:

.. code-block:: python

    # Global optimization with differential evolution
    results_de = fitter.optimize(method='differential_evolution')
    print(f"Differential evolution: χ² = {results_de.reduced_chi2:.3f}, "
          f"time = {results_de.optimization_time:.2f}s")

Least Squares with Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robust fitting for data with outliers:

.. code-block:: python

    # Least squares with Huber loss (robust to outliers)
    results_huber = fitter.optimize(method='least_squares', loss='huber')
    print(f"Huber robust fit: χ² = {results_huber.reduced_chi2:.3f}")
    
    # Compare parameter differences
    print("\nParameter comparison:")
    print("Parameter                   Gradient   DiffEvol   Huber")
    print("-" * 55)
    for param in ['NFW.m_vir', 'NFW.r_s', 'Miyamoto-Nagai Disk.m_tot']:
        gd_val = results_gd.best_fit_params[param]
        de_val = results_de.best_fit_params[param]
        huber_val = results_huber.best_fit_params[param]
        print(f"{param:25s} {gd_val:9.2e} {de_val:9.2e} {huber_val:9.2e}")

Working with Real Data
----------------------

Real pulsar data presents additional challenges.

Loading Real Data
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    
    # Example: Create realistic mock data that mimics real observations
    # In practice, you would load this from a file
    real_data = {
        'pulsar': ['J1713+0747', 'J1909-3744', 'J0437-4715', 'J1744-1134', 'J2145-0750'],
        'l': [16.33, 359.93, 253.40, 10.87, 74.87],      # Galactic longitude
        'b': [24.98, -29.87, -42.28, 9.19, -32.38],     # Galactic latitude  
        'd': [1.15, 1.14, 0.156, 0.36, 0.50],           # Distance (kpc)
        'alos_obs': [1.23, -0.87, 2.45, -1.56, 0.89],   # Observed acceleration (mm/s/yr)
        'alos_err': [0.15, 0.12, 0.18, 0.24, 0.21],     # Uncertainty (mm/s/yr)
        'timing_years': [15, 12, 20, 8, 10]             # Years of timing data
    }
    
    df = pd.DataFrame(real_data)
    
    print("Real pulsar dataset:")
    print(df[['pulsar', 'l', 'b', 'd', 'alos_obs', 'alos_err']].round(2))

Data Quality Assessment
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Analyze data quality before fitting
    
    # Signal-to-noise ratios
    snr = np.abs(df['alos_obs']) / df['alos_err']
    print(f"\nData quality assessment:")
    print(f"  Number of pulsars: {len(df)}")
    print(f"  Distance range: [{df['d'].min():.2f}, {df['d'].max():.2f}] kpc")
    print(f"  S/N range: [{snr.min():.1f}, {snr.max():.1f}]")
    print(f"  Detections (>2σ): {np.sum(snr > 2)}/{len(df)}")
    
    # Check for outliers
    median_acc = np.median(np.abs(df['alos_obs']))
    outliers = np.abs(df['alos_obs']) > 3 * median_acc
    print(f"  Potential outliers: {np.sum(outliers)} pulsars")
    
    if np.any(outliers):
        print("  Outlier pulsars:")
        for i in np.where(outliers)[0]:
            print(f"    {df.iloc[i]['pulsar']}: {df.iloc[i]['alos_obs']:+.2f} ± {df.iloc[i]['alos_err']:.2f}")

Fitting to Real Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Set up fitter for real data
    real_fitter = optimize.Fitter()
    
    # Use a more complex model for real data
    real_model = NFW(m_vir=1e12, r_s=20.0) + MiyamotoNagaiDisk(m_tot=5e10, a=3.0, b=0.3)
    real_fitter.set_model(real_model)
    
    # Load real data
    real_fitter.set_data(
        df['l'], df['b'], df['d'], 
        df['alos_obs'], df['alos_err'], 
        frame='gal'
    )
    
    # Use robust noise model for potential outliers
    from peebee.noise import LorentzNoise
    real_fitter.set_noise_model(LorentzNoise(gamma=1.0))
    
    # Configure parameters with physically motivated priors
    real_fitter.configure_params({
        'NFW.m_vir': (5e11, 2e12),
        'NFW.r_s': (10.0, 30.0),
        'Miyamoto-Nagai Disk.m_tot': (2e10, 1e11),
        'Miyamoto-Nagai Disk.a': (2.0, 4.0),
        'Miyamoto-Nagai Disk.b': (0.1, 0.5),
        'noise.gamma': (0.1, 5.0)
    })
    
    # Run optimization
    real_results = real_fitter.optimize(method='differential_evolution')
    
    print(f"\nReal data fit results:")
    print(f"Reduced χ²: {real_results.reduced_chi2:.3f}")
    print(f"AIC: {real_results.aic:.1f}")

Parameter Constraints and Uncertainties
---------------------------------------

Understanding parameter uncertainties is crucial for scientific interpretation.

Hessian-Based Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Extract uncertainties from optimization
    print("\nParameter uncertainties (from Hessian):")
    print("Parameter                    Value         ±1σ Error    Relative")
    print("-" * 70)
    
    for param in real_results.best_fit_params:
        if param in real_fitter.param_bounds:
            value = real_results.best_fit_params[param]
            uncertainty = real_results.uncertainties.get(param, 0)
            if uncertainty > 0:
                relative = uncertainty / value * 100
                print(f"{param:25s} {value:12.2e} ± {uncertainty:9.2e} ({relative:5.1f}%)")
            else:
                print(f"{param:25s} {value:12.2e} (no uncertainty)")

Model Comparison and Selection
-----------------------------

Compare different model structures to find the best representation.

Information Criteria
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compare models with different complexity
    
    models_to_test = [
        ("Halo only", NFW(m_vir=1e12, r_s=20.0)),
        ("Disk only", MiyamotoNagaiDisk(m_tot=5e10, a=3.0, b=0.3)),
        ("Halo + Disk", NFW(m_vir=1e12, r_s=20.0) + MiyamotoNagaiDisk(m_tot=5e10, a=3.0, b=0.3))
    ]
    
    model_results = []
    
    for name, model in models_to_test:
        # Set up fitter
        test_fitter = optimize.Fitter()
        test_fitter.set_model(model)
        test_fitter.set_data(df['l'], df['b'], df['d'], df['alos_obs'], df['alos_err'])
        test_fitter.set_noise_model(GaussianNoise(sigma=1.0))
        
        # Configure parameters based on model type
        bounds = {'noise.sigma': (0.1, 5.0)}
        if 'NFW' in str(model):
            bounds.update({'NFW.m_vir': (5e11, 2e12), 'NFW.r_s': (10.0, 30.0)})
        if 'Miyamoto' in str(model):
            bounds.update({
                'Miyamoto-Nagai Disk.m_tot': (1e10, 1e11),
                'Miyamoto-Nagai Disk.a': (1.0, 5.0),
                'Miyamoto-Nagai Disk.b': (0.1, 1.0)
            })
        
        test_fitter.configure_params(bounds)
        
        # Fit model
        result = test_fitter.optimize(method='gradient_descent')
        
        model_results.append({
            'name': name,
            'nparams': model.nparams + 1,  # +1 for noise
            'chi2': result.reduced_chi2 * (len(df) - model.nparams - 1),
            'reduced_chi2': result.reduced_chi2,
            'aic': result.aic,
            'bic': result.aic + (model.nparams + 1) * np.log(len(df)) - 2 * (model.nparams + 1)
        })
    
    # Display comparison
    print("\nModel comparison:")
    print("Model          Nparams    χ²     χ²_red    AIC     BIC")
    print("-" * 55)
    for result in model_results:
        print(f"{result['name']:12s} {result['nparams']:7d} {result['chi2']:7.1f} "
              f"{result['reduced_chi2']:7.3f} {result['aic']:7.1f} {result['bic']:7.1f}")

Residual Analysis
-----------------

Analyze fit residuals to identify systematic issues.

.. code-block:: python

    # Calculate residuals for best model
    best_model = real_fitter.model
    best_model.set_params(real_results.best_fit_params)
    
    alos_predicted = best_model.alos(df['l'], df['b'], df['d'])
    residuals = df['alos_obs'] - alos_predicted
    normalized_residuals = residuals / df['alos_err']
    
    print("Residual analysis:")
    print("Pulsar        Observed  Predicted  Residual   Norm.Res  Significance")
    print("-" * 70)
    for i, (_, row) in enumerate(df.iterrows()):
        significance = abs(normalized_residuals.iloc[i])
        flag = "***" if significance > 3 else ""
        print(f"{row['pulsar']:12s} {row['alos_obs']:8.2f} {alos_predicted[i]:8.2f} "
              f"{residuals.iloc[i]:8.2f} {normalized_residuals.iloc[i]:8.2f} "
              f"{significance:8.2f}σ {flag}")
    
    # Statistical tests
    print(f"\nResidual statistics:")
    print(f"  RMS residual: {np.std(residuals):.3f} mm/s/yr")
    print(f"  Mean |residual|: {np.mean(np.abs(residuals)):.3f} mm/s/yr")
    print(f"  Normalized residuals mean: {np.mean(normalized_residuals):+.2f}")
    print(f"  Normalized residuals std: {np.std(normalized_residuals):.2f}")
    print(f"  Outliers (>3σ): {np.sum(np.abs(normalized_residuals) > 3)} / {len(residuals)}")

Best Practices
--------------

For reliable parameter inference:

1. **Start with simple models**: Test individual components before fitting complex composites
2. **Use physical priors**: Set parameter bounds based on observational constraints
3. **Test multiple algorithms**: Compare local and global optimization results
4. **Check convergence**: Verify results don't depend on starting values
5. **Analyze residuals**: Look for systematic patterns that suggest missing physics
6. **Consider model selection**: Use information criteria to choose appropriate complexity
7. **Validate with synthetic data**: Test your analysis pipeline on mock datasets

Troubleshooting Common Issues
----------------------------

**Poor convergence**: 
  - Try different starting values or use global optimization
  - Check that parameter bounds are reasonable
  - Ensure data quality (remove outliers, check uncertainties)

**Unphysical parameters**:
  - Tighten parameter bounds based on observational constraints
  - Check for parameter degeneracies
  - Consider whether model structure is appropriate

**Large residuals**:
  - Examine data for outliers or systematic errors
  - Consider more complex model structures
  - Use robust noise models (Lorentz, PowerLaw)

**Unstable uncertainties**:
  - Check that Hessian is positive definite
  - Use profile likelihood for better uncertainty estimates
  - Bootstrap resampling for non-Gaussian uncertainties

Next Steps
----------

With fitted models in hand, you can:

- :doc:`noise_models`: Explore robust fitting techniques for difficult datasets
- Compare your results with literature values and physical expectations
- Use fitted models to predict accelerations for new pulsar discoveries
- Extend analysis to include time-dependent effects or non-standard dark matter models