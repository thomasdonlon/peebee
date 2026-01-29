# PEEBEE Development Context Document

**Last Updated**: January 29, 2026  
**Version**: 1.2.1  
**Maintainer**: Tom Donlon (thomasdonlon)  

## Project Overview

### Mission Statement
PEEBEE is a Python package designed for **Galactic structure research using pulsar timing data**, specifically focusing on the intersection between pulsar accelerations and Milky Way structure. The package provides a framework for analyzing pulsar acceleration data from arbitrary surveys to study Galactic dynamics, dark matter substructure, and gravitational potential modeling.

### Core Scientific Applications
- **Dark matter substructure detection and characterization**
- **Galactic rotation curve analysis**
- **Pulsar acceleration data analysis from multiple surveys**
- **Gravitational potential model fitting and validation**
- **Line-of-sight acceleration calculations for pulsar timing**

### Project Scope
- **FOCUSED**: Milky Way science only (no extragalactic applications)
- **TARGET USERS**: Research group focused on pulsar/Galactic structure research
- **NOT INTENDED**: For broad community use (specialized research tool)

## Architecture Overview

### Core Design Philosophy
1. **Ease of use prioritized over pure performance** (but performance cannot be completely sacrificed)
2. **Backwards compatibility is NOT important** - breaking changes acceptable for improvements
3. **Mixed programming paradigm**: Object-oriented for optimizers, flexible approach elsewhere
4. **Computational efficiency required** for optimization routines (core functionality)

### Package Structure
```
peebee/
├── __init__.py          # Import all submodules
├── models.py            # Gravitational potential models (CORE)
├── optimize.py          # Optimization and fitting routines (PRIORITY)
├── convenience.py       # Helper functions for pulsar calculations
├── transforms.py        # Coordinate system transformations
├── sampling.py          # Sampling and bootstrapping routines
├── noise.py            # Noise model implementations
└── glob.py             # Global utilities and constants
```

## Core Components

### 1. Models System (`models.py`)
**Purpose**: Gravitational potential models for acceleration calculations

**Base Architecture**:
- `Model` base class with standard interface
- `CompositeModel` class for combining multiple potentials
- All models implement `accel(x, y, z)` and `alos()` methods

**Available Model Types**:
- **Dark Matter**: NFW, Hernquist
- **Stellar Components**: Plummer, Miyamoto-Nagai Disk, Point Mass
- **Disk Structure**: Exponential Disk, Anharmonic Disk, Sinusoidal Disk
- **Spiral Structure**: Cox-Gomez Spiral Arms
- **Specialized**: Oort Expansion, Cross, Uniform acceleration models
- **External Library Wrappers**: GalaPotential, GalpyPotential

**Key Features**:
- Coordinate frame conversion (Galactic ↔ Cartesian)
- Line-of-sight acceleration calculation (`alos()`)
- Tangential acceleration calculation (`atan()`)
- Parameter management with optional/required parameters
- Support for disabled/fixed parameters during optimization

### 2. Optimization System (`optimize.py`) - **CURRENT PRIORITY**
**Purpose**: Fit gravitational models to acceleration data

**Current Status**: Needs revamping for easier use and less clunky interface

**Core Components**:
- `Fitter` class for interfacing with scipy optimizers
- Support for various noise models
- Parameter constraint handling
- Composite model optimization

**Planned Improvements**:
- More object-oriented approach
- Simpler user interface
- Better parameter management
- MCMC support (future enhancement)

### 3. Convenience Functions (`convenience.py`)
**Purpose**: Common pulsar timing calculations

**Key Functions**:
- `pbdot_gr()`: Gravitational wave orbital decay
- `pdot_shk()`: Shklovskii effect calculation
- `alos_obs()`: Observed line-of-sight acceleration
- Coordinate transformations and utilities

### 4. Coordinate Transformations (`transforms.py`)
**Purpose**: Handle different coordinate systems

**Capabilities**:
- Galactic ↔ Cartesian conversions
- Heliocentric ↔ Galactocentric frames
- Decorator-based frame conversion for functions

### 5. Sampling and Noise (`sampling.py`, `noise.py`)
**Purpose**: Monte Carlo sampling and noise modeling

**Features**:
- Population sampling from potential models
- Various noise models (Gaussian, Lorentzian, Power-law)
- Bootstrap and uncertainty propagation tools

## Development Priorities

### Immediate Priority (Next Steps)
1. **Optimizer Revamp**: Make optimization interface easier to use and less clunky
   - Simplify user interface
   - Improve object-oriented design
   - Better parameter management
   - Enhanced error handling

### Medium-Term Goals
2. **Quick TODOs**: Address simple, high-impact TODOs first
   - Parameter name collision handling
   - Array/single value input consistency
   - Documentation improvements

### Long-Term Vision
3. **Astropy Units Integration**: Full astropy unit support throughout package
   - **HIGH IMPORTANCE** but **HIGH COMPLEXITY**
   - Automatic unit conversion
   - Unit-aware calculations
   - Backward-compatible implementation

4. **Enhanced Model Library**: Expand available potential models
   - Additional dark matter profiles
   - More stellar component models
   - Improved galactic disk models

5. **MCMC Integration**: Add Markov Chain Monte Carlo optimization support
   - Parameter uncertainty estimation
   - Posterior sampling
   - Model comparison tools

## Technical Specifications

### Dependencies
**Core Requirements**:
- `numpy`: Array operations and mathematics
- `scipy`: Optimization routines
- `astropy`: Astronomical calculations and units
- `matplotlib`: Plotting (if needed)

**External Potential Libraries**:
- `gala`: Gravitational potential library
- `galpy`: Galactic dynamics library
- **Both supported equally** - no preference between them

**Additional**:
- `uncertainties`: Error propagation
- Python >= 3.6.0

### Key Constants and Conventions
- **Coordinate System**: Left-handed Galactic coordinates
- **Units**: kpc, M_sun, s (internal), with kpc/s² for accelerations
- **Solar Position**: (r_sun, 0, 0) = (8.0, 0, 0) kpc default
- **Physical Constants**: G, c, etc. defined in `glob.py`

### Code Standards

**Documentation Style**:
Follow existing docstring format:
```python
def function_name(param1, param2):
    """
    Brief description of what the function does.
    
    :param1: Description (units)
    :param2: Description (units)
    """
```

**Decorators**:
- `@fix_arrays`: Handles single values vs arrays consistently
- `@convert_to_frame('gal'|'cart')`: Automatic coordinate conversion

**Error Handling**:
- Informative error messages for parameter mismatches
- Validation of coordinate inputs
- Graceful handling of edge cases

## Known Issues and TODOs

### High Priority TODOs
- **Optimizer revamp** (immediate priority)
- **Dict name collisions** when combining models with identical parameter names
- **Astropy units integration** (complex but important)

### Medium Priority TODOs
- Allow single inputs as well as arrays (currently arrays only)
- All acceleration functions should have `fix_arrays` and `convert_to_frame()` decorators
- Global solar position setting/getting functionality
- Parameter bounds and constraint handling improvements

### Known Compatibility Issues
- **Astropy v6.1.0 + Gala v1.8.1**: Errors when computing accelerations
- **Solution**: Downgrade astropy to v5.3.0 or upgrade Gala to latest version

## Testing and Validation

**Current Approach**: External testing by development team
**No automated test suite** - validation done through scientific use cases

**Validation Methods**:
- Comparison with known analytical solutions
- Cross-validation with gala/galpy calculations
- Scientific benchmarking against literature values

## Future Development Pathway

### Phase 1 (Immediate): Optimizer Enhancement
- Redesign `optimize.py` with better OOP structure
- Simplify user interface for model fitting
- Improve parameter management system

### Phase 2 (Short-term): Code Quality Improvements  
- Address quick-fix TODOs
- Improve error handling and validation
- Enhance documentation

### Phase 3 (Medium-term): Feature Expansion
- Additional potential models
- Enhanced noise models
- Better coordinate system support

### Phase 4 (Long-term): Advanced Features
- Full astropy units integration
- MCMC optimization support
- Advanced model comparison tools

## Usage Patterns

**Typical Workflow**:
1. **Data Preparation**: Import pulsar position/acceleration data
2. **Model Selection**: Choose appropriate potential model(s)
3. **Model Fitting**: Use optimizer to fit parameters
4. **Analysis**: Extract fitted parameters and uncertainties
5. **Validation**: Compare results with observations

**Code Examples**:
```python
# Model creation
model = peebee.models.NFW(m_vir=1e12, r_s=20.0)

# Acceleration calculation
ax, ay, az = model.accel(x, y, z)
alos = model.alos(l, b, d, frame='gal')

# Optimization (current - to be improved)
fitter = peebee.optimize.Fitter(model)
result = fitter.fit_model(data)
```

---

**Note**: This document should be updated as the project evolves, particularly after the optimizer revamp and any major architectural changes.