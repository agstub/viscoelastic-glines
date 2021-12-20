# All model parameters and options are recorded here.
import numpy as np
#-------------------------------------------------------------------------------
#-----------------------------MODEL OPTIONS-------------------------------------

# Turn the tidal cycle 'on' or 'off'.
tides = 'on'

# Turn 'on' or 'off' real-time plotting that saves a png figure called 'surfs' at
# each time step of the free surface geometry.

realtime_plot = 'on'

# Turn 'on' or 'off' Newton convergence information:
print_convergence = 'on'

# Mesh resolution at the lower boundary
DX_s = 100.0                  # Element width at lower boundary (in meters)
                              # This is used for (1) setting the element width in
                              # gendomain.py and (2) selecting the mesh in main.py.

DX_h = 250.0                  # Element width at the upper surface (in meters)

#-----------------------------MODEL PARAMETERS----------------------------------
#-------------------------------------------------------------------------------

# Material parameters

A0 = 3.1689e-24                    # Glen's law coefficient (ice softness, Pa^{-n}/s)
n = 3.0                            # Glen's law exponent

nu = 0.5                           # Poisson ratio

E = 1.0e9                          # Young's modulus

G = E/(2*(1+nu))                   # Shear modulus

rho_i = 917.0                      # Density of ice
rho_w = 1000.0                     # Density of water
g = 9.81                           # Gravitational acceleration
C = 1.0e8                          # Sliding law friction coefficient

# Numerical parameters
eta0 = 1e13                        # viscosity at zero deviatoric stress

eps_v = (2*A0*eta0)**(2/(1-n))     # Flow law regularization parameter
eps_p = 1.0e-12                    # Penalty method parameter
quad_degree = 16                   # Quadrature degree for weak forms

tol = 1.0e-2                       # Numerical tolerance for boundary geometry:
                                   # s(x,t) - b(x) > tol on ice-water boundary,
                                   # s(x,t) - b(x) <= tol on ice-bed boundary.

# Geometry parameters
Lngth = 20*1000.0                  # Length of the domain
Hght = 500.0                       # (Initial) Height of the domain

sea_level = Hght*(917.0/1000.0)    # Sea level elevation.
                                   # (Initial sea level for the tides problem)

# Time-stepping parameters
if tides == 'off':
    nt_per_year = 2000             # Number of timesteps per year.
    t_final = 0.15*3.154e7         # Final time (yr*sec_per_year).
elif tides == 'on':
    nt_per_year = 100*1000         # Number of timesteps per year.
    t_final = 0.003*3.154e7        # Final time (yr*sec_per_year).


nt = int(nt_per_year*t_final/3.154e7) # Number of time steps
dt = t_final/nt                       # Timestep size

nx = 1000
X_fine = np.linspace(0,Lngth,num=nx)  # Horizontal coordinate for computing surface
                                      # slopes and plotting.

# Set inflow speed boundary condition for marine ice sheet problem
U0   = 1000.0/3.154e7                 # Inflow speed 1 km/yr

#-------------------------------------------------------------------------------
