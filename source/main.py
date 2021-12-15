#-------------------------------------------------------------------------------
# This program solves a linear small-deformation Maxwell viscoelasticy problem
# with grounding line migration (constant Newtonian viscosity)
#-------------------------------------------------------------------------------
import sys
sys.path.insert(0, './scripts')

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from stokes import stokes_solve,get_zero_m
from geometry import interface,bed
from meshfcns import mesh_routine
import os
from params import (rho_i,g,tol,t_final,nt_per_year,Lngth,Hght,nt,dt,
                    print_convergence,X_fine,nx,tides,DX_s)

if os.path.isdir('results')==False:
    os.mkdir('results')   # Make a directory for the results.

if print_convergence == 'off':
    set_log_level(40)    # Suppress Newton convergence information if desired.

# Create VTK files
vtkfile_u = File('results/stokes/u.pvd')
vtkfile_p = File('results/stokes/p.pvd')

# Load mesh
if tides=='on':
    meshname = 'tides'+'_DX'+str(int(DX_s))+'.xml'
elif tides=='off':
    meshname = 'marine'+'_DX'+str(int(DX_s))+'.xml'
    # Create initial mesh for tides simulation by running the marine model with
    # tides turned off.
    new_mesh = File('./meshes/tides_DX'+str(int(DX_s))+'.xml')

mesh = Mesh('./meshes/'+meshname)

# Define arrays for saving surfaces, lake volume, water pressure, and
# grounding line positions over time.
Gamma_s = np.zeros((nx,nt))       # Basal surface
Gamma_h = np.zeros((nx,nt))       # Upper surface
s_mean = np.zeros(nt)             # Mean elevation of ice-water interface
h_mean = np.zeros(nt)             # Mean elevation of surface above ice-water interface
x_left = np.zeros(nt)             # Left grounding line position
x_right = np.zeros(nt)            # Right grounding line position
P_res = np.zeros(nt)              # Penalty functional residual

t = 0                             # time

W_stress = TensorFunctionSpace(mesh,"CG",1) # functionspace for dev. stress tensor
tau = Function(W_stress)                    # initial deviatoric stress tensor (zero)

# Begin time stepping
for i in range(nt):

    print('-----------------------------------------------')
    print('Iteration '+str(i+1)+' out of '+str(nt))

    if t==0:
        # Set initial conditions.
        F_h = lambda x: Hght                  # Ice-air surface function
        F_s = lambda x: interface(x)    # Lower surface function

        w = get_zero_m(mesh)              # Placeholder for first iteration.

        mesh,F_s,F_h,s_mean_i,h_mean_i,XL,XR = mesh_routine(w,mesh,dt)



    # Solve the Stoke problem.
    # Returns solutions "w" and penalty functional residual "Perr_i"
    w,P_res_i,tau = stokes_solve(mesh,F_h,t,w,tau)

    # Solve the surface kinematic equations, move the mesh, and compute the
    # grounding line positions.
    mesh,F_s,F_h,s_mean_i,h_mean_i,XL,XR = mesh_routine(w,mesh,dt)

    # Save quantities of interest.
    P_res[i] = P_res_i
    s_mean[i] = s_mean_i
    h_mean[i] = h_mean_i
    x_left[i] = XL
    x_right[i] = XR
    Gamma_s[:,i] = F_s(X_fine)
    Gamma_h[:,i] = F_h(X_fine)

    # Save (u,p) solution for viewing in Paraview.
    # Save Stokes solution
    _u, _p = w.split()
    _u.rename("vel", "U")
    _p.rename("press","P")
    vtkfile_u << (_u,t)
    vtkfile_p << (_p,t)

    # Update time
    t += dt

# Save quantities of interest.
t_arr = np.linspace(0,t_final,num=int(nt_per_year*t_final/3.154e7))

np.savetxt('results/Gamma_s',Gamma_s)
np.savetxt('results/Gamma_h',Gamma_h)
np.savetxt('results/s_mean',s_mean)
np.savetxt('results/h_mean',h_mean)
np.savetxt('results/x_left',x_left)
np.savetxt('results/x_right',x_right)
np.savetxt('results/P_res',P_res)
np.savetxt('results/X',X_fine)           # X = spatial coordinate
np.savetxt('results/t',t_arr)            # t = time coordinate


if tides=='off':
    new_mesh << mesh
