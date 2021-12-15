# This file contains the functions needed for solving the Stokes system.
from params import rho_i,g,tol,rho_w,C,eps_p,eps_v,sea_level,dt,quad_degree,Lngth,eta,lamda
from boundaryconds import mark_boundary,apply_bcs
from tides import sl_change
import numpy as np
from dolfin import *

def dPi(u,nu):
        # (Derivative of ) penalty function for enforcing impenetrability on the ice-bed boundary.
        un = dot(u,nu)
        return un+abs(un)

def Pi(u,nu):
        # Penalty function for enforcing impenetrability on the ice-bed boundary.
        un = dot(u,nu)
        return 0.5*(un**2.0+un*abs(un))

def weak_form(u,p,v,q,f,g_base,g_out,ds,nu,T,tau):
    # Weak form of the marine ice sheet problem
    F = (2*eta/(1+lamda))*inner(sym(grad(u)),sym(grad(v)))*dx+(lamda/(1+lamda))*inner(tau,sym(grad(v)))*dx\
         + (- div(v)*p + q*div(u))*dx - inner(f, v)*dx \
         + (g_base+Constant(rho_w*g*dt)*inner(u,nu))*inner(nu, v)*ds(4)\
         + Constant(C)*inner(dot(T,u),dot(T,v))*ds(3)\
         + Constant(1.0/eps_p)*dPi(u,nu)*dot(v,nu)*ds(3)\
         + (g_base+Constant(rho_w*g*dt)*inner(u,nu))*inner(nu, v)*ds(3)\
         + g_out*inner(nu, v)*ds(2)
    return F

def compute_stress(w,W_vel,W_stress,tau_prev):
        # computes the deviatoric stress from the velocity solution and
        # the stress from the previous timestep

        u = project(w.sub(0),W_vel)
        # deviatoric stress at previous timestep
        tau_prev = project(tau_prev,W_stress)
        # "viscous" component of deviatoric stress
        tau_v = project((2*eta/(1+lamda))*sym(grad(u)),W_stress)
        # deviatoric stress at current timestep
        tau = project(tau_v+(lamda/(1+lamda))*tau_prev,W_stress)
        return tau

def stokes_solve(mesh,F_h,t,w,tau_prev):
        # Stokes solver using Taylor-Hood elements.

        # Define function spaces
        P1 = FiniteElement('P',triangle,1)     # Pressure
        P2 = FiniteElement('P',triangle,2)     # Velocity
        element = MixedElement([[P2,P2],P1])
        W = FunctionSpace(mesh,element)        # Function space for (u,p)

        W_vel = FunctionSpace(mesh, MixedElement([P2,P2]))
        W_stress = TensorFunctionSpace(mesh,"CG",1)

        #---------------------Define variational problem------------------------
        w = Function(W)
        (u,p) = split(w)
        (v,q) = TestFunctions(W)

        # Neumann condition at outflow boundary
        h_out = float(F_h(Lngth))        # Surface elevation at outflow boundary
        g_out = Expression('rho_i*g*(h_out-x[1])',rho_i=rho_i,g=g,h_out=h_out,degree=1)

        # Neumann condition at ice-water boundary
        g_base = Expression('rho_w*g*(sea_level-x[1])',rho_w=rho_w,g=g,sea_level=sea_level+sl_change(t),degree=1)

        f = Constant((0,-rho_i*g))        # Body force
        nu = FacetNormal(mesh)            # Outward-pointing unit normal to the boundary
        I = Identity(2)                   # Identity tensor
        T = I - outer(nu,nu)              # Tangential projection operator

        # Mark bounadries of mesh and define a measure for integration
        boundary_markers= mark_boundary(mesh)
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        # Define weak form and apply boundary conditions on the inflow boundary

        bcs_u =  apply_bcs(W,boundary_markers)    # Apply Dirichlet BC

        # Solve for (u,p).
        Fw = weak_form(u,p,v,q,f,g_base,g_out,ds,nu,T,tau_prev)

        solve(Fw == 0, w, bcs=bcs_u,solver_parameters={"newton_solver":{"relative_tolerance": 1e-14,"maximum_iterations":200}},form_compiler_parameters={"quadrature_degree":quad_degree,"optimize":True,"eliminate_zeros":False})

        # compute deviatoric stress tensor for use in next time step
        tau = compute_stress(w,W_vel,W_stress,tau_prev)

        # Compute penalty functional residiual
        P_res = assemble(Pi(u,nu)*ds(3))

        return w,P_res,tau


def get_zero_m(mesh):
        # Get zero element of function space for marine ice sheet problem.
        # Only used for setting initial conditions; see main.py.

        # Define function spaces
        P1 = FiniteElement('P',triangle,1)     # Pressure
        P2 = FiniteElement('P',triangle,2)     # Velocity
        element = MixedElement([[P2,P2],P1])
        W = FunctionSpace(mesh,element)        # Function space for (u,p)

        w = Function(W)

        return w
