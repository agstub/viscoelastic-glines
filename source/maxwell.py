# This file contains the functions needed for solving the upper-convected Maxwell problem.
from params import rho_i,g,tol,rho_w,C,eps_p,eps_v,sea_level,dt,quad_degree,Lngth,n,A0,G
from boundaryconds import mark_boundary,apply_bcs
from tides import sl_change
import numpy as np
from dolfin import *

def dPi(u,nu):
        # (Derivative of) penalty function for enforcing impenetrability on the ice-bed boundary.
        un = dot(u,nu)
        return un+abs(un)

def Pi(u,nu):
        # Penalty function for enforcing impenetrability on the ice-bed boundary.
        un = dot(u,nu)
        return 0.5*(un**2.0+un*abs(un))

def eta(tau):
        # viscosity in terms of the deviatoric stress
        return (0.5/A0)*(inner(tau,tau)+eps_v)**((1-n)/2.0)

def lamda(tau):
        return eta(tau)/G

def ucd(tau,tau_prev,u):
        # upper-convected time derivative
        return ((tau-tau_prev)/dt + dot(u,nabla_grad(tau)) - dot(grad(u),tau)-dot(tau,grad(u).T))

def weak_form(tau,mu,u,p,v,q,f,g_base,g_out,ds,nu,T,tau_prev):
    # Weak form of the residual equations
    F1 =  inner(tau,grad(v))*dx + (- div(v)*p + q*div(u))*dx - inner(f, v)*dx \
         + inner(tau + lamda(tau)*ucd(tau,tau_prev,u)-2*eta(tau)*sym(grad(u)),mu)*dx\

    F2 = + (g_base+Constant(rho_w*g*dt)*inner(u,nu))*inner(nu, v)*ds(4)\
         + Constant(C)*inner(dot(T,u),dot(T,v))*ds(3)\
         + Constant(1.0/eps_p)*dPi(u,nu)*inner(v,nu)*ds(3)\
         + (g_base+Constant(rho_w*g*dt)*inner(u,nu))*inner(nu, v)*ds(3)\
         + g_out*inner(nu, v)*ds(2)
    return F1+F2

def maxwell_solve(mesh,F_h,t,w,tau_prev):
        # Stokes solver using Taylor-Hood elements.

        # Define finite elements and function space
        P1 = FiniteElement('P',mesh.ufl_cell(),1)     # Pressure
        P2 = FiniteElement('P',mesh.ufl_cell(),2)     # Velocity
        T1 = TensorElement("CG", mesh.ufl_cell(), 2,symmetry=True) # stress
        element = MixedElement([[P2,P2],P1,T1])
        W = FunctionSpace(mesh,element)               # Function space for (u,p)

        #---------------------Define variational problem------------------------
        w = Function(W)
        (u,p,tau) = split(w)
        (v,q,mu) = TestFunctions(W)

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

        bcs =  apply_bcs(W,boundary_markers)    # Apply Dirichlet BC

        # Solve for (u,p).
        Fw = weak_form(tau,mu,u,p,v,q,f,g_base,g_out,ds,nu,T,tau_prev)

        solve(Fw == 0, w, bcs=bcs,solver_parameters={"newton_solver":{"relative_tolerance": 1e-14,"maximum_iterations":100}},form_compiler_parameters={"quadrature_degree":quad_degree,"optimize":True,"eliminate_zeros":False})

        # compute deviatoric stress tensor for use in next time step
        W_stress = TensorFunctionSpace(mesh,"CG",1)
        tau = project(w.sub(2),W_stress)

        # Compute penalty functional residiual
        P_res = assemble(Pi(u,nu)*ds(3))

        return w,P_res,tau


def get_zero_m(mesh):
        # Get zero element of function space for marine ice sheet problem.
        # Only used for setting initial conditions; see main.py.

        # Define function spaces
        P1 = FiniteElement('P',mesh.ufl_cell(),1)     # Pressure
        P2 = FiniteElement('P',mesh.ufl_cell(),2)     # Velocity
        T1 = TensorElement("CG", mesh.ufl_cell(), 1, symmetry=True) # stress
        element = MixedElement([[P2,P2],P1,T1])
        W = FunctionSpace(mesh,element)        # Function space for (u,p)

        w = Function(W)

        return w
