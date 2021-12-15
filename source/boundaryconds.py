#-------------------------------------------------------------------------------
# This file contains functions that:
# (1) define the boundaries (ice-air,ice-water,ice-bed) of the mesh,
# (2) mark the boundaries of the mesh, AND ...
# (3) create Dirichlet boundary conditions on one or both side walls of the domain.
#-------------------------------------------------------------------------------
from params import tol,U0,Lngth,Hght
from geometry import bed
import numpy as np
from dolfin import *

#-------------------------------------------------------------------------------
# Define SubDomains for ice-water boundary, ice-bed boundary, inflow (x=0) and
# outflow (x=Length of domain). The parameter 'tol' is a minimal water depth
# used to distinguish the ice-water and ice-bed surfaces.

class WaterBoundary(SubDomain):
    # Ice-water boundary.
    # Note: This boundary is marked first and all of the irrelevant portions are
    # overwritten by the other boundary markers. This results in a "last grounded"
    # scheme as described in Gagliardini et al. (2016), The Cryosphere.
    def inside(self, x, on_boundary):
        return (on_boundary and (x[1]<0.5*Hght))

class BedBoundary(SubDomain):
    # Ice-bed boundary
    def inside(self, x, on_boundary):
        return (on_boundary and ((x[1]-bed(x[0]))<=tol))

class LeftBoundary(SubDomain):
    # Left boundary
    def inside(self, x, on_boundary):
        return (on_boundary and np.abs(x[0])<tol)

class RightBoundary(SubDomain):
    # Right boundary
    def inside(self, x, on_boundary):
        return (on_boundary and np.abs(x[0]-Lngth)<tol)

#-------------------------------------------------------------------------------

def mark_boundary(mesh):
    # Assign markers to each boundary segment (except the upper surface).
    # This is used at each time step to update the markers.
    #
    # Boundary marker numbering convention:
    # 1 - Left boundary
    # 2 - Right boundary
    # 3 - Ice-bed boundary
    # 4 - Ice-water boundary

    boundary_markers = MeshFunction('size_t', mesh,dim=1)
    boundary_markers.set_all(0)

    # Mark ice-water boundary
    bdryWater = WaterBoundary()
    bdryWater.mark(boundary_markers, 4)

    # Mark ice-bed boundary
    bdryBed = BedBoundary()
    bdryBed.mark(boundary_markers, 3)

    # Mark inflow boundary
    bdryLeft = LeftBoundary()
    bdryLeft.mark(boundary_markers, 1)

    # Mark outflow boundary
    bdryRight = RightBoundary()
    bdryRight.mark(boundary_markers, 2)

    return boundary_markers

#------------------------------------------------------------------------------

def apply_bcs(W,boundary_markers):
    # Apply inflow and outflow boundary conditions to the system.
    # These are applied to the horizontal velocity component.
    bcu1 = DirichletBC(W.sub(0).sub(0), Constant(U0), boundary_markers,1)

    BC = [bcu1]
    return BC
