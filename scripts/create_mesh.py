#-------------------------------------------------------------------------------
# Generate 2D initial domain for grounding line problems
# To convert the msh file to an xml file, use the following bash command
# in the FEniCS Docker image:
# dolfin-convert mesh_name.msh mesh_name.xml
#
# Note: This always assumes a horizontal upper surface.
#-------------------------------------------------------------------------------
import sys
sys.path.insert(0, './source')

import numpy as np
import matplotlib.pyplot as plt
from geometry import bed,interface
from params import Hght,Lngth,DX_s,DX_h
import subprocess

nx = int(Lngth/DX_s)+1                  # Number of grid points in x direction.

X = np.linspace(0,Lngth,num=nx)         # array for horizontal coordinate
S = interface(X)                        # Ice-water interface array

#-------------------------------------------------------------------------------

fname = 'marine_DX'+str(int(DX_s))+'.geo'

fI = open(fname,"w")   # Generate mesh file.

# Define points of that will be used to create the mesh.
# Bottom (ice-water interface)
for i in range(nx):
    fI.write('//+ \n Point(%d) = {%f, %f, 0, %f}; \n'%(i+1,X[i],S[i],DX_s))

# Top right corner
fI.write('//+ \n Point(%d) = {%f, %f, 0, %f}; \n'%(nx+1,X[-1],Hght,DX_h))

# Top left corner
fI.write('//+ \n Point(%d) = {%f, %f, 0, %f}; \n'%(nx+2,X[0],Hght,DX_h))

NP = nx+2


# Define lines used to create mesh.
lines = []

for i in range(NP-1):
    fI.write('//+ \n Line(%d) = {%d,%d}; \n'%( i+1,i+1,i+2))
    lines.append(i+1)

fI.write('//+ \n Line(%d) = {%d,%d}; \n'%(NP,NP,1))
lines.append(NP)

fI.write('//+ \n Line Loop(1) = {%s};\n' % str(lines)[1:-1])


fI.write('//+ \n Plane Surface(1) = {1};\n')

fI.close()

# Generate msh files
bashCommand1 = "gmsh -2 -format msh2 "+fname
process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
output, error = process1.communicate()
