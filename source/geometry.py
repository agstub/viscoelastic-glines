#-------------------------------------------------------------------------------
# Define bed topography and initial ice-water interface functions.
# They should be equal on one margin of the domain for the
# marine ice sheet problem (i.e., the grounded portion).
# NOTE: initial ice-water elevation should be set to z=0 (i.e. interface=0 where interface>bed)
#-------------------------------------------------------------------------------

import numpy as np
from params import Lngth,Hght,rho_w,rho_i

#-------------------- Generate Bed Topography-----------------------------------
def bed(x):
    Bed = -np.abs(600*x/(0.5*Lngth) - 600)
    return Bed

#------------------Generate initial ice-water/ice-bed interface-----------------
def interface(x):
    Int = 0.5 * bed(x + 0.5*Lngth*bed(x)/600)
    return Int
#-------------------------------------------------------------------------------

# Generate initial ice surface profile

def elevation0(x):
    fl = (rho_w / rho_i-1)
    el = Hght-1.1*fl*interface(x)
    return el

import matplotlib.pyplot as plt
from params import X_fine
plt.figure(figsize=(8,6))
X0 = X_fine/1000 - 0.5*Lngth/1000
plt.plot(X0,elevation0(X_fine),color='royalblue',linewidth=2)
plt.fill_between(X0,y1=interface(X_fine), y2=elevation0(X_fine),facecolor='aliceblue',alpha=1.0)
plt.fill_between(X0[interface(X_fine)>bed(X_fine)],y1=bed(X_fine)[interface(X_fine)>bed(X_fine)], y2=interface(X_fine)[interface(X_fine)>bed(X_fine)],facecolor='slateblue',alpha=0.5)
plt.fill_between(X0,y1=0*X0-1e4, y2=bed(X_fine),facecolor='burlywood',alpha=0.75)
plt.plot(X0[interface(X_fine)>bed(X_fine)],interface(X_fine)[interface(X_fine)>bed(X_fine)],color='crimson',linewidth=2)
plt.plot(X0,bed(X_fine),color='k',linewidth=2)
plt.xlabel(r'$x$ (km)',fontsize=20)
plt.ylabel(r'$z$ (m)',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-0.5*Lngth/1000,0.5*Lngth/1000)
plt.ylim(np.min(bed(X_fine))-100,np.max(elevation0(X_fine))+100)
plt.tight_layout()
plt.savefig('initial_geometry',bbox_inches='tight')
