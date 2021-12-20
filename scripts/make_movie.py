# This program creates a group of png images of the upper and lower surfaces
# over time. After running this, change to the 'pngs' directory and run the
# following bash script, where frame_rate is an integer (e.g., 50):
#
# ffmpeg -r frame_rate -f image2 -s 1920x1080 -i %01d.png -vcodec libx264 -pix_fmt yuv420p -vf scale=1280:-2 movie.mp4
#
# (This requires ffmpeg: https://ffmpeg.org/)

import sys
sys.path.insert(0, './source')

import matplotlib.pyplot as plt
import numpy as np
from geometry import Lngth, interface, bed,Hght
from params import tol,sea_level,tides
from tides import sl_change
import subprocess
import os

# Create directory for png's
if os.path.isdir('pngs')==False:
    os.mkdir('pngs')

# Load relevant files
Gamma_s = np.loadtxt('results/Gamma_s')             # Lower surface
Gamma_h = np.loadtxt('results/Gamma_h')             # Upper surface
x_left = np.loadtxt('results/x_left')               # Left grounding line
x_right = np.loadtxt('results/x_right')             # Right grounding line


# Create array for plotting
NX = np.shape(Gamma_s)[0]                           # (Uniform) Grid spacing
NT = np.shape(Gamma_s)[1]                           # Number of time steps
X = np.loadtxt('results/X')                         # x-coordinate array
t = np.loadtxt('results/t')                         # time array

# Loop over time steps
for i in range(NT):
    plt.figure(figsize=(8,6))

    if tides == 'off':
        plt.title(r'$t=$'+format(t[i]/3.154e7,'.2f')+' yr',loc='left',fontsize=20)
    else:
        plt.title(r'$t=$'+format(t[i]/(3.154e7/12.0/30.0),'.2f')+' d',loc='left',fontsize=20)

    # Plot upper surface
    p1, = plt.plot(X/1000-0.5*Lngth/1000,Gamma_h[:,i]-0.98*Hght,color='royalblue',linewidth=2,label=r'$h$')

    # Plot ice, water, and bed domains; colored accordingly.
    plt.fill_between(X/1000-0.5*Lngth/1000,y1=Gamma_s[:,i], y2=Gamma_h[:,i]-0.98*Hght,facecolor='aliceblue',alpha=1.0)
    plt.fill_between(X/1000-0.5*Lngth/1000,bed(X),Gamma_s[:,i],facecolor='slateblue',alpha=0.5)
    plt.fill_between(X/1000-0.5*Lngth/1000,-18*np.ones(np.size(X)),bed(X),facecolor='burlywood',alpha=1.0)

    # Plot bed surface
    p2, = plt.plot(X/1000-0.5*Lngth/1000,bed(X),color='k',linewidth=2,label=r'$\beta$')


    p3, = plt.plot(X[(Gamma_s[:,i]-bed(X)>tol)]/1000-0.5*Lngth/1000,Gamma_s[:,i][(Gamma_s[:,i]-bed(X)>tol)],'o',markersize=1,color='crimson',label=r'$s>\beta$')

    p3, = plt.plot(X[(Gamma_s[:,i]-bed(X)>tol)]/1000-0.5*Lngth/1000,1e5+Gamma_s[:,i][(Gamma_s[:,i]-bed(X)>tol)],'-',linewidth=2,color='crimson',label=r'$s>\beta$')

    # Plot grounding lines
    p4, = plt.plot(np.array([x_right[i]/1000-0.5*Lngth/1000]),np.array([np.min(bed(X))-1.0]),marker=r'^',color='k',linestyle='None',markersize=10,label=r'$x_\pm$')

    # plot the flotation elevation and sea level for the marine problem
    p5 = plt.axhline(y=0.02*sea_level+sl_change(t[i]),xmin=0.8,linestyle='--',color='seagreen',linewidth=2,label=r'$\ell$')
    p6 = plt.axhline(y=(1.0/0.917)*(sea_level+sl_change(t[i]))-Hght+10+(1-1.0/0.917)*Gamma_s[:,i][-1],xmin=0.8,linestyle='--',color='purple',linewidth=2,label=r'$h_f$')

    # Label axes and save png:
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.ylabel(r'$z$ (m)',fontsize=20)

    plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10],fontsize=16)
    plt.yticks([-2,0,2,8,10,12],[r'$-2$',r'$0$',r'$2$',r'$h_0-2$',r'$h_0$',r'$h_0+2$'],fontsize=16)
    plt.ylim(np.min(bed(X))-2.0,15.0,8)

    plt.xlim(-0.5*Lngth/1000,0.5*Lngth/1000)
    plt.tight_layout()

    lgd = plt.legend([p1,p4,p2,p5,p3,p6],[r'$h$',r'$x_g$',r'$\beta$',r'$\ell$',r'$s>\beta$',r'$h_f$'],fontsize=20,bbox_to_anchor=(0.915, -0.12),ncol=3)

    plt.savefig('pngs/'+str(i),bbox_extra_artists=(lgd,),bbox_inches='tight')
    plt.close()
