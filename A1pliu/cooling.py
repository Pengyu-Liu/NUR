#script for 1. Cooling rates in cosmological simulations

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def bisection(x,x_new):
    #find the nearest two points to x_new using bisection
    #return the indexes of these two points in the array x
    N=x.size
    #edge1 is the lower point and edge2 is the upper point
    edge1=0
    edge2=N-1
    while (edge2-edge1)>1:
        middle=int((edge2+edge1)/2)
        if x_new>x[middle]:
            #if x_new>x[middle], update the lower edge
            edge1=middle
        else:
            #if x_new<=x[middle], update the upper edge
            edge2=middle

    return edge1,edge2

def linear_interp3d(cube,x,y,z,xitp,yitp,zitp):
    #cube is the 3d array that to be interpolated. axis=0 is z-axis, axis=1 is y-axis and axis=2 is x-axis
    #x,y,z are the grids of the cube;
    #xitp,yitp,zitp are coordinates of new points

    #number of new points
    num=xitp.size
    #p is used to store the interpolated values
    p=np.zeros(num)
    for i in range(num):
        z1=zitp[i]
        y1=yitp[i]
        x1=xitp[i]
        #find the nearest points to the new point
        zl,zu=bisection(z,z1)
        yl,yu=bisection(y,y1)
        xl,xu=bisection(x,x1)
        #at zl plane
        #interpolate along x axis
        #the below linear interpolation already contains: if x1=x[xl],p1=cube[zl,yl,xl]; if x1=x[xu], p1=cube[zl,yl,xu]
        p1=(x1-x[xl])*(cube[zl,yl,xu]-cube[zl,yl,xl])/(x[xu]-x[xl])+cube[zl,yl,xl]
        p2=(x1-x[xl])*(cube[zl,yu,xu]-cube[zl,yu,xl])/(x[xu]-x[xl])+cube[zl,yu,xl]
        #interpolate along y axis
        p3=(y1-y[yl])*(p2-p1)/(y[yu]-y[yl])+p1
        #at zu plane
        #interpolate along x axis
        p4=(x1-x[xl])*(cube[zu,yl,xu]-cube[zu,yl,xl])/(x[xu]-x[xl])+cube[zu,yl,xl]
        p5=(x1-x[xl])*(cube[zu,yu,xu]-cube[zu,yu,xl])/(x[xu]-x[xl])+cube[zu,yu,xl]
        #interpolate along y axis
        p6=(y1-y[yl])*(p5-p4)/(y[yu]-y[yl])+p4

        #interpolate along z axis
        p[i]=(z1-z[zl])*(p6-p3)/(z[zu]-z[zl])+p3

    return p


#get the filename we need to read
z=[]
for f_name in os.listdir('CoolingTables'):
    if f_name.endswith('.hdf5'):
        if f_name[2:7]=='colli':
            continue
        elif f_name[2:7]=='photo':
            continue
        elif f_name[2:7] not in z:
            z.append(f_name[2:7])

z.sort()

#create arrays
Z=np.array(z,dtype='float')
cool_metalfree=np.zeros((Z.size,352,81))
cool_metal=np.zeros_like(cool_metalfree)
ne_nh=np.zeros_like(cool_metalfree)
ne_nh_sol=np.zeros_like(cool_metalfree)

#read in data
f=h5py.File('CoolingTables/z_{:}.hdf5'.format(z[0]),'r')
#Temperature bins
T=np.array(f['Total_Metals/Temperature_bins'])
#Hydrogen density bins
H=np.array(f['Total_Metals/Hydrogen_density_bins'])
#ne/nH in the solar system
ne_nh_sol[0]=np.array(f['Solar/Electron_density_over_n_h'])
#He to H mass fraction=0.258([2,])
#Because the electron density contributed from heavy elements is very small, I take the electron density from He and H as the ne/nH
ne_nh[0]=np.array(f['Metal_free/Electron_density_over_n_h'][2,])
cool_metalfree[0]=np.array(f['Metal_free/Net_Cooling'][2,])
#Because the coefficient for every metal element is the same, I use the net_cooling from Total_Metals
cool_metal[0]=np.array(f['Total_Metals/Net_cooling'])


metallicity=0.25
for i in range(1,Z.size):
    f=h5py.File('CoolingTables/z_{:}.hdf5'.format(z[i]),'r')
    ne_nh[i]=np.array(f['Metal_free/Electron_density_over_n_h'][2,])
    cool_metalfree[i]=np.array(f['Metal_free/Net_Cooling'][2,])
    cool_metal[i]=np.array(f['Total_Metals/Net_cooling'])
    ne_nh_sol[i]=np.array(f['Solar/Electron_density_over_n_h'])


#(a)
T_new=np.copy(T)
Z_new=np.zeros_like(T)
Z_new[:]=3
H_new=np.zeros_like(T)
H_den=np.array([1,1e-2,1e-4,1e-6])


for i in range(H_den.size):
    H_new[:]=H_den[i]
    #interpolate
    cool_metalfree1=linear_interp3d(cool_metalfree,H,T,Z,H_new,T_new,Z_new)
    cool_metal1=linear_interp3d(cool_metal,H,T,Z,H_new,T_new,Z_new)
    ne_nh1=linear_interp3d(ne_nh,H,T,Z,H_new,T_new,Z_new)
    ne_nh_sol1=linear_interp3d(ne_nh_sol,H,T,Z,H_new,T_new,Z_new)
    #calculate total cooling rate
    total_cool=cool_metalfree1+metallicity*cool_metal1*ne_nh1/ne_nh_sol1
    plt.loglog(T,total_cool,label='nH={:}'.format(H_den[i]))

plt.legend()
plt.xlabel('T(K)')
plt.ylabel('total cooling rate/n_H^2 (erg s^-1 cm^3)')
plt.title('z=3')
plt.savefig('./plots/cooling1a.png')
plt.close()



#(b)interpolate z from 0 to 8.989
metallicity=0.5
Z_b=np.linspace(0,Z[-1],100)
H_new[:]=1e-4


for i in tqdm(range(100)):
    Z_new[:]=Z_b[i]
    #interpolate
    cool_metalfree1=linear_interp3d(cool_metalfree,H,T,Z,H_new,T_new,Z_new)
    cool_metal1=linear_interp3d(cool_metal,H,T,Z,H_new,T_new,Z_new)
    ne_nh1=linear_interp3d(ne_nh,H,T,Z,H_new,T_new,Z_new)
    ne_nh_sol1=linear_interp3d(ne_nh_sol,H,T,Z,H_new,T_new,Z_new)
    #calculate total cooling rate
    total_cool=cool_metalfree1+metallicity*cool_metal1*ne_nh1/ne_nh_sol1
    plt.loglog(T,total_cool)
    plt.xlabel('T(K)')
    plt.ylabel('total cooling rate/n_H^2 (erg s^-1 cm^3)')
    plt.title('z={:1.4f}'.format(Z_b[i]))
    plt.ylim(1e-24,1e-19)
    plt.savefig('./plots/snap%04d.png'%i)
    plt.close()
