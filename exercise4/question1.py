#q1
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# we need to import the time module from astropy
from astropy.time import Time
# import some coordinate things from astropy
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u
from astropy import constants as const
#(a)
print('run 1a')
# set the current time
t = Time('2020-12-07 10:00')
solar=['sun','mercury','venus','earth','moon','mars','jupiter','saturn','uranus','neptune']
position=np.zeros((10,3))
velocity=np.zeros((10,3))
with solar_system_ephemeris.set('jpl'):
    for i,j in enumerate(solar):
        body = get_body_barycentric_posvel(j, t)
        position[i,0]=body[0].x.to_value(u.AU)
        position[i,1]=body[0].y.to_value(u.AU)
        position[i,2]=body[0].z.to_value(u.AU)
        velocity[i,0]=body[1].x.to_value(u.AU/u.d)
        velocity[i,1]=body[1].y.to_value(u.AU/u.d)
        velocity[i,2]=body[1].z.to_value(u.AU/u.d)

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.scatter(position[:,0],position[:,1])
for i in range(10):
    plt.text(position[i,0]+0.2,position[i,1]+0.2,solar[i],fontsize=8)
plt.xlabel('x[AU]')
plt.ylabel('y[AU]')
plt.subplot(122)
plt.scatter(position[:,0],position[:,2])
for i in range(10):
    plt.text(position[i,0]+0.2,position[i,2]+0.2,solar[i],fontsize=8)
plt.xlabel('x[AU]')
plt.ylabel('z[AU]')
plt.tight_layout()
plt.savefig('./plots/position_1a.png',dpi=200)
plt.close()

#(b) only consider the force between the sun on other planets, including moon
print('run 1b')
def distance(vector):
    return np.sqrt((np.sum(np.square(vector))))
#0.5 day
step=0.5
step_num=int((200*u.year).to(u.day)/(step*u.day))
G=(const.G).to(u.au**3/(u.kg*u.day**2)).value
#orbit of solar system objects: axis=0, object; axis=1, step; axis=2, x,y,z
orbit_b=np.zeros((10,step_num,3))
orbit_b[:,0,:]=np.copy(position)
#unit: kg
mass=[const.M_sun.value,3.385e23,4.867e24, const.M_earth.value, 7.348e22, 6.39e23, const.M_jup.value,
            5.683e26, 8.681e25, 1.024e26]
#initialization velocity (0.5*step)
orbit_vel_b=np.copy(velocity)
a_sun=0
for j in range(1,10):
        #calculate ditance between sun and planets
        d=orbit_b[0,0]-orbit_b[j,0]
        #calculate the acceleration of each planet
        force =d*G/(distance(d)**3)
        #inital kick
        orbit_vel_b[j]+= 0.5*step*force*mass[0]
        #sun
        a_sun-=force*mass[j]
#initial kick of sun
orbit_vel_b[0]+=0.5*step*a_sun

#start
for i in range(1,step_num):
    a_sun=0
    for j in range(1,10):
        #planet drifts
        orbit_b[j,i,:]=orbit_b[j,i-1,:]+step*orbit_vel_b[j]
        #calculate ditance between sun and planets
        d=orbit_b[0,i]-orbit_b[j,i]
        #calculate the acceleration of each planet
        force =d*G/(distance(d)**3)
        #sun kicks the planet
        orbit_vel_b[j]+= step*force*mass[0]
        #sun
        a_sun-=force*mass[j]
    #sun drifts
    orbit_b[0,i,:]=orbit_b[0,i-1,:]+step*orbit_vel_b[0]
    #kick
    orbit_vel_b[0]+=step*a_sun

time=np.arange(step_num)*step/365
def plot_function1(orbit,ordd):
    plt.figure(figsize=(8,5))
    plt.subplot(321)
    plt.plot(time,orbit[ordd[0],:,0],label=solar[ordd[0]])
    plt.ylabel('x[AU]')
    plt.legend(loc='upper right')
    plt.subplot(322)
    plt.plot(time,orbit[ordd[1],:,0],label=solar[ordd[1]])
    plt.ylabel('x[AU]')
    plt.legend(loc='upper right')
    plt.subplot(323)
    plt.plot(time,orbit[ordd[0],:,1],label=solar[ordd[0]])
    plt.ylabel('y[AU]')
    plt.subplot(324)
    plt.plot(time,orbit[ordd[1],:,1],label=solar[ordd[1]])
    plt.ylabel('y[AU]')
    plt.subplot(325)
    plt.plot(time,orbit[ordd[0],:,2],label=solar[ordd[1]])
    plt.ylabel('z[AU]')
    plt.xlabel('time[year]')
    plt.subplot(326)
    plt.plot(time,orbit[ordd[1],:,2],label=solar[ordd[1]])
    plt.ylabel('z[AU]')
    plt.xlabel('time[year]')
    plt.tight_layout()

def plot_function(orbit,planets):
    #plot orbit from venus to neptune
    plt.figure(figsize=(8,5))
    plt.subplot(311)
    for i in planets:
        plt.plot(time,orbit[i,:,0],label=solar[i])
    plt.ylabel('x[AU]')
    plt.legend(loc='upper right')
    plt.subplot(312)
    for i in planets:
        plt.plot(time,orbit[i,:,1],label=solar[i])
    plt.ylabel('y[AU]')
    plt.subplot(313)
    for i in planets:
        plt.plot(time,orbit[i,:,2],label=solar[i])
    plt.ylabel('z[AU]')
    plt.xlabel('time[year]')
    plt.tight_layout()

plot_function1(orbit_b,ordd=[0,1])
plt.savefig('./plots/sun_mec_1b.png',dpi=200)
plt.close()
plot_function(orbit_b,planets=np.arange(2,4))
plt.savefig('./plots/venus_earth_1b.png',dpi=200)
plt.close()
plot_function(orbit_b,planets=[4])
plt.savefig('./plots/moon_1b.png',dpi=200)
plt.close()
plot_function(orbit_b,planets=np.arange(5,10))
plt.savefig('./plots/mars_1b.png',dpi=200)
plt.close()

#(c)pp interactions
print('run 1c')
orbit_c=np.zeros((10,step_num,3))
orbit_c[:,0,:]=np.copy(position)
#initialization velocity (0.5*step)
orbit_vel_c=np.copy(velocity)
#force tensor
force=np.zeros((10,10,3))
for i in range(10):
    for j in range(i+1,10):
        #calculate distance between particles
        d=orbit_c[i,0]-orbit_c[j,0]
        #calculate the force between each particle pairs
        force[i,j] =-d*G/(distance(d)**3)
        force[j,i] =- force[i,j]
#calculate total acceleration for each particle
for i in range(10):
    force[:,i,:]=force[:,i,:]*mass[i]
accel=np.sum(force,axis=1)
#initial kick
orbit_vel_c+=0.5*step*accel

#start
for k in range(1,step_num):
    #drift
    orbit_c[:,k,:]=orbit_c[:,k-1,:]+step*orbit_vel_c
    #calculate next acceleration
    force=np.zeros((10,10,3))
    for i in range(10):
        for j in range(i+1,10):
            d=orbit_c[i,k]-orbit_c[j,k]
            force[i,j] =-d*G/(distance(d)**3)
            force[j,i] =- force[i,j]
    #calculate total acceleration for each particle
    for i in range(10):
        force[:,i,:]=force[:,i,:]*mass[i]
    accel=np.sum(force,axis=1)
    #kick
    orbit_vel_c+=step*accel

plot_function1(orbit_c,ordd=[0,1])
plt.savefig('./plots/sun_mec_1c.png',dpi=200)
plt.close()
plot_function(orbit_c,planets=np.arange(2,4))
plt.savefig('./plots/venus_earth_1c.png',dpi=200)
plt.close()
plot_function(orbit_c,planets=[4])
plt.savefig('./plots/moon_1c.png',dpi=200)
plt.close()
plot_function(orbit_c,planets=np.arange(5,10))
plt.savefig('./plots/mars_1c.png',dpi=200)
plt.close()

#(d) Runge-Kutta 4th
print('run 1d')
#Because Runge-Kutta 4th calculates 4 orders in each time step, the accuracy is high.
# To save the time, I set the time step here to be 1day.
step=1
step_num=int((200*u.year).to(u.day)/(step*u.day))

def accel_matrix(location):
    #calculate the acceleration of all particles at a given location
    force=np.zeros((10,10,3))
    for i in range(10):
        for j in range(i+1,10):
            d=location[i]-location[j]
            force[i,j] =-d*G/(distance(d)**3)
            force[j,i] =- force[i,j]
    for i in range(10):
        force[:,i,:]=force[:,i,:]*mass[i]
    accel=np.sum(force,axis=1)
    return accel

orbit_d=np.zeros((10,step_num,3))
orbit_d[:,0,:]=np.copy(position)
orbit_vel_d=np.copy(velocity)

#start
for k in range(1,step_num):
    accel1=accel_matrix(orbit_d[:,k-1,:])
    k1=step*orbit_vel_d
    accel2=accel_matrix(orbit_d[:,k-1,:]+0.5*k1)
    orbit_vel2=orbit_vel_d+0.5*step*accel2
    k2=step*orbit_vel2
    accel3=accel_matrix(orbit_d[:,k-1,:]+0.5*k2)
    orbit_vel3=orbit_vel_d+0.5*step*accel3
    k3=step*orbit_vel3
    accel4=accel_matrix(orbit_d[:,k-1,:]+k3)
    orbit_vel4=orbit_vel_d+step*accel4
    k4=step*orbit_vel4
    #weighted combination of position
    orbit_d[:,k,:]=orbit_d[:,k-1,:]+k1/6+k2/3+k3/3+k4/6
    #weighted combination of velocity
    orbit_vel_d=orbit_vel_d+step*(accel1/6+accel2/3+accel3/3+accel4/6)

plt.figure(figsize=(5,5))
order=[0,2,3,4,5,6,7,8,9]
time2=np.arange(step_num)*step/365
for i in order:
    plt.plot(time2,orbit_d[i,:,0],label='rk '+solar[i])
    plt.plot(time,orbit_c[i,:,0],'-.',ms=2,label='lp '+solar[i])
plt.xlabel('time[year]')
plt.ylabel('x[AU]')
plt.legend(bbox_to_anchor=(1.5, 1),loc='upper right')
plt.savefig('./plots/all_x_1d.png',bbox_inches="tight",dpi=200)


plt.figure(figsize=(5,5))
for i in order:
    plt.plot(time2,orbit_d[i,:,1],label='rk '+solar[i])
    plt.plot(time,orbit_c[i,:,1],'-.',label='lp '+solar[i])
plt.xlabel('time[year]')
plt.ylabel('y[AU]')
plt.legend(bbox_to_anchor=(1.5, 1),loc='upper right')
plt.savefig('./plots/all_y_1d.png',bbox_inches="tight",dpi=200)

plt.figure(figsize=(5,5))
for i in order:
    plt.plot(time2,orbit_d[i,:,2],label='rk '+solar[i])
    plt.plot(time,orbit_c[i,:,2],'-.',label='lp '+solar[i])
plt.xlabel('time[year]')
plt.ylabel('z[AU]')
plt.legend(bbox_to_anchor=(1.5, 1),loc='upper right')
plt.savefig('./plots/all_z_1d.png',bbox_inches="tight",dpi=200)

plt.figure(figsize=(8,5))
plt.subplot(311)
plt.plot(time,orbit_b[1,:,0],label='lp mercury')
plt.plot(time2,orbit_d[1,:,0],label='rk mercury')
plt.ylabel('x[AU]')
plt.ylim(-2,10)
plt.legend(loc='upper right')
plt.subplot(312)
plt.plot(time,orbit_b[1,:,1],label='lp mercury')
plt.plot(time2,orbit_d[1,:,1],label='rk mercury')
plt.ylabel('y[AU]')
plt.ylim(-2,10)
plt.legend(loc='upper right')
plt.subplot(313)
plt.plot(time,orbit_b[1,:,1],label='lp mercury')
plt.plot(time2,orbit_d[1,:,1],label='rk mercury')
plt.ylabel('z[AU]')
plt.ylim(-2,10)
plt.legend(loc='upper right')
plt.xlabel('time[year]')
plt.tight_layout()
plt.savefig('./plots/mercury_1d.png',dpi=200)
