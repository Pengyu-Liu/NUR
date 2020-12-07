#question 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', interpolation='nearest', origin='lower')
np.random.seed(121)
positions=np.random.uniform(low=0,high=16,size=(3,1024))

#(a)
def bisection(x,x_new):
    #find the nearest two points to x_new using bisection
    #return the indexes of these two points in the array x
    N=x.size
    #edge1 is the lower point and edge2 is the upper point
    edge1=0
    edge2=N-1
    #periodic boundary conditions
    if x_new<=x[0]:
        return -1,0
    if x_new>x[-1]:
        return -1,0
    while (edge2-edge1)>1:
        middle=int(0.5*(edge2+edge1))
        if x_new>x[middle]:
            #if x_new>x[middle], update the lower edge
            edge1=middle
        else:
            #if x_new<=x[middle], update the upper edge
            edge2=middle

    return edge1,edge2

grid=np.linspace(0.5,15.5,16)
mass_grid=np.zeros((16,16,16))

for i in range(1024):
    #(x,y,z)
    nearp=np.zeros((3,2),dtype='int')
    weight=np.zeros((3,2),dtype='float64')
    #find the nearest 8 points
    for j in range(3):
        nearp[j]=bisection(grid,positions[j,i])
        if nearp[j,0]==-1:
            weight[j,0]=np.min([np.abs(positions[j,i]+0.05),np.abs(positions[j,i]-15.5)])
        else:
            weight[j,0]=abs(positions[j,i]-grid[nearp[j,0]])
#         if nearp[j,1]==0:
#             weight[j,1]=np.min([np.abs(positions[j,i]-0.05),np.abs(positions[j,i]-16.5)])
#         else:
#             weight[j,1]=abs(positions[j,i]-grid[nearp[j,1]])
        weight[j,1]=1-weight[j,0]
    #add a fraction of mass to the 8 points
    for k in range(2):
        for l in range(2):
            for m in range(2):
                 mass_grid[nearp[0,k],nearp[1,l],nearp[2,m]]+=(1-weight[0,k])*(1-weight[1,l])*(1-weight[2,m])

rho_mean=1024/(16**3)
rho=(mass_grid-rho_mean)/rho_mean
#plot rho
fig,axs=plt.subplots(2,2,figsize=(10,10))
for i,ax in zip([4.5,9.5,11.5,14.5],axs.ravel()):
    im=ax.imshow(rho[:,:,int(i)])
    ax.set_title('z={:}'.format(i))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.subplots_adjust(top=0.95)
fig.suptitle('delta')
plt.savefig('./plots/delta.png',dpi=150)
plt.close()

#(b)
def DFT(x,Nj):
    #x is the array we want to do DFT and Nj is its size
    #recursive DFT
    if Nj>1:
        mid=int(0.5*Nj)
        #divide x to left and right half
        DFT(x[:mid],mid)
        DFT(x[mid:],mid)
        for k in range(0,mid):
            #combine left and right
            t=x[k]
            x[k]=t+np.exp(2j*np.pi*k/Nj)*x[k+mid]
            x[k+mid]=t-np.exp(2j*np.pi*k/Nj)*x[k+mid]

def bit_reverse_int(x):
    #x is an int array
    x_new=np.copy(x)
    for i in range(x.size):
        x_new[i]=int(format(x[i],'04b')[::-1],2)
    return x_new


def FFT(x):
    #FFT routinue using cooley-Tukey algorithm, x is a 1d array
    N=x.size
    #bit reverse indices
    reverse_index=bit_reverse_int(np.arange(N))
    result=x[reverse_index].astype(complex)
    DFT(result,N)
    return result

def iDFT(x,Nj):
    #inverse DFT
    #the same as DFT, but j becomes -j
    if Nj>1:
        mid=int(0.5*Nj)
        iDFT(x[:mid],mid)
        iDFT(x[mid:],mid)
        for k in range(0,mid):
            t=x[k]
            x[k]=t+np.exp(-2j*np.pi*k/Nj)*x[k+mid]
            x[k+mid]=t-np.exp(-2j*np.pi*k/Nj)*x[k+mid]
def iFFT(x):
    #inverse FFT
    N=x.size
    #bit reverse
    reverse_index=bit_reverse_int(np.arange(N))
    result=x[reverse_index].astype(complex)/N
    iDFT(result,N)
    return result

#3D FFT
def FFT_3d(cube):
    cube_fft=cube.astype(complex)
    for i in range(cube.shape[0]):
        #implement 1d FFT to each row
        for k in range(cube.shape[1]):
            cube_fft[i,k,:]=FFT(cube[i,k,:])
        #implement 1d FFT to each column
        for k in range(cube.shape[2]):
            cube_fft[i,:,k]=FFT(cube_fft[i,:,k])
    #implement 1d FFT to each aisle
    for i in range(cube.shape[1]):
        for k in range(cube.shape[2]):
            cube_fft[:,i,k]=FFT(cube_fft[:,i,k])
    return cube_fft

def iFFT_3d(cube):
    #inverse 3D FFT
    cube_fft=cube.astype(complex)
    for i in range(cube.shape[0]):
        for k in range(cube.shape[1]):
            cube_fft[i,k,:]=iFFT(cube[i,k,:])
        for k in range(cube.shape[2]):
            cube_fft[i,:,k]=iFFT(cube_fft[i,:,k])
    for i in range(cube.shape[1]):
        for k in range(cube.shape[2]):
            cube_fft[:,i,k]=iFFT(cube_fft[:,i,k])
    return cube_fft

#FFT on rho
rho_fft=FFT_3d(rho)
rho_fft_norm=np.copy(rho_fft)
for l in range(rho.shape[0]):
    #convert k [0,N-1] back to [-N/2,N/2-1]
    if l>(0.5*rho.shape[0]):
                kx=l-rho.shape[0]
    else:
        kx=l
    for m in range(rho.shape[1]):
        if m>(0.5*rho.shape[1]):
                ky=m-rho.shape[1]
        else:
            ky=m
        for n in range(rho.shape[2]):
            if n>(0.5*rho.shape[0]):
                kz=n-rho.shape[0]
            else:
                kz=n
            #Fourier-transformed potential
            rho_fft_norm[l,m,n]=rho_fft[l,m,n]/(kx**2+ky**2+kz**2)
rho_fft_norm[0,0,0]=rho_fft[0,0,0]
#iFFT to get the potential. Potential is the real part
potential=iFFT_3d(rho_fft_norm)
#plot potential for the same slices
fig,axs=plt.subplots(2,2,figsize=(10,10))
for i,ax in zip([4.5,9.5,11.5,14.5],axs.ravel()):
    #because the imaginary part is around the round off error of float64, we can plot real part as absolute value
    im=ax.imshow(potential.real[:,:,int(i)])
    ax.set_title('z={:}'.format(i))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.subplots_adjust(top=0.95)
fig.suptitle('potential')
plt.savefig('./plots/potential.png',dpi=150)
plt.close()
#plot log10(abs(fourier-transformed potential)) for the same slices
fig,axs=plt.subplots(2,2,figsize=(10,10))
for i,ax in zip([4.5,9.5,11.5,14.5],axs.ravel()):
    im=ax.imshow(np.log10(abs(rho_fft_norm[:,:,int(i)])))
    ax.set_title('z={:}'.format(i))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.subplots_adjust(top=0.95)
fig.suptitle('log10(fft(potential))')
plt.savefig('./plots/fftpotential.png',dpi=150)
plt.close()
