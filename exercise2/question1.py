#question1
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#(a)
def RNG(seed,low,up,n):
    #Seed is the seed. Low is the lower limit and up is the upper limit.
    #n is the number of random numbers we need to generate
    ran=np.zeros(n)
    init=np.uint64(seed)
    a1=np.uint64(21)
    a2=np.uint64(35)
    a3=np.uint64(4)
    #MLCG
    m=2**64
    a=2685821657736338717
    for i in range(n):
        #64 bit xor shift
        x=init
        x=x^(x>>a1)
        x=x^(x<<a2)
        x=x^(x>>a3)
        #x as input for a multip linear congruential generator
        Id=np.uint64(int(x)*a%m)
        #use the high 32bits
        ran[i]=low+(up-low)*(Id>>np.uint(32))/(2**32)
        #update the state of next number
        init=Id
    return ran

print('run (a)')
seed=31
print('The seed is:',seed)
a1=RNG(seed,0,1,1000)

plt.scatter(a1[:-1],a1[1:])
plt.xlabel('xi')
plt.ylabel('xi+1')
plt.title('1000 random numbers')
plt.savefig('./plots/Q1ascatter.png',dpi=150)
plt.close()
a2=RNG(seed,0,1,1000000)
yerr=np.sqrt(1e6*0.05)

entries, edges, _ =plt.hist(a2,bins=20,range=(0,1),histtype='step',label='sample')
#plot poisson uncertainties(the same for all bins)
bin_centers=0.5*(edges[:-1]+edges[1:])
yexp=np.ones_like(bin_centers)*1e6*0.05
plt.errorbar(bin_centers,yexp,yerr=np.sqrt(1e6*0.05),fmt='r.',capsize=3,label='Poisson')
#only show [45000,55000] to see the number more clearly
plt.ylim(45000,55000)
plt.xlabel('bin')
plt.ylabel('number')
plt.title('1000000 random numbers histogram')
plt.legend()
plt.savefig('./plots/Q1ahistogram.png',dpi=150)
plt.close()
#calculate the Pearson correlation coefficient for 100000 numbers
#r(xi,xi+1)
ri_i1=(np.mean(a2[:99999]*a2[1:100000])-np.mean(a2[:99999])*np.mean(a2[1:100000]))/(np.std(a2[:99999])*np.std(a2[1:100000]))
#r(xi,xi+2)
ri_i2=(np.mean(a2[:99998]*a2[2:100000])-np.mean(a2[:99998])*np.mean(a2[2:100000]))/(np.std(a2[:99998])*np.std(a2[2:100000]))

print('The Pearson correlation coefficient r(xi,xi+1) is: ',ri_i1)
print('The Pearson correlation coefficient r(xi,xi+2) is: ',ri_i2)
#write out result
file1=open('Q1a.txt','w')
file1.write('\nThe seed is: ')
file1.write(str(seed))
file1.write('\nThe Pearson correlation coefficient r(xi,xi+1) is:{:.6f}'.format(ri_i1))
file1.write('\nThe Pearson correlation coefficient r(xi,xi+2) is:{:.6f}'.format(ri_i2))
file1.close()

#(b)
print('run (b)')
def Hernquist(r):
    Mdm=1e12
    #unit of a:kpc
    a=80
    return Mdm*a/(2*np.pi*r*np.power(r+a,3))


def cdf_Hernq(r):
    a=80
    #the cumulative distribution function of the Hernquist function(normalized to 1)
    return np.power(r/(a+r),2)

#the inverse of CDF
def invercdf(x):
    a=80
    return a*np.sqrt(x)/(1-np.sqrt(x))
#sampling, a2 is the result of 1a(1e6 uniformed numbers in [0,1])
inver=invercdf(a2)
inver_sum,inver_bin=np.histogram(inver,bins=10000,range=(0,10000))
inver_frac=np.zeros_like(inver_sum)
#the enclosed fraction of particles
for i in range(inver_frac.size):
    inver_frac[i]=np.sum(inver_sum[:i+1])
inver_frac=inver_frac/np.sum(inver_sum)
plt.plot(inver_bin[1:],inver_frac,label='sampled particles fraction')

#the expected amount of enclosed fraction of mass
radius=np.linspace(0,10000,1000)
expected_frac=cdf_Hernq(radius)
plt.plot(radius,expected_frac,label='expect mass fraction')
plt.legend()
plt.xlabel('r[kpc]')
plt.ylabel('fraction')
plt.savefig('./plots/Q1bfraction.png',dpi=150)
plt.close()

#(c)
print('run (c)')
#phi using 1000 random numbers in a2
phi=2*np.pi*np.copy(a2[10000:11000])
#theta using another 1000 random numbers in a2
theta=np.arccos(1-2*np.copy(a2[20000:21000]))
x=np.copy(inver[:1000])*np.sin(theta)*np.cos(phi)/(4*np.pi*np.sqrt(inver[:1000]))
y=np.copy(inver[:1000])*np.sin(theta)*np.sin(phi)/(4*np.pi*np.sqrt(inver[:1000]))
z=np.copy(inver[:1000])*np.cos(theta)/(4*np.pi*np.sqrt(inver[:1000]))
#3d scatter plot
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z)
ax.set_xlabel('x[kpc]')
ax.set_ylabel('y[kpc]')
ax.set_zlabel('z[kpc]')
ax.set_title('3D scatter plot of 1000 particles')
plt.savefig('./plots/Q1c3d.png',dpi=150)
plt.close()
#make a plot of theta and phi
fig2=plt.figure()
ax2=fig2.add_subplot(111,projection='3d')
ax2.scatter(np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta))
ax2.set_xlabel('x[kpc]')
ax2.set_ylabel('y[kpc]')
ax2.set_zlabel('z[kpc]')
ax2.set_title('random numbers on a sphere')
plt.savefig('./plots/Q1csphere.png',dpi=150)
plt.close()

#(d)
print('run (d)')
#ridder's method for differentiation
def ridder_der(f,x,h0=0.1,m=10,d=2):
    #m:order; h0:initial interval(need to be adjusted for different functions)
    #two arrays to store previous and current results
    rdr=np.zeros(m)
    rdr2=np.zeros(m)
    hh=h0
    for i in range(m):
        #central difference
        rdr[i]=0.5*(f(x+hh)-f(x-hh))/hh
        hh=hh/d

    result=np.copy(rdr[0])
    den=1
    mul=d**2
    #initial error(larger number)
    err=10
    for i in range(1,m):
        den*=mul
        for j in range(m-i):
            #calculate new values
            rdr2[j]=(den*rdr[j+1]-rdr[j])/(den-1)
            errn=max(abs(rdr2[j]-rdr[j]),abs(rdr2[j]-rdr[j+1]))
            if errn<err:
                #compare recent results: if error goes down, go to higher order
                err=errn
                result=rdr2[j]
            #terminate early if the error grows
            elif errn>3*err:
                 return result
        #put new values to the old array
        rdr=np.copy(rdr2)
    return result

numer_result=ridder_der(Hernquist,x=1.2*80,h0=1,m=10,d=2)
print('The numerical result is:{:.10f}'.format(numer_result))

#analytical result of Hernquist
def Hernquist_rd(r):
    Mdm=1e12
    #unit of a:kpc
    a=80
    return (Mdm*a*0.5/np.pi)*(-(1/(np.power(r,2)*np.power(r+a,3)))-3/(r*np.power(r+a,4)))
analy_result=Hernquist_rd(1.2*80)
print('The analytical result is:{:.10f}'.format(analy_result))

#write out result
file2=open('Q1d.txt','w')
file2.write('The numerical result is:{:.10f}'.format(numer_result))
file2.write('\nThe analytical result is:{:.10f}'.format(analy_result))
file2.close()

#(e)
print('run (e)')
def NR(f,a,h=0.001):
    #Newton-Raphson method
    #use ridder's differentiation to calculate the derivative, h is the initial h0 for ridder_der
    #accuracy
    acc=1e-6
    #the maximum number of bisection iteration
    term=int(1e3)
    x0=a
    for i in range(term):
        #notice: need to adjust h for different function
        x0_der=ridder_der(f,x0,h0=h)
        if x0_der==0:
            #if the derivative ==0, return this point, though it's not a good result
            return x0
        x1=x0-f(x0)/x0_der
        if abs(f(x1))<acc:
            #find the root successfully
            return x1
        x0=x1
    #don't find a root with an accuracy smaller than acc under the maximum number of iterations
    #return the most recent one
    return x0

def f1(x):
    pc=150
    return Hernquist(x)-200*pc

def f2(x):
    pc=150
    return Hernquist(x)-500*pc

def mass(x):
    Mdm=1e12
    a=80
    return Mdm*np.power(x,2)/np.power(a+x,2)

#using Newton-Raphson method to find the root
R200=NR(f1,1,h=0.0001)
R500=NR(f2,1,h=0.0001)
M200=mass(R200)
M500=mass(R500)
#check
print('check function(R200) value:',f1(R200))

print('R200[kpc] is:',R200)
print('M200[Msolar] is:',M200)


#check
print('check function(R500) value:',f2(R500))
print('R500[kpc] is:',R500)
print('M500[Msolar] is:',M500)
#write out results
file3=open('Q1e.txt','w')
file3.write('check function(R200) value:{:}'.format(f1(R200)))
file3.write('\nR200[kpc] is:{:.6f}'.format(R200))
file3.write('\nM200[solar] is:{:.6e}'.format(M200))
file3.write('\ncheck function(R500) value:{:}'.format(f2(R500)))
file3.write('\nR500[kpc] is:{:.6f}'.format(R500))
file3.write('\nM500[solar] is:{:.6e}'.format(M500))
file3.close()

#(f)
print('run (f)')
def mergesort(arr,left,right,index):
    #mergesort arr from arr[left] to arr[right]
    #caution: arr will be changed to the sorted array!
    #Copy it before using this function if you want to keep the initial arr.
    #also return the index after sorting
    #recursion
    if left<right:
        mid=int(0.5*(left+right))
        #sort the left part
        mergesort(arr,left,mid,index)
        #sort the right part
        mergesort(arr,mid+1,right,index)

        #merge and sort arr from arr[left] to arr[right]. mid is the last index of the left part
        #Both of arr[left:mid+1] and arr[mid+1:right] are already sorted
        i=left
        j=mid+1
        arr_new=np.zeros_like(arr[left:right+1])
        index_new=np.zeros_like(index[left:right+1])
        d=0
        while i<=mid and j<=right:
            #put the smaller one to arr_new[d]
            if arr[i]<=arr[j]:
                arr_new[d]=arr[i]
                index_new[d]=index[i]
                i+=1
            else:
                arr_new[d]=arr[j]
                index_new[d]=index[j]
                j+=1
            d+=1
        #One part is ended and all of the rest of another part are smaller or larger than previous elements.
        if i<=mid:
            #put the rest of the left part to arr_new(because are they sorted, we put them directly to arr_new)
            arr_new[d:]=np.copy(arr[i:mid+1])
            index_new[d:]=np.copy(index[i:mid+1])
        else:
            #put the rest of the right part to arr_new
            arr_new[d:]=np.copy(arr[j:right+1])
            index_new[d:]=np.copy(index[j:right+1])
        #put arr_new back to arr
        arr[left:right+1]=arr_new
        index[left:right+1]=index_new

#minization
def downhill_simplex(f,init_x):
    #downhill method to find the minimum of f
    #N dimension requires N+1 points
    #dimention
    N=init_x.size
    #generate N+1 points
    step1=-100
    step2=10
    point=np.zeros((N+1,N))
    point[0]=np.copy(init_x)
    fun=np.zeros(N+1)
    fun[0]=f(point[0])
    for i in range(N):
        point[i+1,0]=point[i,0]+step1
        point[i+1,0]=point[i,0]+step2
        fun[i+1]=f(point[i+1])
    #maximum number of iteration
    term=10000
    #target accuracy
    acc=1e-10
    #record the best point of each iterations
    trace=np.copy(point[0])

    #start iteration
    for i in range(term):
        #sort
        fun_ind=np.arange(N+1)
        mergesort(fun,0,N,fun_ind)
        point=point[fun_ind]
        #check if find the minimum
        ran=abs((fun[0]-fun[-1])/(0.5*(fun[0]+fun[-1])))
        if ran<acc:
            #find the best guess x0
            trace=np.vstack((trace,point[0]))
            return point[0],trace
        #centroid of the first N points
        xcen=np.mean(point[:-1],axis=0)
        #propose a new point by reflecting
        xtry=2*xcen-point[-1]
        if f(xtry)>=fun[0] and f(xtry)<fun[-1]:
            #new points is better but not the best,accept it
            point[-1]=np.copy(xtry)
            fun[-1]=f(xtry)
            trace=np.vstack((trace,point[0]))
        elif f(xtry)<fun[0]:
            #new point is the very best
            #propose a new point by expanding further
            xexp=2*xtry-xcen
            if f(xexp)<f(xtry):
                #xexp is better, accept it
                point[-1]=np.copy(xexp)
                fun[-1]=f(xexp)
                trace=np.vstack((trace,xexp))
            else:
                #accept the reflected one
                point[-1]=np.copy(xtry)
                fun[-1]=f(xtry)
                trace=np.vstack((trace,xtry))
        else:
            #xtry is the baddest
            trace=np.vstack((trace,point[0]))
            #propose a new point by contracting
            xtry=0.5*(xcen+point[-1])
            if f(xtry)<fun[-1]:
                #accept
                point[-1]=np.copy(xtry)
                fun[-1]=f(xtry)
            else:
                #zoom on the best point by contracting all others to it
                point[1:]=0.5*(xcen+point[1:])
                for i in range(N):
                    fun[i+1]=f(point[i+1])
    #the target accuracy is not reached after the maximum of iterations
    #return the current best one and trace
    return point[0],trace

def Hernquist2d(var):
    x,y=var
    a=80
    Mdm=1e12
    G=4.3*1e-6 #unit:kpc km^2 s^-2 Msol^-1
    return -G*Mdm/(a+np.sqrt(np.power(x-1.3,2)+2*np.power(y-4.2,2)))

def distance(point,end):
    return np.sqrt(np.sum(np.power(point-end,2)))

final_point,trace= downhill_simplex(Hernquist2d,np.array([-1000,-200]))

#calculate the distance from the final point
dis=np.zeros(trace.shape[0])
for i in range(trace.shape[0]):
    dis[i]=distance(trace[i],trace[-1])

print('The minimum of this potential is {:}'.format(Hernquist2d(final_point)))
print('The final point is',final_point)

file4=open('Q1f.txt','w')
file4.write('The minimum of this potential is {:.6f}'.format(Hernquist2d(final_point)))
file4.write('\nThe final point(x,y) is:{:}'.format(final_point))
file4.close()

plt.plot(dis)
plt.xlabel('the number of iterations')
plt.ylabel('distance')
plt.savefig('./plots/Q1f.png',dpi=150)
plt.close()
