#Question3

import numpy as np
import matplotlib.pyplot as plt

#(a)
#function that need to be integrated
def f_itg(x,a=2.2,b=0.5,c=3.1):
    return np.power(x,a-1)*np.exp(-np.power(x/b,c))

def simpson(f,a,b,N):
    #simpson integration
    #f: integrand; a: lower limit; b: upper limit; N: number of intervals
    h=(b-a)/N
    #N+1 points
    x=np.linspace(a,b,N+1)
    y=f(x)

    #if N is even
    if N%2 ==0:
        slice1=np.arange(0,N-1,2)
        slice2=np.arange(1,N,2)
        slice3=np.arange(2,N+1,2)
        result=(h/3)*(np.sum(y[slice1])+np.sum(y[slice3])+4*np.sum(y[slice2]))

    else:
        #if N is odd, the last interval uses the trapzoid
        slice1=np.arange(0,N-2,2)
        slice2=np.arange(1,N-1,2)
        slice3=np.arange(2,N,2)
        result=(h/3)*(np.sum(y[slice1])+np.sum(y[slice3])+4*np.sum(y[slice2]))+h*0.5*(y[N-1]+y[N])

    return result


b=0.5
a=2.2
#1000 intervals can already gives a very good resutls
integ=np.power(b,3-a)*4*np.pi*simpson(f_itg,0,5,1000)
A=1/integ
fA=open('A.txt','w')
fA.write(str(A))
fA.close()


#(b)
def linear_interp(x1,y1,x_new1,log):
    #boolean:log. If log=True, then do interpolation in log space
    #x1 and y1 are known points, x_new1 are x coordinates of new points
    if log==True:
        x=np.log(x1)
        y=np.log(y1)
        x_new=np.log(x_new1)
    else:
        #otherwise do interpolation in linear space
        x=np.copy(x1)
        y=np.copy(y1)
        x_new=np.copy(x_new1)

    num=x_new.size
    y_new=np.zeros_like(x_new)
    N=x.size
    for i in range(num):
        #bisection: find the nearest two points
        edge1=0
        edge2=N-1
        while (edge2-edge1)>1:
            middle=int((edge2+edge1)/2)
            if x_new[i]>x[middle]:
                edge1=middle
            else:
                edge2=middle
        #calculate slope and interpolate
        y_new[i]=(x_new[i]-x[edge1])*(y[edge2]-y[edge1])/(x[edge2]-x[edge1])+y[edge1]

    if log==True:
        return np.exp(y_new)
    else:
        return y_new

#function n(x)
def f(x,a=2.2,b=0.5,c=3.1):
    return np.power(x/b,a-3)*np.exp(-np.power(x/b,c))

Nsat=100
x1=np.array([1e-4,1e-2,1e-1,1,5])
y1=A*Nsat*f(x1)
#Because f(5) is extremly close to 0 and underflows in the 64bit float type, I truncate f(5) to be 1e-30.
#Though it causes an error, I know the error instead of unknown underflow error.
y1[-1]=1e-30
plt.loglog(x1,y1,'o')
#The five data points change dramatically in linear space. When we plot them in log space,
#they show piecewise patterns. Point 1-4 decreases slowly, but the last point is very small
#compared to the former 4 points.
#Because there are only 5 points and x,y span in a large range, I chose to do interpolation in loglog space.
#It is a piecewise function based on just these points, lagrange polynomial like Neville's algorithm can
#produce large wrinkles between these points. Cubic spline can interpolate piecewise functions when
#the functions are smooth (1st and 2nd derivatives are continuous.)
#Cubic spline takes all points into consideration and
#can also cause large wrinkles between points if points change dramatically, which is our case.
#So cubic spline is not a good choice. Akima spline should be a good choice in this case because it
#can give natural and smooth results based on a small number of points. However, I have little time
#to implement it. Instead, I chose to do the linear interpolation in log space, because it can also produce
#results that do not deviate far from the known points, which is a convenient way to do interpolation when
#there are only a few points and changes dramatically at some range.

#produce points for interpolation
x=np.linspace(0.0001,0.01,10)
x=np.append(x,np.linspace(0.02,0.1,10))
x=np.append(x,np.linspace(0.2,1.3,10))
x=np.append(x,5)

#linear interpolate in loglog space
yitp=linear_interp(x1,y1,x,log=True)

plt.loglog(x,yitp,'-.',label='linear interp')
plt.xlabel('x')
plt.ylabel('n(x)')
plt.legend()
plt.savefig('./plots/3binterp.png')
plt.close()

#c
def poisson(lm,k):
    #lm:lambda(mean), k
    if k==0:
        #because of factorial, we need to calculate differently when k=0
        return np.exp(-lm)
    else:
        #to prevent overflow, calculate at log space
        arr=np.arange(1,k+1)
        p=k*np.log(lm)-lm-np.sum(np.log(arr))
        #return p to linear
        return np.exp(p)

#write out results to 3c.txt
file=open('3c.txt','w')
file.write('\nP(1,0)='+str(poisson(1,0)))
file.write('\nP(5,10)='+str(poisson(5,10)))
file.write('\nP(3,21)='+str(poisson(3,21)))
file.write('\nP(2.6,40)='+str(poisson(2.6,40)))
file.write('\nP(101,200)='+str(poisson(101,200)))
file.close()
