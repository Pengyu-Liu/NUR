#q1
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gammainc
import warnings
warnings.filterwarnings("ignore")

def simpson(f,xl,xu,N,args):
    #f: integrand xl: lower limit; xu: upper limit. N: number of intervals
    h=(xu-xl)/N
    #N+1 points
    x=np.linspace(xl,xu,N+1)
    y=f(x,args)

    #if N is even
    if N%2 ==0:
        slice1=np.arange(0,N-1,2)
        slice2=np.arange(1,N,2)
        slice3=np.arange(2,N+1,2)
        result=(h/3)*(np.sum(y[slice1])+np.sum(y[slice3])+4*np.sum(y[slice2]))

    else:
        #if N is odd, the last interval using the trapzoid
        slice1=np.arange(0,N-2,2)
        slice2=np.arange(1,N-1,2)
        slice3=np.arange(2,N,2)
        result=(h/3)*(np.sum(y[slice1])+np.sum(y[slice3])+4*np.sum(y[slice2]))+h*0.5*(y[N-1]+y[N])

    return result


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

        #sort arr from arr[left] to arr[right]. mid is the last index of the left part
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

#minimization
def downhill_simplex(f,init_x,other1,other2,other3):
    #downhill method to find the minimum of f
    #N dimension requires N+1 points
    #dimention
    N=init_x.size
    #generate N+1 points
    step1=-0.1
    step2=0.3
    step3=-0.5
    point=np.zeros((N+1,N))
    point[0]=np.copy(init_x)
    fun=np.zeros(N+1)
    fun[0]=f(point[0],other1,other2,other3)
    for i in range(N):
        if i==0:
            point[i+1]=point[i]+step1
        if i==1:
            point[i+1,1]=point[i,1]+step2
        if i==2:
            point[i+1,2]=point[i,2]+step3
        fun[i+1]=f(point[i+1],other1,other2,other3)
    #maximum number of iteration
    term=10000
    #target accuracy
    acc=1e-13

    #start iteration
    for i in range(term):
        #sort
        fun_ind=np.arange(N+1)
        mergesort(fun,0,N,fun_ind)
        point=point[fun_ind]
        #check if find the minimum
        ran=abs((fun[0]-fun[-1])/(0.5*(fun[0]+fun[-1])))
        #ran=abs(np.mean(point[0]-point[-1]))
        if ran<acc:
            return point[0]
        #centroid of the first N points
        xcen=np.mean(point[:-1],axis=0)
        #propose a new point by reflecting
        xtry=2*xcen-point[-1]
        if f(xtry,other1,other2,other3)>=fun[0] and f(xtry,other1,other2,other3)<fun[-1]:
            #new points is better but not the best,accept it
            point[-1]=np.copy(xtry)
            fun[-1]=f(xtry,other1,other2,other3)
        elif f(xtry,other1,other2,other3)<fun[0]:
            #new point is the very best
            #propose a new point by expanding further
            xexp=2*xtry-xcen
            if f(xexp,other1,other2,other3)<f(xtry,other1,other2,other3):
                #xexp is better, accept it
                point[-1]=np.copy(xexp)
                fun[-1]=f(xexp,other1,other2,other3)
            else:
                #accept the reflected one
                point[-1]=np.copy(xtry)
                fun[-1]=f(xtry,other1,other2,other3)
        else:
            #xtry is the baddest
            #propose a new point by contracting
            xtry=0.5*(xcen+point[-1])
            if f(xtry,other1,other2,other3)<fun[-1]:
                #accept
                point[-1]=np.copy(xtry)
                fun[-1]=f(xtry,other1,other2,other3)
            else:
                #zoom on the best point by contracting all others to it
                point[1:]=0.5*(xcen+point[1:])
                for i in range(N):
                    fun[i+1]=f(point[i+1],other1,other2,other3)
    #the target accuracy is not reached after the maximum of iterations
    #return the current best one
    return point[0]

def n_integrand(x,args):
    a,b,c=args
    return np.power(x,a-1)*np.exp(-np.power((x/b),c))

def chi_square_easy(par,N_bin,Nsat,bins_edge):
    #calculate chi square
    #N_bin is observed value
    a,b,c=par
    #normalization factor, containing coefficients of n(x)
    norm_coef=simpson(n_integrand,xl=0,xu=5,N=10000,args=(a,b,c))
    N_binmean=np.zeros_like(N_bin,dtype='float64')
    for i in range(N_binmean.size):
        #expected value by model
        N_binmean[i]= Nsat*simpson(n_integrand,xl=bins_edge[i],xu=bins_edge[i+1],N=1000,args=(a,b,c))/norm_coef
    #take mean as variance
    chi_sq=np.sum(np.square(N_bin-N_binmean)/N_binmean)
    return chi_sq

def poisson_likelihood(par,N_bin,Nsat,bins_edge):
    #calculate -lnL
    a,b,c=par
    norm_coef=simpson(n_integrand,xl=0,xu=5,N=10000,args=(a,b,c))
    N_binmean=np.zeros_like(N_bin,dtype='float64')
    for i in range(N_binmean.size):
        N_binmean[i]=Nsat*simpson(n_integrand,xl=bins_edge[i],xu=bins_edge[i+1],N=1000,args=(a,b,c))/norm_coef
    lnL=-np.sum(N_bin*np.log(N_binmean)-N_binmean)
    return lnL

def G_test(y_ob,y_mod,k):
    #calculate G value and Q value
    #y_ob is an array containing observed values and y_mod is an array containing expected values
    #k is the number of degrees of freedom
    mask=(y_ob!=0)
    #y_ob=0 part contributes 0 to G
    G_value=2*np.sum(y_ob[mask]*np.log(y_ob[mask]/y_mod[mask]))
    #use the incomplete gamma functions from scipy
    P=gammainc(0.5*k,0.5*G_value)
    Q=1-P
    return G_value,Q

# loop all steps for the 5 files
num=[11,12,13,14,15]
for mnum in num:
    #read in data
    f1=open('satgals_m{:}.txt'.format(mnum))
    lines=f1.readlines()
    #total number of halos in a file
    halo=int(lines[3])
    f1.close()
    #read in all satellites in a file
    sat_num=np.loadtxt('satgals_m{:}.txt'.format(mnum),skiprows=4)
    #mean number of satellites in each halo
    Nsat=sat_num.shape[0]/halo
    #choose 30 radial bins in [0,5]. There are many satellites locates in a relative small range.
    #The number of bins should not be too large(many empty bins) or too small(cannot show the distribution).
    #We also used 20 bins for handin2. so I think 30 is a reasonable number.
    bin_num=30
    #bin in log space because this number density distribution is more regular in log space as showed in previous exercises
    bins=np.logspace(-4,np.log10(5), bin_num+1)
    hist,_=np.histogram(sat_num[:,0],bins=bins)
    bins_cen=0.5*(bins[:-1]+bins[1:])
    #hist2 is used to exlude 0 points when shows in loglog space
    hist2=(hist!=0)
    #mean number per halo in each bin
    N_binsat=hist/halo

    #(a) chi square easy fitting
    #initial guess for a, b and c
    a=3
    b=0.5
    c=3.1
    #use downhill simplex from handin2 to minimize chi square. p1_chi contains the best fit values
    p1_chi=downhill_simplex(chi_square_easy,init_x=np.array([a,b,c]),other1=N_binsat,other2=Nsat,other3=bins)
    chi_value=chi_square_easy(p1_chi,N_binsat,Nsat,bins)
    print('p1 of chi sqaure fitting:',p1_chi)
    print('chi square value:',chi_value)
    #write out result
    file1=open('chisquare{:}.txt'.format(mnum),'w')
    file1.write('File: satgals m{:2d}'.format(mnum))
    file1.write('\nBest fit a,b and c are :{0:.10f} {1:.10f} {2:.10f}'.format(p1_chi[0],p1_chi[1],p1_chi[2]) )
    file1.write('\nThe mean number of satellites in each halo Nsat is :{:.10f}'.format(Nsat))
    file1.write('\nThe minimum value of chi square is :{:.10f}'.format(chi_value))
    file1.close()
    #calculate the best fit profile
    #N=10000 (1000 for indivual intervals) for simpson method can make the sum of all expected values very close to the sum of all observed values
    norm_coef=simpson(n_integrand,xl=0,xu=5,N=10000,args=p1_chi)
    N_binmean_chi=np.zeros_like(N_binsat)
    for i in range(N_binsat.size):
        N_binmean_chi[i]=Nsat*simpson(n_integrand,xl=bins[i],xu=bins[i+1],N=1000,args=p1_chi)/norm_coef
    #plot binned data and the best fit profile together
    plt.loglog(bins_cen[hist2],N_binsat[hist2],'o',label='binned data')
    plt.loglog(bins_cen,N_binmean_chi,label='chi square fitting')
    plt.xlabel('x')
    plt.ylabel('Ni')
    plt.title('satgals m{:}'.format(mnum))
    plt.legend()
    plt.savefig('./plots/m{:}chifit.png'.format(mnum),dpi=150)
    plt.close()
    #(b) poisson
    #initial guess
    a=3
    b=0.5
    c=3.1
    #minimize -lnL
    p1_poi=downhill_simplex(poisson_likelihood,init_x=np.array([a,b,c]),other1=N_binsat,other2=Nsat,other3=bins)
    poi_value=poisson_likelihood(p1_poi,N_binsat,Nsat,bins)
    print('p1 of Poisson log likelihood fitting:',p1_poi)
    print('-lnL:',poi_value)
    #write out result
    file1=open('poisson{:2d}.txt'.format(mnum),'w')
    file1.write('File: satgals m{:2d}'.format(mnum))
    file1.write('\nBest fit a,b and c are :{0:.10f} {1:.10f} {2:.10f}'.format(p1_poi[0],p1_poi[1],p1_poi[2]) )
    file1.write('\nThe minimum value of -lnL(a,b,c) is :{:.10f}'.format(poi_value))
    file1.close()
    #calculate the best fit profile
    norm_coef=simpson(n_integrand,xl=0,xu=5,N=10000,args=p1_poi)
    N_binmean_poi=np.zeros_like(N_binsat)
    for i in range(N_binsat.size):
        N_binmean_poi[i]=Nsat*simpson(n_integrand,xl=bins[i],xu=bins[i+1],N=1000,args=p1_poi)/norm_coef
    #plot binned data and the best fit profile together
    plt.loglog(bins_cen[hist2],N_binsat[hist2],'o',label='binned data')
    plt.loglog(bins_cen,N_binmean_poi,label='Poisson fitting')
    plt.xlabel('x')
    plt.ylabel('Ni')
    plt.title('satgals m{:}'.format(mnum))
    plt.legend()
    plt.savefig('./plots/m{:}poifit.png'.format(mnum),dpi=150)
    plt.close()

    #(c)
    #the number of degrees of freedom equals to the number of bins minus the number of parameters.
    #Because the integral of n(x)=Nsat, we should also minus this condition from degrees of freedom.
    k_free=bin_num-4
    G_stat=np.zeros((2,2))
    G_stat[0,:]=G_test(N_binsat,N_binmean_chi,k_free)
    G_stat[1,:]=G_test(N_binsat,N_binmean_poi,k_free)
    #write out result
    file1=open('Gtest{:2d}.txt'.format(mnum),'w')
    file1.write('File: satgals m{:2d}'.format(mnum))
    file1.write('\nG value and Q value for Chi square fitting are {0:.10f}, {1:.10f}'.format(G_stat[0,0],G_stat[0,1]))
    file1.write('\nG value and Q value for Poisson likelihood fitting are {0:.10f}, {1:.10f}'.format(G_stat[1,0],G_stat[1,1]))
    file1.close()
