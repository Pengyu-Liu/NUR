#question2
import numpy as np
import matplotlib.pyplot as plt

seed=31
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

#(a)
print('run (a)')
def bracketmin(f,x1,x2):
    a=x1
    b=x2
    if f(a)<f(b):
        #switch a and b to ensure that f(b)<f(a)
        a,b=b,a
    #propose a new point
    #golden ratio
    w=1.618
    c=b+(b-a)*w
    while f(c)<f(b):
        #use a,b,c to find a new point by fitting a parabola
        d=b-0.5*(np.power(b-a,2)*(f(b)-f(c))-np.power(b-c,2)*(f(b)-f(a)))/((b-a)*(f(b)-f(c))-(b-c)*(f(b)-f(a)))
        if ((d-c)*(d-b))<0:
            #d is between b and c
            if f(d)<f(c):
                #find the bracket
                return b,d,c
            elif f(d)>f(b):
                #find the bracket
                return a,b,d
            #the parabola is a bad fit
            d=c+(c-b)*w
        else:
            #d is beyond c
            if abs(b-d)>20*abs(c-b):
                #d is too far away. take another step
                d=c+(c-b)*w
        #move all points over
        a,b,c=b,c,d
#         if f(a)<f(b):
#             #always ensure f(b)<f(a)
#             a,b=b,a

    return a,b,c

def golden_search_min(f,bracket):
    #bracket is a tuple(a<c)
    a,b,c=bracket
    acc=1e-6
    w=0.38197
    #maximum of iteration
    term=int(1e6)
    for i in range(term):
        #identify the larger interval
        #choose d inside the interval in a self-similar way
        if abs(a-b)>abs(b-c):
            d=b+(a-b)*w
        else:
            d=b+(c-b)*w
        if abs(c-a)>acc:
            if f(d)<f(b):
                #tighten towards d
                if ((d-a)*(d-b))<0:
                    #d is in between a and b
                    c,b=b,d
                else:
                    #d is in between b and c
                    a,b=b,d
            else:
                if ((d-a)*(d-b))<0:
                    #d is in between a and b
                    a=d
                else:
                    #d is in between b and c
                    c=d
        else:
            #reached the accuracy
            if f(d)<f(b):
                return d
            else:
                return b

def Nsate(x):
    a=2.4
    b=0.25
    c=1.7
    Nsat=100
    A=256/(5*np.power(np.pi,1.5))
    return -4*np.pi*A*Nsat*np.power(x,a-1)*np.power(1/b,a-3)*np.exp(-np.power(x/b,c))


bracket=bracketmin(Nsate,0,5)
x_max=golden_search_min(Nsate,bracket)
N_max=-Nsate(x_max)
print('The braket is: ',bracket)
print('x at the maximum is:',x_max)
print('N(x) at the maximum is:', N_max)
#write out result
file1=open('Q2a.txt','w')
file1.write('The braket after bracketmin is:{:}'.format(bracket))
file1.write('\nx at the maximum is:{:.10f}'.format(x_max))
file1.write('\nN(x) at the maximum is:{:.10f}'.format(N_max))
file1.close()

#(b)
print('run (b)')
def N_norm(x):
    a=2.4
    b=0.25
    c=1.7
    A=256/(5*np.power(np.pi,1.5))
    return 4*np.pi*A*np.power(x,a-1)*np.power(1/b,a-3)*np.exp(-np.power(x/b,c))

def gx(x):
    #normalize N to 1
    N_max=2.7010960100989907
    return N_norm(x)/N_max
#rejection sample
def reject_sample(f,x_ran,num):
    #num is the number of points we want to generate
    point=np.zeros(num)
    #seed is the seed of we set for the entire program(global variable)
    ths=np.copy(seed)
    a,b=x_ran
    count=0
    #the number of random numbers we generate every time
    smal=10000
    while count<num:
        y1=RNG(seed=ths,low=0,up=1,n=smal)
        ths+=11
        x1=RNG(seed=ths,low=a,up=b,n=smal)
        ths+=3
        i=0
        while i in range(smal) and count<num:
            if y1[i]<=f(x1[i]):
                #accept it
                point[count]=x1[i]
                count+=1
            i+=1
    return point
#generate 10000 points
po_num=10000
point_sam=reject_sample(gx,x_ran=(0,5),num=po_num)
N_sam=np.zeros(po_num)
for i in range(po_num):
    N_sam[i]=-Nsate(point_sam[0])

#make a plot
arra=np.arange(0.0001,5,0.001)
#normalize N(x)
plt.plot(np.log10(arra),np.log10(-Nsate(arra)/100),label='N(x)')
#log bin
bins=np.logspace(np.log10(0.0001),np.log10(5), 21)
hist,bin_edges=np.histogram(point_sam,bins=bins)
hist2=np.copy(hist!=0)
bins_cen=0.5*(bins[:-1]+bins[1:])
width=bin_edges[1:]-bin_edges[:-1]
plt.plot(np.log10(bins_cen[hist2]),np.log10(hist[hist2]/width[hist2])-4,'-o',label='sampled')
plt.legend()
plt.xlabel('log10(x)')
plt.ylabel('log10(N(x))')
plt.savefig('./plots/Q2b.png',dpi=150)
plt.close()
#(c)
print('run (c)')
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

#select 100 random satellite galaxies
sele_num=100
drawn=np.zeros(sele_num)
record=np.ones(10000,dtype='bool')
ths=seed
#refer to Fisher Yates shuffle
for i in range(100):
    #generate a random index from 0 and the number of remaining index list
    d1=np.round(RNG(seed=ths,low=0,up=10000-1-i,n=1))
    ths+=1
    temp=0
    #count the d1th number from the remaining index list
    for j in range(10000):
        if temp==d1:
            break
        else:
            if record[i]==True:
                temp+=1
            else:
                continue
    #drawn this d1th number out and inacitive this index
    drawn[i]=np.copy(point_sam[temp])
    record[temp]=False

#sort the 100 drawn galaxies from smallest to largest radius
drawn_ind=np.arange(0,sele_num)
mergesort(drawn,0,sele_num-1,drawn_ind)

r=np.linspace(drawn[0],drawn[-1],100)
num_r=np.zeros_like(r)
for i in range(r.size):
    num_r[i]=np.sum(drawn<=r[i])
#plot the number of galaxies within r
plt.plot(r,num_r)
plt.xscale("log")
plt.xlim(1e-4,5)
plt.xlabel('r')
plt.ylabel('number of galaxies within r')
plt.savefig('./plots/Q2c.png',dpi=150)
plt.close()

#(d)
print('run (d)')
#find the bin from b containing the largest number of galaxies
histnew=np.copy(hist)
hist_ind=np.arange(histnew.size)
mergesort(histnew,0,histnew.size-1,hist_ind)
in1=hist_ind[-1]
#sorting
mask=(point_sam>=bin_edges[in1])*(point_sam<bin_edges[in1+1])
radial_bin=point_sam[mask]
radial_bin2=np.copy(radial_bin)
radial_size=radial_bin2.size
radial_ind=np.arange(0,radial_size)
mergesort(radial_bin2,0,radial_size-1,radial_ind)
#median of this bin
if radial_size%2==0:
    #radial_size is even
    radial_median=0.5*(radial_bin2[int(radial_size/2)]+radial_bin2[int(radial_size/2)-1])
else:
    #odd
    radial_median=radial_bin2[int(radial_size/2)]
#16th percentile
radial_16=radial_bin2[round(0.16*radial_size)]
#84th percentile
radial_84=radial_bin2[round(0.84*radial_size)]

print('median is:',radial_median)
print('16% is:',radial_16)
print('84% is:',radial_84)

filed=open('Q2d.txt','w')
filed.write('The median of this radial bin is:{:.10f}'.format(radial_median))
filed.write('\nThe 16th percentile of this radial bin is:{:.10f}'.format(radial_16))
filed.write('\nThe 84th percentile of this radial bin is:{:.10f}'.format(radial_84))
filed.close()
halo=np.zeros(100)
for i in range(100):
    halo[i]=np.sum(mask[100*i:100*(i+1)])

halo_mean=np.mean(halo)
_,_,_=plt.hist(halo,bins=100,label='data')

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

poi_k=np.arange(0,100)
poi_dis=np.zeros_like(poi_k,dtype='float64')
for i in range(poi_k.size):
    poi_dis[i]=poisson(halo_mean,poi_k[i])

plt.plot(poi_k,100*poi_dis,label='poisson')
plt.xlabel('The number of galaxies')
plt.ylabel('frequency')
plt.legend()
plt.savefig('./plots/Q2d.png',dpi=150)
plt.close()
