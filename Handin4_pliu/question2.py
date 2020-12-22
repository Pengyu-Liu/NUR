#question 2
import numpy as np
import matplotlib.pyplot as plt

#(a)
galaxy_data=np.loadtxt('galaxy_data.txt')
feature_matrix=np.copy(galaxy_data[:,:-1])
#feature scale
for i in range(4):
    mean=np.mean(feature_matrix[:,i])
    std=np.std(feature_matrix[:,i])
    feature_matrix[:,i]=(feature_matrix[:,i]-mean)/std
#all features for the first 5 objects
print(feature_matrix[:5])
np.savetxt('features_first5.txt',feature_matrix[:5],'%1.6f')

#(b)
def sigmoid(x):
    return 1/(1+np.exp(-x))

def cost_function(y,y_hat):
    #cost function of logistic regression
    return -np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))/y.size

def logistic_regression(w,sets):
    #w: weighs; sets: feature pairs
    y_hat=sigmoid(w[0]*feature_matrix[:,sets[0]]+w[1]*feature_matrix[:,sets[1]]+w[2])
    cost=cost_function(galaxy_data[:,-1],y_hat)
    return cost

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

        #np.array([[1,1,1],[0.9,1,1],[1,0.8,0.9],[1,0.9,0.8]]
#minization
def downhill_simplex(f,init_x,other):
    #downhill method to find the minimum of f
    #N dimension requires N+1 points
    #dimention
    N=init_x.size
    #generate N+1 points
    point=np.zeros((N+1,N))
    point[0]=np.copy(init_x)
    point[1]=point[0]+[-0.1,0.1,0]
    point[2]=point[0]+[0,-0.1,-0.1]
    point[3]=point[0]+[-0.2,-0.1,-0.2]
    fun=np.zeros(N+1)
    fun[0]=f(point[0],other)
    fun[1]=f(point[1],other)
    fun[2]=f(point[2],other)
    fun[3]=f(point[3],other)
    #maximum number of iteration
    term=10000
    #target accuracy
    acc=1e-10
    #record the cost function of each iterations
    trace=[]

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
            trace.append(fun[0])
            return point[0],trace
        #centroid of the first N points
        xcen=np.mean(point[:-1],axis=0)
        #propose a new point by reflecting
        xtry=2*xcen-point[-1]
        if f(xtry,other)>=fun[0] and f(xtry,other)<fun[-1]:
            #new points is better but not the best,accept it
            point[-1]=np.copy(xtry)
            fun[-1]=f(xtry,other)
            trace.append(fun[0])
        elif f(xtry,other)<fun[0]:
            #new point is the very best
            #propose a new point by expanding further
            xexp=2*xtry-xcen
            if f(xexp,other)<f(xtry,other):
                #xexp is better, accept it
                point[-1]=np.copy(xexp)
                fun[-1]=f(xexp,other)
                trace.append(fun[-1])
            else:
                #accept the reflected one
                point[-1]=np.copy(xtry)
                fun[-1]=f(xtry,other)
                trace.append(fun[-1])
        else:
            #xtry is the baddest
            trace.append(fun[0])
            #propose a new point by contracting
            xtry=0.5*(xcen+point[-1])
            if f(xtry,other)<fun[-1]:
                #accept
                point[-1]=np.copy(xtry)
                fun[-1]=f(xtry,other)
            else:
                #zoom on the best point by contracting all others to it
                point[1:]=0.5*(xcen+point[1:])
                for i in range(N):
                    fun[i+1]=f(point[i+1],other)
    #the target accuracy is not reached after the maximum of iterations
    #return the current best one and trace
    return point[0],trace

#initialization
w=np.array([1,1,1])
#column pairs
sets1=[0,1]
sets2=[1,3]
#minimize the cost function of logistic regression
result=np.zeros((2,4))
result[0,:3],cost_value1=downhill_simplex(logistic_regression,w,sets1)
result[1,:3],cost_value2=downhill_simplex(logistic_regression,w,sets2)
result[0,-1]=cost_value1[-1]
result[1,-1]=cost_value2[-1]
print(result)
np.savetxt('weight_2b.txt',result,'%1.6f')
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(cost_value1,'-o',label='[0,1]',ms=4)
plt.xlabel('interation')
plt.ylabel('cost function')
plt.legend()
plt.subplot(122)
plt.plot(cost_value2,'-o',label='[1,3]',ms=4)
plt.xlabel('interation')
plt.ylabel('cost function')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/cost_2b.png',dpi=200)
plt.close()
#(c)
real=np.copy(galaxy_data[:,-1]).astype('bool')
#columns are: TP, TN, FP, FN, F1
ratio=np.zeros((2,5))
for k,i in enumerate([sets1,sets2]):
    y_res=sigmoid(result[k,0]*feature_matrix[:,i[0]]+result[k,1]*feature_matrix[:,i[1]]+result[k,2])
    predict=(y_res>=0.5)
    TP=np.sum(predict&real)
    TN=np.sum((~predict)&(~real))
    FP=np.sum(predict)-TP
    FN=np.sum(~predict)-TN
    P=TP/(TP+FP)
    R=TP/(TP+FN)
    ratio[k,0]=TP
    ratio[k,1]=TN
    ratio[k,2]=FP
    ratio[k,3]=FN
    #F1 score
    ratio[k,4]=2*P*R/(P+R)
print(ratio)
np.savetxt('results_2b.txt',ratio,'%1.4f')
#plot
x0=np.arange(-2,2,0.001)
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.scatter(feature_matrix[real,sets1[0]],feature_matrix[real,sets1[1]],marker='^',s=4,label='1')
plt.scatter(feature_matrix[~real,sets1[0]],feature_matrix[~real,sets1[1]],marker='o',s=4,label='0')
plt.plot(x0,-(result[0,0]/result[0,1])*x0-(result[0,2]/result[0,1]),color='g',label='decision boundary')
plt.legend()
plt.xlabel('feature0')
plt.ylabel('feature1')
plt.ylim(-5,5)

plt.subplot(122)
plt.scatter(feature_matrix[real,sets2[0]],feature_matrix[real,sets2[1]],marker='^',s=4,label='1')
plt.scatter(feature_matrix[~real,sets2[0]],feature_matrix[~real,sets2[1]],marker='o',s=4,label='0')
plt.plot(x0,-(result[1,0]/result[1,1])*x0-(result[1,2]/result[1,1]),color='g',label='decision boundary')
plt.legend()
plt.xlabel('feature1')
plt.ylabel('feature3')
plt.ylim(-5,5)
plt.tight_layout()
plt.savefig('./plots/plot_2c.png',dpi=200)
plt.close()
