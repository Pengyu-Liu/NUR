#code for question2
import numpy as np
#2(a)
def LU_decom(M):
    #M is the Matrix to be decomposed and should have a size of N*N
    A=np.copy(M)
    n1,n2=A.shape
    if n1!=n2:
        print('Error: The size of the matrix should be N*N.')
        return
    #an array to store the permutation
    indexmax=np.zeros(n1,dtype=np.int)
    #Improved Crout's algorithm on slide14 in lecture 2
    for k in range(n1):
        # find the row with the largest pivot candidate from row>=k
        ind=np.argmax(abs(A[k:,k]))+k
        if ind!=k:
            #if ind!=k,ind must be large than k. row[ind] has the largest absolute value and swap row ind,k
            A[[ind,k],:]=A[[k,ind],:]
        #record the swap: if ind=k, no swap; if ind>k, ind is the swaped row for row k
        indexmax[k]=ind
        if A[k,k]==0:
            print('Error: The matrix is singular.')
            return
        for i in range(k+1,n1):
            #alpha(ik)
            A[i,k]=A[i,k]/A[k,k]
            #loop over columns j>k to compute alpha(ij) and beta(ij)
            A[i,(k+1):]-=A[i,k]*A[k,(k+1):]

    #return A(the LU matrix) and pivot index array
    return A, indexmax

def LU_solve(A,pivot,b):
    #solve x
    x=np.copy(b)
    n1=x.size
    #forward substitution
    for i in range(n1):
        #Because we swapped rows in LU decomposition, now also need to swap rows for x
        mid=x[pivot[i]]
        x[pivot[i]]=x[i]
        #alpha(ii)=1, so no need to divide
        if i==0:
            x[i]=mid
        else:
            x[i]=mid-np.sum(A[i,0:i]*x[0:i])

    #back substitution
    for i in range(n1-1,-1,-1):
    #In forward substitution, we swapped row. Don't need to swap in back substitution
        if i==(n1-1):
            x[i]=x[i]/A[i,i]
        else:
            x[i]=(x[i]-np.sum(A[i,(i+1):]*x[(i+1):]))/A[i,i]

    return x


#read in data
wgs=np.loadtxt('wgs.dat',dtype=np.float32)
wss=np.loadtxt("wss.dat",dtype=np.float32)
#LU decomposition
LU,piv=LU_decom(wss)
#solve
f=LU_solve(LU,piv,wgs)


np.savetxt('2aLU.txt',LU,fmt='%1.5f')
np.savetxt('2af.txt',f)
file1=open('2af.txt','a')
file1.write('\nThe sum of f is: ')
file1.write(str(sum(f)))
file1.write('\nThe error is: ')
file1.write(str(abs(1-sum(f))))
file1.close()


#2(b) single iterative
wgs_new=np.dot(wss,f)
#use the same LU and piv, no need to do LU deecomposition again
delt_f=LU_solve(LU,piv,wgs_new-wgs)
f_new=f-delt_f

np.savetxt('2bf.txt',f_new)
file2=open('2bf.txt','a')
file2.write('\nThe sum of the imroved f is: ')
file2.write(str(sum(f_new)))
file2.write('\nThe error is: ')
file2.write(str(abs(1-sum(f_new))))
file2.close()
