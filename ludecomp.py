import numpy as np 

def augment(A,b): 
	return np.column_stack((A,b))


def first_column_zeros(A):
	B=np.copy(A)
	for i in range(1, len(B)):
		x=B[i,0]/B[0,0]
		B[i,:]=B[i,:]-x*B[0,:]
	return B

def row_echelon(A,b): 
	B=augment(A,b)
	for i in range(0, np.size(B,0)):
		B[i:np.size(B,0),
		i:np.size(B,1)]=first_column_zeros(B[i:np.size(B,0), i:np.size(B,1)])
	return B


def LU_decomposition(A):
	U=np.copy(A)
	L=np.identity(len(A))
	m=len(A)
	n=len(A[0])
	for j in range(n-1):
		for i in range(j+1, m):
			L[i,j] = U[i,j]/U[j,j]
			U[i] = U[i] - np.dot(L[i,j],U[j])
	return L,U

def forward_substitution(L,b): # Accepts a lower triangular square matrix L and a vector b, solves Ly=b for y.
	n=len(b)
	y=np.zeros(n)
	for i in range(n):
		y[i]=(b[i]-np.dot(y,L[i,:])/L[i,i])
	return y

def back_substitution(U,y):# Accepts an upper triangular square matrix U and a vector b, solves Ux=b for x.
	n=len(y)
	X=np.zeros(n)
	for i in range(n-1,-1,-1):
		X[i]=((y[i]-np.dot(X,U[i,:]))/U[i,i])
	return X

def LU_solver(A,b): 
	X,Y=LU_decomposition(A)
	y=forward_substitution(X,b)
	x=back_substitution(Y,y)
	return x
