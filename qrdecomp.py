import numpy as np 

#This function reads in two vectors and outputs a boolean variable describing whether the two vectors are orthogonal.

def orthogonal_check(a,b):
	cutoff = 1e-12
	twodot = np.dot(a, b)
	checkvar = False
	if np.abs(twodot) < cutoff:
	checkvar = True
	else:
	checkvar = False
	return checkvar

# testing function
a=np.array([1,2,-1,4])
b=np.array([2,-2,-6,-1])

orthogonal_check(a,b)

# This function reads in a list of vectors and checks whether they form an orthogonal set. Orthcheck should be a Boolean value (True or False).

def orth_set_check(vect_set):
	for i in range(len(vect_set)):
		for j in range(i+1, len(vect_set)):
			if orthogonal_check(vect_set[i], vect_set[j]) != True:
				return False
	return True

# testing the function
p=[np.array([1,-1,0]),np.array([1,1,1]),np.array([0,1,-1])]
q=[np.array([1,0,0]),np.array([0,2,0]),np.array([0,0,3])]

orth_set_check(q)

# This function accepts a vector and returns a unit vector in the same direction.

def normalize(v):
	w = np.sqrt(np.dot(v,v))
	return v/w

# testing the function
normalize(np.array([1,1,1,1]))

# This function accepts two vectors a and b and returns the projection of a onto b.

def proj(a,b):
	dotvalue = ((np.dot(a, b))/(np.dot(b,b)))
	return dotvalue*b

# testing the function
a=np.array([1,4,1])
b=np.array([1,1,1])
proj(a,b)


# This function accepts a list of linearly independent vectors V, and produces a new list X of orthonormal vectors which span the same space as a the vector of V.

def gram_schmidt(V):
	X=V.copy()
	n=len(V)
	for i in range(1,n):
		for j in range(i):
  			X[i] = X[i] - proj(V[i], X[j])
	for i in range(n):
		X[i] = (normalize(X[i]))
	return X

# testing the function
L=[np.array([1,3,2,4,0]),np.array([-1,0,4,5,0]),np.array([0,2,2,2,2]),np.array([3,2,3,2,0])]
A=gram_schmidt(L)
print(orth_set_check(A))


# This function accepts a matrix A as a 2D NumPy Array, and returns two new matrices Q and R.

def QR_decomposition(A):
	X=np.copy(A)
	b=[X[:,i] for i in range(len(X[0,:]))]
	Y=np.copy(A)
	Q=gram_schmidt(b)
	for i in range(len(Q)):
		Y[:,i] = Q[i]
	R=np.matmul(np.transpose(Y), X)
	return Y,R

# testing the function
A=np.transpose(np.array([[1,3,2,4,0],[-1,0,4,5,0],[0,2,2,2,2],[3,2,3,2,0]]))
np.shape(A)[1]
Q,R=QR_decomposition(A)

print(Q)
print(R)
np.round(np.matmul(Q,R))

# This Function accepts an upper triangular square matrix U and a vector b, solves Ux=b for x. 

def back_substitution(U,b): 
	n=len(b)
	x=np.array([0.0 for i in range(n)])
	for i in range(n-1,-1, -1):
		r=(b[i]-sum([x[j]*U[i][j] for j in range(i+1,n)]))/U[i][i]
		x[i]=r
	return x

# This function accepts an invertible matrix A and a vector b, solves Ax=b for x.

def QR_solver(A,b):
	Q,R=QR_decomposition(A)
	X=np.matmul(np.transpose(Q), b)
	return back_substitution(R,X)

A=np.array([[3,1,-2],[1.5,2,-5],[2,-4,1]])
b=np.array([1.1,3,-2])
QR_solver(A,b) 
