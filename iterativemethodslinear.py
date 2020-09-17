import numpy as np 
import matplotlib.pyplot as plt 
# Performs one iteration of the Jacobi method for system (1) applied to the point (x,y).

def jacobi1_iteration(x,y): 
	new_x = 1/7*(6+y)
	new_y = 1/5*(x+4)
	return [new_x, new_y]

# Performs n iterations of the Jacobi method on system (1) with starting estimate (0,0).

def jacobi1_method(n):
	x_n=0
	y_n=0
	for i in range(n):
		[x_n, y_n] = jacobi1_iteration(x_n, y_n)
	return [x_n, y_n]

# Performs one iteration of the Gauss-Seidel method for system (1) applied to the point (x,y).

def gs1_iteration(x,y): 
	new_x = 1/7*(6+y)
	new_y = 1/5*(new_x + 4)
	return [new_x, new_y]

# Performs n iterations of the Gauss-Seidel method on system (1) with starting estimate (0,0).

def gs1_method(n): 
	x_n=0
	y_n=0
	for i in range(n):
		[x_n, y_n] = gs1_iteration(x_n, y_n)
	return [x_n, y_n]

# Finds the error of the nth approximation of the solution to system (1) using the Gauss-Seidel method.

def gs1_error(n): 
	true_value = [1,1]
	gs_error = np.linalg.norm(np.array(true_value)-np.array(gs1_method(n)))
	return gs_error

# This command uses the function gs1_error to create a new function vect_gs1_error which will accept NumPy arrays of various sizes as input, instead of just a single number.

vect_gs1_error=np.vectorize(gs1_error)  



# This creates a NumPy array of values of the form [0,1,2,...,48,49], similar to the np.linspace command.  The 1 in the function tells NumPy to count up by ones.

n_vals=np.arange(0,50,1)



# This creates the plot, and labels the axes.

plt.title('Error of the Gauss-Seidel Method Applied to System 1')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.plot(n_vals,vect_gs1_error(n_vals),'ro')
plt.show()

# Gives one iteration of the Gauss-Seidel method for system (4) applied to the point (x,y).

def gs2_iteration(x,y): 
	new_x = (1+y)
	new_y = (5+ -2*new_x)
	return [new_x, new_y]

# Performs n iterations of the Gauss-Seidel method on system (4) with starting estimate (0,0).

def gs2_method(n): 
	x_n=0
	y_n=0
	for i in range(n):
		[x_n, y_n] = gs2_iteration(x_n, y_n)
	return [x_n, y_n]

# Finds the error of the nth approximation of the solution to system (4) using the Gauss-Seidel method.

def gs2_error(n): 
	true_value = [2,1]
	gs_error = np.linalg.norm(np.array(true_value)-np.array(gs2_method(n)))
	return gs_error

vect_gs2_error=np.vectorize(gs2_error)  

n_vals=np.arange(0,50,1)

plt.title('Error of the Gauss-Seidel Method Applied to System 4')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.plot(n_vals,vect_gs2_error(n_vals),'ro')
plt.show()

# Gives one iteration of the Gauss-Seidel method for the final system, applied to the point (x,y,z).

def gs3_iteration(x,y,z): 
	new_x = (2/5*y + -3/5*z + -8/5)
	new_y = (-1/4*new_x + z + 102/4)
	new_z = (1/2*new_x + 1/2*new_y -90/4) 
	return [new_x, new_y, new_z]

# Performs n iterations of the Gauss-Seidel method on the final system with starting estimate (0,0,0).

def gs3_method(n): 
	x_n=0
	y_n=0
	z_n=0
	for i in range(n):
		[x_n, y_n, z_n] = gs3_iteration(x_n, y_n, z_n)
	return [x_n, y_n, z_n]

