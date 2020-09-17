import numpy as np
import skimage 
from skimage import io
import matplotlib.pyplot as plt

# This function accepts integers m and n, and an array of singular values s and returns the Sigma matrix.

def sigma(m,n,s):
	L=np.zeros((m,n))
	for i in range(len(s)):
	L[i,i] = s[i]
	return L

# This function accepts arrays u,s, and v_t, and returns the corresponing array A.

def reconstructed_array(u,s,v_t):
	m=len(np.transpose(u))
	n=len(v_t)
	S=sigma(m,n,s)
	L = np.matmul(u, S)
	V = np.matmul(L,v_t)
	return V

# This function accepts an array A and an integer k, and returns a rank k approximation of A as computed by an SVD.

def lower_rank(A,k):
	u,s,v_t=np.linalg.svd(A)
	m=len(np.transpose(u))
	n=len(np.transpose(v_t))
	s_prime=s[0:k]
	Q=reconstructed_array(u,s_prime,v_t)
	return Q

RGB_array = io.imread('Lab13Image.png')            
gray_array=skimage.color.rgb2gray(RGB_array)

def show_color(array):
	plt.figure(figsize=(10,10))
	plt.grid(None)
	plt.imshow(array)
	return None

def show_gray(array):
	plt.figure(figsize=(10,10))
	plt.grid(None)
	plt.imshow(array,cmap='gray',vmin=0,vmax=1)
	return None


m=len(gray_array)
n=len(np.transpose(gray_array))
original_size=m*n

u,s,v_t=np.linalg.svd(gray_array)
plt.plot(s)
min_rank=100

u,s,v_t=np.linalg.svd(gray_array)
rank_100_size=len(u)*100+100+len(np.transpose(v_t))*100
relative_size=rank_100_size/original_size

