
import numpy as np

def get_eigenweights(A):

	# Check if all cols sum to 1
	if not np.allclose(np.sum(A, axis=0), 1):
		print("Rows do not sum to 1")
		return None

	# Find the leading eigenvector
	eigenvalues, eigenvectors = np.linalg.eig(A)

	leading_eigenvector_ind = np.argmax(eigenvalues)
	leading_eigenvector = eigenvectors[:,leading_eigenvector_ind]

	# Normalise the leading eigenvector
	normalised_eigenvector = leading_eigenvector/sum(leading_eigenvector)

	return np.real(normalised_eigenvector)

def normalise(A):
	
	A = np.array(A)
	return A / A.sum(axis=1, keepdims=True)
	
A = [
	[1,1,1,0],
	[1,0,1,0],
	[0,0,0,1],
	[0,0,1,1]
	]

A = [
	[0,1,0,0,0,1,1],
	[1,0,1,0,0,1,0],
	[0,1,0,1,0,1,0],
	[0,0,1,0,1,0,1],
	[0,0,0,1,0,0,1],
	[1,1,0,0,0,0,1],
	[1,0,1,1,1,1,0],
	]


A  = [
	[0,1,1,0,1,1],
	[1,0,0,0,0,1],
	[1,0,0,1,0,1],
	[0,0,1,0,1,0],
	[0,1,0,1,0,1],
	[1,1,1,0,1,0]]




A  = normalise(A)
print(A)

w = get_eigenweights(A.T)

print(w)

print((w[0] + w[1])*6/2)

xs = [1,2,5,6,7,8,9,10]

