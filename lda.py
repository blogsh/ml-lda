from numpy import *
import matplotlib.pyplot as plt
import numpy.random as rand
import scipy.special as fun

# Define process parameters
alpha = 1
beta = [1, 1]

phi = 0.5
theta = [
	[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
	[0.0, 0.0, 0.0, 0.5, 0.0, 0.5]
]

# Generate data
n = 1000
Z = array([int(rand.random() < phi) for _ in range(n)])
W = array([where(rand.multinomial(1, theta[Z[i]]))[0][0] for i in range(n)])

# Start at random assignment
Z = array([int(rand.random() < 0.5) for _ in range(n)])

# Initialize counts
p = zeros((2))
nz = array([sum(Z == j) for j in range(2)])
nzk = fromfunction(
	vectorize(lambda j, k: sum(multiply((Z == j), (W == k)))), 
	(2, 6))

# Go!
while True:
	for i in range(n):
		# Remove current variable from counts
		nz[Z[i]] -= 1
		nzk[Z[i], W[i]] -= 1

		# Compute probabilities
		p = array([(nz[j] + alpha) * prod(nzk[j,:] + beta[j]) / (nz[j] + 6 * beta[j]) for j in range(2)])
		p /= sum(p)

		# Choose new assignment and update counts
		Z[i] = 1 - where(rand.multinomial(1, p))[0][0]
		nz[Z[i]] += 1
		nzk[Z[i], W[i]] += 1

	# Compute estimates
	phiEst = (nz[1] + alpha - 1) / (n + 2 * alpha - 2)
	thetaEst = [(nzk[j,:] + beta[j] - 1) / (nz[j] + 6 * beta[j] - 6) for j in range(2)] 

	# Compute log-likelihoods
	mask0 = thetaEst[0] > 0
	mask1 = thetaEst[1] > 0

	llW = sum(multiply(log(thetaEst[0][mask0]), nzk[0,mask0]))
	llW += sum(multiply(log(thetaEst[1][mask1]), nzk[1,mask1]))
	llZ = nz[1] * log(phiEst) +  nz[0] * log(1 - phiEst)

	print('phi', phiEst)
	print('thetaA', thetaEst[0])
	print('thetaB', thetaEst[1])
	print('log likelihood', llW + llZ)
	print()
