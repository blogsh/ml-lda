from numpy import *
import matplotlib.pyplot as plt
import numpy.random as rand
import scipy.special as fun
import scipy.optimize as opt

# Define process parameters
phi = 0.7
theta = [
	[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
	[0.0, 0.0, 0.0, 0.5, 0.0, 0.5]
]

# Generate data
n = 1000
Z = array([int(rand.random() < phi) for _ in range(n)])
W = array([where(rand.multinomial(1, theta[Z[i]]))[0][0] for i in range(n)])
Zref = Z[:]

# Parameter estimation
def estimate(Z):
	nz = array([sum(Z == j) for j in range(2)])
	nzk = fromfunction(
		vectorize(lambda j, k: sum(multiply((Z == j), (W == k)))), 
		(2, 6))

	thetaEst = [nzk[j,:] / nz[j] for j in range(2)]
	phiEst = nz[1] / n

	mask0 = thetaEst[0] > 0
	mask1 = thetaEst[1] > 0

	llW = sum(multiply(log(thetaEst[0][mask0]), nzk[0,mask0]))
	llW += sum(multiply(log(thetaEst[1][mask1]), nzk[1,mask1]))
	llZ = nz[1] * log(phiEst) +  nz[0] * log(1 - phiEst)

	return thetaEst, phiEst, llW, llZ

# Define objective (likelihood)
def objective(Z):
	thetaEst, phiEst, llW, llZ = estimate(Z)
	return llW + llZ

# Start at random assignment
Z = array([int(rand.random() < 0.5) for _ in range(n)])
obj = objective(Z)

optZ = Z[:]
optObj = obj

# Simulated Annealing
T = 1000
while T > 1e-10:
	T = 0.999 * T
	newZ = Z[:]
	i = rand.randint(n)
	newZ[i] = 1 - newZ[i]
	newObj = objective(newZ)

	if random.random() < 1 if newObj > obj else exp(-(obj - newObj) / T):
		Z = newZ[:]
		obj = newObj

	if obj > optObj:
		optObj = obj
		optZ = Z[:]
		
		thetaEst, phiEst, llW, llZ = estimate(Z)
		print('log likelihood sum:', llW + llZ)
		print('phi', phiEst)
		print('thetaA', thetaEst[0])
		print('thetaB', thetaEst[1])
		print()
