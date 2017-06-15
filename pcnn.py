

import numpy as np
import scipy.ndimage.filters as fi
from scipy.signal import fftconvolve as conv

'''
Algorithm for the Intersecting Cortical Model (ICM)
----------------------------------------------------

Symbols:

S is the input image
F is the internal neuron state
Y is the neuron outputs
T is the state of the dynamic thresholds
g and f are scalers, such that 0 < g < f < 1
h is a large scaler used to increase the dynamic threshold after firing
W{} describes the connections between the neurons

Equations:

1) F_ij[n+1] = f*F_ij[n] + S_ij + W{Y}_ij
2) Y_ij[n+1] = 1 if F_ij[n+1] > T_ij[n] else 0
3) T_ij[n+1] = g*T_ij[n] + h*Y_ij[n+1]

Curvature flow model of W:

4) W{A} = A' = [[F_2a'{M{A'}} + F_1a'{A'}] < 0.5]
5) A' = A + [F_1a{M{A}} > 0.5]
6) [F_1a{X}]_ij = X_ij if A_ij == 0 else 0
7) [F_2a{X}]_ij = X_ij if A_ij == 1 else 0
8) [X > d]_ij = 1 if X_ij >= d else 0
9) [X < d]_ij = 1 if X_ij <= d else 0
'''


def gaussian_kernel(size=5, sigma=1.):
	if size % 2 == 0:
		size += 1
	k = np.zeros((size,size))
	k[size/2, size/2] = 1
	k = fi.gaussian_filter(k, sigma)
	return k


class ICM(object):

	def __init__(self, w, h):
		'''
		'''
		self.f = 0.6
		self.g = 0.4
		self.h = 100.

		size = (h,w)
		self.F = np.zeros(size)
		self.Y = np.zeros(size)
		self.T = np.ones(size) * self.h * 5
		self.W = gaussian_kernel()



	def step(self, S):
		'''
		'''
		F = S + self.f*self.F + conv(self.Y, self.W, mode='same')
		Y = np.where(F > self.T, 1, 0)
		T = self.g*self.T + self.h*Y

		self.F = F
		self.Y = Y
		self.T = T


