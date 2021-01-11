'''
This module contains implementations of various pulse coupled neural network
models.

References
----------
TODO
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.ndimage.filters as fi
import skimage.filters as skf
from scipy.signal import fftconvolve as conv
from scipy.misc import imresize


def gaussian_kernel(size=5, sigma=1.):
    if size % 2 == 0:
        size += 1
    k = np.zeros((size,size))
    k[size/2, size/2] = 1
    k = fi.gaussian_filter(k, sigma)
    return k


def preprocess(img, size=None):
    '''Preprocess the input image. Resize and convert to greyscale.
    '''
        if size is not None:
            img = imresize(img, size, 'bicubic')
    img = img @ [0.3, 0.6, 0.1]
    return img


class PulseNet(object):
    '''A simple pulsing net where each each neuron pulses independently of all
    others. When a nueron fires, the threshold gets reset to a default value 
    and then decays over time. In this model, the neurons connected to higher-
    intensity inputs will fire more frequently. It's not a particularly useful
    model, but serves as a good starting point.
    '''

    def __init__(self, w, h):
        '''
        '''
        size = (w,h)
        self.F = np.zeros(size)
        self.Y = np.zeros(size)
        self.T = np.zeros(size)

        self.v = 500.
        self.t = 0.7


    def step(self, S):
        '''
        '''
        F = S
        Y = np.where(F > self.T, 1, 0)
        T = np.where(Y==1, self.v, self.t*self.T)

        self.F = F
        self.Y = Y
        self.T = T

        return self.Y



class ICM(PulseNet):
    '''Intersecting Cortical Model.
    This is a simplified version of Eckhorn's original biologically derived
    model, created specifically to be used in image processing tasks. It is
    based on the following equations:

    1) F_ij[n+1] = f*F_ij[n] + S_ij + W{Y}_ij
    2) Y_ij[n+1] = 1 if F_ij[n+1] > T_ij[n] else 0
    3) T_ij[n+1] = g*T_ij[n] + h*Y_ij[n+1]

    S is the input image
    F is the internal neuron state
    Y is the neuron outputs
    T is the state of the dynamic thresholds
    0 < g < f < 1, scalers
    h is a large scaler used to increase the dynamic threshold after firing
    W{} describes the connections between the neurons

    Curvature flow model of W:
    4) W{A} = A' = [[F_2a'{M{A'}} + F_1a'{A'}] < 0.5]
    5) A' = A + [F_1a{M{A}} > 0.5]
    6) [F_1a{X}]_ij = X_ij if A_ij == 0 else 0
    7) [F_2a{X}]_ij = X_ij if A_ij == 1 else 0
    8) [X > d]_ij = 1 if X_ij >= d else 0
    9) [X < d]_ij = 1 if X_ij <= d else 0
    '''
    
    def __init__(self, w, h, update='autowave'):
        '''
        '''
        update_methods = {
            'autowave': self._centripetal_autowave_update,
            'smooth': self._smooth_kernel_update
        }
        if update not in update_methods:
            raise Exception('{} is not a valid update method ({})'.format(
                update, ','.join(update_methods)))

        self.update = update_methods[update]

        self.f = 0.5
        self.g = 0.45
        self.h = 150.

        size = (h,w)
        self.F = np.zeros(size)
        self.Y = np.zeros(size)
        self.T = np.ones(size) * self.h * 5
        self.W = gaussian_kernel()


    def step(self, S):
        '''
        '''
        F = S + self.f*self.F + self.update()
        Y = np.where(F > self.T, 1, 0)
        T = self.g*self.T + self.h*Y

        self.F = F
        self.Y = Y
        self.T = T

        return self.Y


    def _smooth_kernel_update(self):
        return conv(self.Y, self.W, mode='same')


    def _centripetal_autowave_update(self):
        '''Curvature flow model of W:
        4) W{A} = A' = [[F_2a'{M{A'}} + F_1a'{A'}] < 0.5]
        5) A' = A + [F_1a{M{A}} > 0.5]
        6) [F_1a{X}]_ij = X_ij if A_ij == 0 else 0
        7) [F_2a{X}]_ij = X_ij if A_ij == 1 else 0
        8) [X > d]_ij = 1 if X_ij >= d else 0
        9) [X < d]_ij = 1 if X_ij <= d else 0
        '''
        M_Y = conv(self.Y, self.W, mode='same')
        Y_p = self.Y + (np.where(self.Y==0, M_Y, 0) >= 0.5)
        M_Yp = conv(Y_p, self.W, mode='same')
        W = Y_p + ((np.where(Y_p==1, M_Yp, 0) + np.where(Y_p==0, Y_p, 0)) <= 0.5)
        return W


class SCM(PulseNet):
    '''
    '''
    def __init__(self):
        pass

    def reset(self, img):
        self.U = np.zeros_like(img)
        self.Y = np.zeros_like(img)
        self.E = np.zeros_like(img)
        self.W = np.array([[0.5,1,0.5],[1,0,1],[0.5,1,0.5]])

        # default parameters chosen according to 
        # Chen, Park, Ma, and Ala (2011)
        sp = skf.threshold_otsu(img)
        self.f = np.log(1 / img.std())
        self.vl = 1
        self.b = ((img.max() / sp) - 1) / (6*self.vl)
        exf = np.exp(-self.f)
        m3 = (1 - exf**3) / (1 - exf)) + 6*self.b*self.vl*exf
        self.ve = exf + 1 + 6*self.b*self.vl
        self.e = np.log(self.ve / (sp*m3))

    def step(self, S):
        U1 = np.exp(-self.f) * self.U
        U2 = S + S * self.b * self.vl * conv(self.Y, self.W, mode='same')
        self.U = U1 + U2
        self.Y = self.U > self.E
        self.E = np.exp(-self.e)*self.E + self.ve*self.Y
        return self.Y


class Simulator(object):
    '''Convenience class for simulating models and viewing the results.
    '''

    def __init__(self):
        pass


    def simulate(self, input_img, model, steps=8):
        '''
        '''
        self.img = input_img
        self.results = []

        for _ in xrange(steps):
            res = model.step(input_img)
            self.results.append(res)


    def plot_results(self):
        '''
        '''
        plt.subplot2grid((2,6), (0,0), rowspan=2, colspan=2)
        plt.imshow(self.img, cmap='gray')
        plt.xticks([])
        plt.yticks([])

        for n,i in enumerate(self.results):
            plt.subplot2grid((2,6), (n/4, n%4 + 2))
            plt.imshow(i)
            plt.xticks([])
            plt.yticks([])

        plt.show()

