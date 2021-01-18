#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Nonlinear dynamics of cantilevered pipe conveying fluid

AE255 Aeroelasticity term paper 

@author: G R Krishna Chand Avatar

"""
###################### LOADING PACKAGES

import os                        # for calling system functions in Ubuntu
import numpy as np               # for Numerical Python library
import matplotlib.pyplot as plt  # for plotting 
from scipy import integrate      # for numerical integration 
import scipy.linalg as la        # for linear algebra operations
import mpmath                    # for arbitrary-precision math library
import control.matlab            # for state-space modelling and time response (can be installed using: pip install control)

''' Font size control abd global plot parameters '''
SMALL_SIZE = 13
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', family='serif')
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#plt.rc('font',**{'family':'serif','serif':['Helvetica']})
#plt.rcParams["font.family"] = "cursive"
#plt.rc('font',**{'family':'serif','serif':['Times']})
#plt.rc('text', usetex=True)

line_width = 2
line_style = ['--', '-.', '-']
marker = ['*', 'd', 'o', '+']
legend = ["Coarse", "Medium", "Fine"] 
colour = ["red", "blue", "green", "magenta"]

'''
Class to perform integration for various coefficients 
'''
class Integrator:
    
    # Init method or constructor   
    def __init__(self, name):  
        self.name = name  
            
    # Numerically integrate 
    def printAgain():
        print("I also work")
        
    # print something
    def printMe():
        print("This class works")
        Integrator.printAgain()
        return None
    
    # integrate a function
    def integrateFunc(f1, i = 0,  lowerLim = 0, upperLim = 1):
        
        # integrate
        val, err = integrate.quadrature(f1, lowerLim, upperLim, args = i, tol=1e-9, rtol=1e-10, maxiter=100) # , limit=100, epsabs=1e-10)
        return val
        
    # numerically integrate product of two functions
    def integrateTwoFunc(f1, f2, i, j, lowerLim = 0, upperLim = 1):
                
        # wrapper function: return product of two functions
        def twoFunc(x, i, j):
            #return ModeShape.phi(x, i)*ModeShape.dPhi(x, j)
            return f1(x, i) * f2(x, j)
        
        # integrate
        val, err = integrate.quadrature(twoFunc, lowerLim, upperLim, args=(i,j), tol=1e-9, rtol=1e-10, maxiter=150)
        return val
    
    # numerically integrate product of three functions
    def integrateThreeFunc(f1, f2, i, j, lowerLim = 0, upperLim = 1):
             
        # lambda function
        identity = lambda x: x
        
        # wrapper function: return product of three functions
        def threeFunc(x, i, j):
            return identity(x)*f1(x, i) * f2(x, j)
        
        # integrate
        val, err = integrate.quadrature(threeFunc, lowerLim, upperLim, args=(i,j), tol=1e-9, rtol=1e-10, maxiter=100)
        return val
    
    # numerically integrate a number array  using 
    # trapezoidal, simpson or rhomberg method
    def numericalIntegration(y, x, scheme = 'simps'):
        
        if (scheme == 'trapz'):
            return integrate.trapz(y, x)
        
        if (scheme == 'simps'):
            return integrate.simps(y, x)
        
        if (scheme == 'rhom'):
            return integrate.rhom(t, dx = x[1] - x[0])
           
        
'''
Class to determine  coefficients
 
    * Stationary damping and stiffness matrices: C_ij, K_ij
    * Analytically determined: b_ij, c_ij, d_ij
    * Numerically determined: alpha_ijkl, beta_ijkl, gamma_ijkl
    
'''
numIntegrationPoints = 2**8 + 1 # number of integration points

class Coefficients:
    
    global modalFreq
    global normalizationFactor
    global numIntegrationPoints # number of integration points
    
    # C_{ij} = \alpha \lambda_j^4 \delta_{ij} + 2\sqrt{\beta} u_0 b_{ij}
    def C(i, j):
        global alpha_        # Kelvin-Voigt viscoelasticity factor
        global beta_         # Mass ratio
        global sigma_        # Non-dimensionalized external dissipation constant
        global U             # Non-dimensionalized flow velocity
        
        # wrapper function
        def kronecker(i, j):
            if (i == j):
                return 1
            else:
                return 0
        return (sigma_ + alpha_ * modalFreq[j]**4)*kronecker(i, j) + 2.0*np.sqrt(beta_)*U*Coefficients.b(i, j)
    
    # K_{ij} = \lambda_j^4 \delta_{ij} + (U^2 - \gamma) c_{ij} + \gamma (d_{ij} + b_{ij})
    def K(i, j):
        global U          # Non-dimensionalized flow velocity
        global gamma_     # Non-dimensionalized acceleration due to gravity
        
        # wrapper function
        def kronecker(i, j):
            if (i == j):
                return 1
            else:
                return 0
        return modalFreq[j]**4 * kronecker(i, j) + (U**2 - gamma_)*Coefficients.c(i, j) + gamma_*(Coefficients.d(i, j) + Coefficients.b(i, j))
    
    # b_{ij} = \int_0^1 \phi_i \phi'_j d\xi
    def b(i, j):
        #return Integrator.integrateTwoFunc(ModeShape.phi, ModeShape.dPhi, i, j)
        # Exact expressions
        if(i!=j):
            return 4.0/((modalFreq[i]/modalFreq[j])**2 + (-1)**(i+j))
        else:
            return 2.0
    
    # c_{ij} = \int_0^1 \phi_i \phi''_j d\xi
    def c(i, j):
        #return Integrator.integrateTwoFunc(ModeShape.phi, ModeShape.d2Phi, i, j)
    
        # Exact expressions
        sigma = lambda x: (np.sinh(x) - np.sin(x))/(np.cosh(x) + np.cos(x))
        
        if(i!=j):
            return 4.0*(modalFreq[j]*sigma(modalFreq[j]) - modalFreq[i]*sigma(modalFreq[i]))/((-1.0)**(i+j) - (modalFreq[i]/modalFreq[j])**2)
        else:
            return modalFreq[j]*sigma(modalFreq[j])*(2.0 - modalFreq[i]*sigma(modalFreq[i]))
  
    # d_{ij} = \int_0^1 \xi \phi_i \phi''_j d\xi
    def d(i, j):
        #return Integrator.integrateThreeFunc(ModeShape.phi, ModeShape.d2Phi, i, j)    
        
        # Exact expressions
        sigma = lambda x: (np.sinh(x) - np.sin(x))/(np.cosh(x) + np.cos(x))
        
        if(i!=j):
            val = 4.0*(modalFreq[j]*sigma(modalFreq[j]) - modalFreq[i]*sigma(modalFreq[i]) + 2.0)/(1.0 - (modalFreq[i]/modalFreq[j])**4)*(-1)**(i+j)
            val -= (3.0 + (modalFreq[i]/modalFreq[j])**4)/(1.0 - (modalFreq[i]/modalFreq[j])**4)*Coefficients.b(i,j) 
            return val
        else:
            return modalFreq[j]*sigma(modalFreq[j])*(2.0 - modalFreq[i]*sigma(modalFreq[i]))/2.0 # c_{ii}/2
    
    ## \alpha_{ijkl}
    def alpha(i, j, k, l):
        
        global U         # Non-dimensionalized flow velocity
        global gamma_    # Non-dimensionalized acceleration due to gravity
        
        xi = np.linspace(0, 1, numIntegrationPoints)
        
        firstTerm = phi(xi, i)*(ModeShape.d4Phi(xi, j)*ModeShape.dPhi(xi, k)*ModeShape.dPhi(xi, l) + \
                                4.0*ModeShape.dPhi(xi, j)*ModeShape.d2Phi(xi, k)*ModeShape.d3Phi(xi, l) + \
                                ModeShape.d2Phi(xi, j)*ModeShape.d2Phi(xi, k)*ModeShape.d2Phi(xi, l) )
        firstTermVal = Integrator.numericalIntegration(firstTerm, xi, scheme = 'rhom')
        
        secondTerm = np.zeros(xi.size) # TO UPDATE
        for n in range(xi.size):
            temp = Integrator.numericalIntegration(ModeShape.dPhi(xi[n:], k)*ModeShape.d2Phi(xi[n:], l), xi[n:])
            secondTerm[n] = ModeShape.phi(xi[n], i)*ModeShape.d2Phi(xi[n], j)*(ModeShape.dPhi(xi[n], k)*ModeShape.dPhi(xi[n], l) - temp)
        secondTermVal = U**2 * Integrator.numericalIntegration(secondTerm, xi, scheme = 'rhom')
        
        thirdTerm = ModeShape.phi(xi, i)*(0.5*ModeShape.dPhi(xi, j)*ModeShape.dPhi(xi, k)*ModeShape.dPhi(xi, l) - \
                                          1.5*(1 - xi)*ModeShape.d2Phi(xi, j)*ModeShape.dPhi(xi, k)*ModeShape.dPhi(xi, l))
        thirdTermVal = gamma * Integrator.numericalIntegration(thirdTerm, xi, scheme = 'rhom')
        
        return (firstTermVal + secondTermVal + thirdTermVal)
        
    ## \beta_{ijkl}
    def beta(i, j, k, l):
        global U         # Non-dimensionalized flow velocity
        global beta_     # Mass ratio
        
        xi = np.linspace(0, 1, numIntegrationPoints)
        term = np.zeros(xi.size)    # complete discrete term array
        
        for n in range(numIntegrationPoints):
            temp = Integrator.numericalIntegration(ModeShape.dPhi(xi[n:], k)*ModeShape.dPhi(xi[n:], l), xi[n:])
            term[n] = ModeShape.phi(xi[n], i)*(ModeShape.dPhi(xi[n], j)*ModeShape.dPhi(xi[n], k)*ModeShape.dPhi(xi[n], l) - \
                                               ModeShape.d2Phi(xi[n], j)*temp)
        
        return (2.0*U*np.sqrt(beta_)*Integrator.numericalIntegration(term, xi, scheme = 'rhom'))
    
    ## \gamma_{ijkl}
    def gamma(i, j, k, l):
        
        xi = np.linspace(0, 1, numIntegrationPoints)
        term = np.zeros(xi.size)  # complete discrete term array
        # tempSecondTerm = np.zeros(xi.size) # partial second term
        
        for n in range(numIntegrationPoints):
            tempFirstTerm = Integrator.numericalIntegration(ModeShape.dPhi(xi[:n+1], k)*ModeShape.dPhi(xi[:n+1], l), xi[:n+1])
            temp2 = np.zeros(xi.size - n) # for (\xi_n -> 1) integration in second term
            for m in range(n, xi.size):
                temp2[m - n] = Integrator.numericalIntegration(ModeShape.dPhi(xi[:m+1], k)*ModeShape.dPhi(xi[:m+1], l), xi[:m+1])
            
            tempSecondTerm = Integrator.numericalIntegration(temp2, xi[n:])
                
            term[n] = ModeShape.phi(xi[n], i)*ModeShape.dPhi(xi[n], j)*tempFirstTerm - \
                ModeShape.phi(xi[n], i)*ModeShape.d2Phi(xi[n], j)*tempSecondTerm
            
        return Integrator.numericalIntegration(term, xi, scheme = 'rhom')


    #### Coefficients of the modal equation for cantilever beam: \ddot{q}_s = \Lambda_s q_s  (\lambda_s is s^th modal frequency)        
    
'''
Class for mode shapes and their derivatives

'''

class ModeShape:
    
    global modalFreq
    global normalizationFactor
    
    # \phi (x) to be normalized
    def normalizePhi(x, modeNo):
        sigma = (np.sinh(modalFreq[modeNo]) - np.sin(modalFreq[modeNo]))/(np.cosh(modalFreq[modeNo]) + np.cos(modalFreq[modeNo]))
        val = np.cosh(modalFreq[modeNo]*x) - np.cos(modalFreq[modeNo]*x) - sigma*(np.sinh(modalFreq[modeNo]*x) - np.sin(modalFreq[modeNo]*x))
        return val
           
    # \phi (x)
    def phi(x, modeNo):
        sigma = (np.sinh(modalFreq[modeNo]) - np.sin(modalFreq[modeNo]))/(np.cosh(modalFreq[modeNo]) + np.cos(modalFreq[modeNo]))
        val = np.cosh(modalFreq[modeNo]*x) - np.cos(modalFreq[modeNo]*x) - sigma*(np.sinh(modalFreq[modeNo]*x) - np.sin(modalFreq[modeNo]*x))
        return val/normalizationFactor[modeNo]
    
    # \phi' (x)
    def dPhi(x, modeNo):
        sigma = (np.sinh(modalFreq[modeNo]) - np.sin(modalFreq[modeNo]))/(np.cosh(modalFreq[modeNo]) + np.cos(modalFreq[modeNo]))
        val = np.sinh(modalFreq[modeNo]*x) + np.sin(modalFreq[modeNo]*x) - sigma*(np.cosh(modalFreq[modeNo]*x) - np.cos(modalFreq[modeNo]*x))
        return modalFreq[modeNo]*val/normalizationFactor[modeNo]
    
    # \phi'' (x)
    def d2Phi(x, modeNo):
        sigma = (np.sinh(modalFreq[modeNo]) - np.sin(modalFreq[modeNo]))/(np.cosh(modalFreq[modeNo]) + np.cos(modalFreq[modeNo]))
        val = np.cosh(modalFreq[modeNo]*x) + np.cos(modalFreq[modeNo]*x) - sigma*(np.sinh(modalFreq[modeNo]*x) + np.sin(modalFreq[modeNo]*x))
        return (modalFreq[modeNo]**2)*val/normalizationFactor[modeNo]
   
    # \phi''' (x)
    def d3Phi(x, modeNo):
        sigma = (np.sinh(modalFreq[modeNo]) - np.sin(modalFreq[modeNo]))/(np.cosh(modalFreq[modeNo]) + np.cos(modalFreq[modeNo]))
        val = np.sinh(modalFreq[modeNo]*x) - np.sin(modalFreq[modeNo]*x) - sigma*(np.cosh(modalFreq[modeNo]*x) + np.cos(modalFreq[modeNo]*x))
        return (modalFreq[modeNo]**3)*val/normalizationFactor[modeNo]
    
    # \phi'''' (x)
    def d4Phi(x, modeNo):
        sigma = (np.sinh(modalFreq[modeNo]) - np.sin(modalFreq[modeNo]))/(np.cosh(modalFreq[modeNo]) + np.cos(modalFreq[modeNo]))
        val = np.cosh(modalFreq[modeNo]*x) - np.cos(modalFreq[modeNo]*x) - sigma*(np.sinh(modalFreq[modeNo]*x) - np.sin(modalFreq[modeNo]*x))
        return (modalFreq[modeNo]**4)*val/normalizationFactor[modeNo]

    # identity: returns the input
    def identity(x, modeNo):
        return x


'''
  Values for MODAL FREQUENCIES for a cantilever beam
  
  Function for NORMALIZATION of the MODAL SHAPES for the cantilever beam
  
'''
# Non-dimensional modal frequencies : \lambda 
#     from Meirovitch's Fundamentals of vibrations (page no: 420)
modalFreq = np.zeros(16)
modalFreq[1] = 1.8751; modalFreq[2] = 4.69409; modalFreq[3] = 7.85476; modalFreq[4] = 10.9955; modalFreq[5] = 14.1372;
modalFreq[6] = 17.2788; modalFreq[7] = 20.4204; modalFreq[8] = 23.5619; modalFreq[9] = 26.7035; modalFreq[10] = 29.8451;
modalFreq[11] = 32.9867; modalFreq[12] = 36.1283; modalFreq[13] = 39.2699; modalFreq[14] = 42.4115; modalFreq[15] = 45.5531;

normalizationFactor = np.zeros(modalFreq.size)
normalizationFactor = np.array([0., 0.99999892, 0.99999988, 1.00000016, 0.99999815, 1.00000112, 1.00000117, 1.00000117, 
                                0.99999906, 0.99999903, 1.00000049, 1.00016218, 1.00232238, 0.9922316,  0.96260839, 0.91879675])

# normalize nodal shapes such that \int_0^1 (\phi_i (x))^2 dx = 1
def normalizeModeShapes():
    global normalizationFactor
    
    for i in range(1,modalFreq.size):
        val = Integrator.integrateTwoFunc(ModeShape.normalizePhi, ModeShape.normalizePhi, i, i)
        normalizationFactor[i] = np.sqrt(val)
    return None

# Normalize modes compulsarily before computations are done
#normalizeModeShapes()



'''
   Class for complete response
'''

class Response:
    
    # Complete response
    def response(q):
        
        global xi # Domain description
        global N 
        
        res = np.zeros(xi.shape)
        for k in range(N):
           res += q[k]*ModeShape.phi(xi, k+1)
           
        return res
    
    # Plot response
    def plotResponse(t, x0, fps=10):  # fps = frames per second
        
        global N  # No of modes used for approximation
        global xi  # Domain description
        global beta_     # Mass ratio = (fluid)/(fluid + solid)
        global gamma_    # Non-dimensionalized acceleration due to gravity
        global alpha_    # Kelvin-Voigt viscoelasticity factor
        global sigma_    # Non-dimensionalized external dissipation constant
        
        global U 
        
        ### Setting up damping matrix (C)  and stiffness matrix (K) 
        # Equation: [I] \ddot{\vb{q}} + [C] \dot{\vb{q}} + [K] \vb{q} = \vb{0}
        C = np.zeros((N,N))
        K = np.zeros((N,N))

        for i in range(N):
            for j in range(N):
                K[i,j] = Coefficients.K(i+1,j+1)
                C[i,j] = Coefficients.C(i+1,j+1)
            
        ### State space model
        # \ddot{\vb{x}} = [A] \vb{x}  where A = [zeros(N,N), identity(N,N); -K, -C]
        # \vb{x} = {\vb{q} \\ \vb{z}},   \vb{z} = \dot{\vb{q}}
    
        A = np.zeros((2*N, 2*N))
        A[0:N, N:2*N] = np.eye(N)
        A[N:2*N, 0:N]  = -K
        A[N:2*N, N:2*N] = -C
        
        B = np.zeros((2*N, 1))
        
        # C = np.zeros((N, 2*N))
        # C[0:N, 0:N] = np.eye(N)
        # C[0:N, N:2*N] =  np.zeros((N,N))
        
        C = np.eye(2*N)

        D = np.zeros((2*N,1))
        
        # State space model
        sys = control.matlab.ss(A, B, C, D)
        #print(sys)
        
        ### Response to initial conditions: 
        q, t = control.matlab.initial(sys, t, x0)
        
        ### Solution reconstruction and plotting
        plt.ion()
        fig, ax = plt.subplots(figsize=(15,10))
        eta = np.zeros((len(xi),))
        for k in range(len(t)): 
            
            # Initialize to zero
            #eta[:] = 0
            
            # Superposition of modes for displacement
            for l in range(N):
                eta += q[k, l]*ModeShape.phi(xi, l+1)
                 
            plt.plot(xi, eta, color=colour[0], label='Displacement')
            ax.set_xlim((0,1)); ax.set_ylim((-1,1))
            ax.set_xlabel(r'$\xi$')
            ax.set_ylabel(r'$\eta $')
            ax.set_title(r'Time: %f' % t[k])
            plt.legend(loc='best')
            ax.grid()
            plt.pause(1.0/fps)
            plt.cla()
        
        #plt.show()
        plt.close()
        return None
    
'''

    UTILITY FUNCTIONS

'''

'''
  Step-wise "ascending" order for a mode
'''
def closestNeighbourMapping(previous, current):
   
    # Minimum distance function
    def distance(x, y):
        mindist = np.sqrt((x.real - y[0].real)**2 + (x.imag - y[0].imag)**2)
        indexTemp = 0

        for i in range(1,len(y)):
        	dist = np.sqrt((x.real - y[i].real)**2 + (x.imag - y[i].imag)**2)
        	if (dist < mindist):
        		mindist =  dist
        		indexTemp = i

        return indexTemp
    
    temp = np.zeros(current.size, dtype=np.complex64)

    current = list(current)

    for i in range(temp.size):
    	ind = distance(previous[i], current)
    	temp[i] = current[ind]
    	del(current[ind])

    return temp
    
'''
  Determine flutter flow velocity : check when the imaginary part becomes negative and return the corresponding U
  
'''

def determineFlutterSpeed(freqArray, U_array):
    
    # Initialize
    nModes, nIntervals = freqArray.shape
    flutterVelocityArray = [-1]*nModes # unphysical flow velocity
    flutterIndices = []   # Indices corresponding to flutter for different modes
    flutterMode = []
    flutterVelocity = []
    
    # Check for sign changes in imaginary part of frequency from positive to negative
    for i in range(nModes):
        for j in range(1, nIntervals-1):
            if (freqArray[i,j-1].real >= 0 and freqArray[i,j+1].imag < 0):
                flutterVelocityArray[i] = U_array[j]
                flutterIndices += [j]
                flutterMode += [i]
                flutterVelocity += [flutterVelocityArray[i]]
                break
    
    # print flutter flow velocity for different modes
    print("*****************************************************")
    print("   Flutter flow velocities for different modes ")
    print("     ")
        
    for i in range(nModes):
        if (flutterVelocityArray[i] != -1):
            print("   U_flutter = " + str(flutterVelocityArray[i]) + " for Mode " + str(i+1))
    
    print("*****************************************************")
    return (flutterMode, flutterIndices, flutterVelocity)


'''
  
   ROOT LOCUS:
       Function to determine variation of pole frequencies for a given set of parameters

'''

def rootLocus(U_array):
    
    global N  # No of modes used for approximation
    global xi  # Domain description
    global beta_     # Mass ratio = (fluid)/(fluid + solid)
    global gamma_    # Non-dimensionalized acceleration due to gravity
    global alpha_    # Kelvin-Voigt viscoelasticity factor
    global sigma_    # Non-dimensionalized external dissipation constant
    
    global U 
    
    
    ### Setting up damping matrix (C)  and stiffness matrix (K) 
    # Equation: [I] \ddot{\vb{q}} + [C] \dot{\vb{q}} + [K] \vb{q} = \vb{0}
    
    C = np.zeros((N,N))
    K = np.zeros((N,N))
    
    ### Real and imaginary parts of the eigenvalues
    frequencyArray = np.zeros((N,U_array.size), dtype=np.complex64)
    
    ### State space form
    # \ddot{\vb{x}} = [A] \vb{x}  where A = [zeros(N,N), identity(N,N); -K, -C]
    # \vb{x} = {\vb{q} \\ \vb{z}},   \vb{z} = \dot{\vb{q}}
    
    A = np.zeros((2*N, 2*N))
    A[0:N, N:2*N] = np.eye(N)
    
    for n in range(U_array.size):
        
        U = U_array[n]     # Non-dimensionalized flow velocity
        
        print("Processing U = %f" % U)
        for i in range(N):
            for j in range(N):
                K[i,j] = Coefficients.K(i+1,j+1)
                C[i,j] = Coefficients.C(i+1,j+1)
        
        # Populate K and C matrices into matrix A        
        A[N:2*N, 0:N]  = -K
        A[N:2*N, N:2*N] = -C
        
        ### Finding eigenvalues
        eigvals, eigvecs = la.eig(A)
        eigvals = -1.0j*eigvals
        #print(eigvals)
        
        ### Record eigenvalues
        if(n==0):
            frequencyArray[:, n] = np.sort(eigvals[::2]) #np.reshape(eigvals[::2], (N,1))
            
        else:
            # Order the eigenvalues according to nearest neighbours 
            frequencyArray[:, n] = closestNeighbourMapping(frequencyArray[:, n-1], np.sort(eigvals[::2]))
            
    ### Plotting variation of complex frequency of lower modes for a particular value of beta_        
    modes_to_plot = 4    # Number of modes to plot
    legend = []          # Dynamically create legend
    
    plt.ion() # interactive mode on
    fig, ax = plt.subplots(figsize=(15,10))
    
    for row in range(modes_to_plot):
        plt.plot(np.real(frequencyArray[row, :]), np.imag(frequencyArray[row, :]), color=colour[row], lw=line_width, \
                 label="Coupled mode "+ str(row + 1)) # marker=marker[row], 
        legend += ["Coupled mode "+ str(row + 1)]
            
    ax.set_xlim((0,130)); ax.set_ylim((-20,30))
    ax.set_xlabel(r'Real $(\omega)$')
    ax.set_ylabel(r'Imag $(\omega)$')
    ax.set_title(r'Variation of complex frequency with $U$ for $\beta$ = ' +  str(beta_))
    ax.grid()
    
    # Adding flutter U to the plot
    (flutterMode, flutterIndices, flutterVelocity) = determineFlutterSpeed(frequencyArray, U_array)
    markers = ['*', 'd']
    if (len(flutterMode) != 0):
        for k in range(len(flutterMode)):
            legend += ["$U_{f," + str(flutterMode[k]) +"}$ = " + str("%.4f"%flutterVelocity[k])]
            ax.scatter([frequencyArray[flutterMode[k], flutterIndices[k]].real], [0], s=150, marker=markers[k], color = 'red')

    # Adding U limits to the plot
    for row in range(modes_to_plot):
        ax.scatter(np.real(frequencyArray[row,0]), np.imag(frequencyArray[row,0]), s=150, \
               marker='o', color='black')
        ax.scatter(np.real(frequencyArray[row,-1]), np.imag(frequencyArray[row,-1]), s=150, \
               marker='x', color='black')         
    legend += ["$U$ = " + str(U_array[0]), "$U$ = " + str(U_array[-1])]
    
    #plt.legend(legend, bbox_to_anchor=(1,0), loc="lower left") #place legend in top right corner
    #plt.legend(legend, bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=N) #place legend above plot
    plt.legend(legend,loc="best") #place legend below plot
    #plt.show()
    figName = "rootLocus_beta" + str(beta_) + "_alpha_" + str(alpha_) + "_gamma_" + str(gamma_) + "_sigma_" + str(sigma_) + ".png"
    plt.savefig(figName, transparent = False, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    os.system('shotwell ' + figName + ' &')
    
    return None

''' 
    *******************************************************************************************************
                                                  MAIN PROGRAM 
    *******************************************************************************************************          
''' 


'''
    
     LINEAR ANALYSIS OF A CANTILEVERED PIPE CONVEYING FLUID
  
'''

### Parameters
beta_   = 0.2     # Mass ratio = (fluid)/(fluid + solid)
gamma_  = 0.0      # Non-dimensionalized acceleration due to gravity
alpha_  = 0.0      # Kelvin-Voigt viscoelasticity factor
sigma_  = 0.0       # Non-dimensionalized external dissipation constant
U       = 0.      # Non-dimensional flow velocity

### Domain discretization
xi = np.linspace(0, 1, 1000)

### Number of modes for Galerkin approximation
N = 10

### Non-dimensionalized flow velocity array
U_array = np.linspace(0,15,200)

### Variation of complex frequencies with flow velocity U
rootLocus(U_array)

'''
   Plotting COMPLETE response for a set of initial conditions
   
'''

U = 1.0

# Time array
t = np.linspace(0,5,100)

# Defining initial conditions
x0 = np.zeros((2*N, 1))
x0[0:4] = 0.2*np.ones((4,1))

# Plot dynamic response
Response.plotResponse(t, x0, fps=5)

########### Test integrate

# val, err = integrate.quad(ModeShape.identity, 0, 1, args=0)
# print(val)

# val = Integrator.integrateTwoFunc(ModeShape.identity, ModeShape.identity, 1, 1)
# print(val)

# val = Integrator.integrateThreeFunc(ModeShape.identity, ModeShape.identity, 1, 1)
# print(val)

# x = np.linspace(0, 1, 1000)
# y = x**2
# val = Integrator.numericalIntegration(y, x, 'simps')
# print(val)

# print(normalizationFactor)

# print(Integrator.integrateTwoFunc(ModeShape.phi, ModeShape.phi, 1, 1))


# # Plot response of a cantilever beam
# x = np.linspace(0, 1, 1000)
# T = np.linspace(0,20, 1000)
# plt.ion()
# fig, ax = plt.subplots(figsize=(15,10))

# for t in T:
#     plt.plot(x, 0.01*np.cos(np.sqrt(modalFreq[1])*t)*ModeShape.phi(x, 1))
#     ax.set_xlim((0,1)); ax.set_ylim((-2,2))
#     ax.set_xlabel(r'$\xi$')
#     ax.set_ylabel(r'$\eta $')
#     ax.set_title(r'Time: %f'%t)
#     ax.grid()
#     plt.pause(0.01)
#     plt.cla()

# #plt.show()


'''
Coding references:
    1. https://www.geeksforgeeks.org/python-classes-and-objects/
    2. https://physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html#numerical-integration
    3. https://www.geeksforgeeks.org/passing-function-as-an-argument-in-python/
    4. https://www.statology.org/matplotlib-legend-outside-plot/
    5. https://www.c-sharpcorner.com/article/create-animated-gif-using-python-matplotlib/
    6. https://stackabuse.com/matplotlib-change-scatter-plot-marker-size/
    7. https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    
'''
