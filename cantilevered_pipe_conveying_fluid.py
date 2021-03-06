#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package cantilevered_pipe_conveying_fluid
#  Dynamics of cantilevered pipe conveying fluid
#  @author G R Krishna Chand Avatar, MTech (Aero) 3rd sem


# ##################################### LOADING PACKAGES  ###########################################

import os                        # for calling system functions in Ubuntu
import numpy as np               # for Numerical Python library
import matplotlib.pyplot as plt  # for plotting 
from scipy import integrate      # for numerical integration 
import scipy.linalg as la        # for linear algebra operations
import mpmath                    # for arbitrary-precision math library
import control.matlab            # for state-space modelling and time response (can be installed using: pip install control)
from celluloid import Camera     # for animation


# ################################ Done LOADING PACKAGES  ###########################################

# Font size control and global plot parameters
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
colour = ["red", "blue", "green", "magenta", "brown"]



## Class containing various integration procedures 
#  Contains both `functional-based and value-based numerical integration procedures
class Integrator:
    
    ## Init method or constructor   
    def __init__(self, name):  
        self.name = name  
            
    ## Print something again
    def printAgain():
        print("I also work")
        
    ## Print something
    def printMe():
        print("This class works")
        Integrator.printAgain()
        return None
    
    ## Integrate a function over a given interval
    # @param f1 Function to be integrated
    def integrateFunc(f1, i = 0,  lowerLim = 0, upperLim = 1):

        # integrate
        val, err = integrate.quadrature(f1, lowerLim, upperLim, args = i, tol=1e-9, rtol=1e-10, maxiter=100) # , limit=100, epsabs=1e-10)
        return val
        
    ## Integrate product of two functions over a given interval
    # @param f1 First function
    # @param f2 Second function
    def integrateTwoFunc(f1, f2, i, j, lowerLim = 0, upperLim = 1):
                
        # wrapper function: return product of two functions
        def twoFunc(x, i, j):
            #return ModeShape.phi(x, i)*ModeShape.dPhi(x, j)
            return f1(x, i) * f2(x, j)
        
        # integrate
        val, err = integrate.quadrature(twoFunc, lowerLim, upperLim, args=(i,j), tol=1e-9, rtol=1e-10, maxiter=150)
        return val
    
    ## Integrate product of three functions over a given interval
    # @param f1 First function
    # @param f2 Second function
    def integrateThreeFunc(f1, f2, i, j, lowerLim = 0, upperLim = 1):
             
        # lambda function
        identity = lambda x: x
        
        # wrapper function: return product of three functions
        def threeFunc(x, i, j):
            return identity(x)*f1(x, i) * f2(x, j)
        
        # integrate
        val, err = integrate.quadrature(threeFunc, lowerLim, upperLim, args=(i,j), tol=1e-9, rtol=1e-10, maxiter=100)
        return val
    
    ## Numerically integrate an array of float numbers using trapezoidal, simpson or rhomberg method
    # @param y Array of numbers to be integrated
    # @param x Array over which integration is desired
    def numericalIntegration(y, x, scheme = 'simps'):
        
        if (scheme == 'trapz'):
            return integrate.trapz(y, x)
        
        if (scheme == 'simps'):
            return integrate.simps(y, x)
        
        if (scheme == 'rhom'):
            return integrate.rhom(t, dx = x[1] - x[0])

## Global variable: number of integration points
# Number of integration points
numIntegrationPoints = 2**8 + 1 # number of integration points

## Class to determine various coefficients
#  The following coefficients are determined:
#   \li \c  Stationary damping and stiffness matrices: \f$C_{ij}, K_{ij}\f$
#   \li \c  Analytically determined: \f$b_{ij}\f$, \f$c_{ij}\f$, \f$d_{ij}\f$
#   \li \c  Numerically determined: \f$\alpha_{ijkl}\f$, \f$\beta_{ijkl}\f$, \f$\gamma_{ijkl}\f$
class Coefficients:
    
    global modalEig
    global normalizationFactor
    global numIntegrationPoints # number of integration points
    
    ## To determine component for damping matrix 
    # \f$C_{ij} = \alpha \lambda_j^4 \delta_{ij} + 2\sqrt{\beta} U b_{ij}\f$
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
        return (sigma_ + alpha_ * modalEig[j]**4)*kronecker(i, j) + 2.0*np.sqrt(beta_)*U*Coefficients.b(i, j)
    
    ## To determine component for stiffness matrix
    # \f$K_{ij} = \lambda_j^4 \delta_{ij} + (U^2 - \gamma) c_{ij} + \gamma (d_{ij} + b_{ij})\f$
    def K(i, j):
        global U          # Non-dimensionalized flow velocity
        global gamma_     # Non-dimensionalized acceleration due to gravity
        
        # wrapper function
        def kronecker(i, j):
            if (i == j):
                return 1
            else:
                return 0
        return modalEig[j]**4 * kronecker(i, j) + (U**2 - gamma_)*Coefficients.c(i, j) + gamma_*(Coefficients.d(i, j) + Coefficients.b(i, j))
    
    ## To determine coefficient \f$b_{ij} = \int_0^1 \phi_i \phi_j' d\xi\f$
    def b(i, j):
        #return Integrator.integrateTwoFunc(ModeShape.phi, ModeShape.dPhi, i, j)
        # Exact expressions
        if(i!=j):
            return 4.0/((modalEig[i]/modalEig[j])**2 + (-1)**(i+j))
        else:
            return 2.0
    
    ## To determine coefficient \f$c_{ij} = \int_0^1 \phi_i \phi_j'' d\xi\f$
    def c(i, j):
        #return Integrator.integrateTwoFunc(ModeShape.phi, ModeShape.d2Phi, i, j)
    
        # Exact expressions
        sigma = lambda x: (np.sinh(x) - np.sin(x))/(np.cosh(x) + np.cos(x))
        
        if(i!=j):
            return 4.0*(modalEig[j]*sigma(modalEig[j]) - modalEig[i]*sigma(modalEig[i]))/((-1.0)**(i+j) - (modalEig[i]/modalEig[j])**2)
        else:
            return modalEig[j]*sigma(modalEig[j])*(2.0 - modalEig[i]*sigma(modalEig[i]))

    ## To determine coefficient \f$d_{ij} = \int_0^1 \xi \phi_i \phi_j'' d\xi\f$
    def d(i, j):
        #return Integrator.integrateThreeFunc(ModeShape.phi, ModeShape.d2Phi, i, j)    
        
        # Exact expressions
        sigma = lambda x: (np.sinh(x) - np.sin(x))/(np.cosh(x) + np.cos(x))
        
        if(i!=j):
            val = 4.0*(modalEig[j]*sigma(modalEig[j]) - modalEig[i]*sigma(modalEig[i]) + 2.0)/(1.0 - (modalEig[i]/modalEig[j])**4)*(-1)**(i+j)
            val -= (3.0 + (modalEig[i]/modalEig[j])**4)/(1.0 - (modalEig[i]/modalEig[j])**4)*Coefficients.b(i,j) 
            return val
        else:
            return modalEig[j]*sigma(modalEig[j])*(2.0 - modalEig[i]*sigma(modalEig[i]))/2.0 # c_{ii}/2
    
    ## To determine coefficient \f$\alpha_{ijkl}\f$
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
    
    ## To determine coefficient \f$\beta_{ijkl}\f$  
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
    
    ## To determine coefficient \f$\gamma_{ijkl}\f$
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


    # Coefficients of the modal equation for cantilever beam: \ddot{q}_s = \Lambda_s q_s  (\lambda_s is s^th modal frequency) 


## Class to define modal shape functions and their derivatives
# Contains function definition of cantilever beam modal shape functions and their derivatives

class ModeShape:
    
    ## Global variables
    # @param modalEig Modal eigenvalues
    # @param normalizationFactor Normalization factors for the modal shape functions
    global modalEig
    global normalizationFactor
    
    ## Unnormalized cantilever beam modal shape functions \f$\phi (x) \f$
    # @param x Spatial domain of the cantilevered pipe
    # @param modeNo Mode number (starts at 1)
    def normalizePhi(x, modeNo):
        sigma = (np.sinh(modalEig[modeNo]) - np.sin(modalEig[modeNo]))/(np.cosh(modalEig[modeNo]) + np.cos(modalEig[modeNo]))
        val = np.cosh(modalEig[modeNo]*x) - np.cos(modalEig[modeNo]*x) - sigma*(np.sinh(modalEig[modeNo]*x) - np.sin(modalEig[modeNo]*x))
        return val

    ##  Function for Normalization of the cantilever beam modal shape functions
    # Normalize modal shapes such that \f$\int_0^1 (\phi_i (x))^2 dx = 1\f$
    def normalizeModeShapes():
        global normalizationFactor
        global modalEig
    
        for i in range(1,modalEig.size):
          val = Integrator.integrateTwoFunc(ModeShape.normalizePhi, ModeShape.normalizePhi, i, i)
          normalizationFactor[i] = np.sqrt(val)
        return None
           
    ## Modal shape function \f$\phi (x)\f$
    # @param x Spatial domain of the cantilevered pipe
    # @param modeNo Mode number (starts at 1)
    def phi(x, modeNo):
        sigma = (np.sinh(modalEig[modeNo]) - np.sin(modalEig[modeNo]))/(np.cosh(modalEig[modeNo]) + np.cos(modalEig[modeNo]))
        val = np.cosh(modalEig[modeNo]*x) - np.cos(modalEig[modeNo]*x) - sigma*(np.sinh(modalEig[modeNo]*x) - np.sin(modalEig[modeNo]*x))
        return val/normalizationFactor[modeNo]
    
    ## First derivative of modal shape function \f$\phi' (x)\f$
    # @param x Spatial domain of the cantilevered pipe
    # @param modeNo Mode number (starts at 1)
    def dPhi(x, modeNo):
        sigma = (np.sinh(modalEig[modeNo]) - np.sin(modalEig[modeNo]))/(np.cosh(modalEig[modeNo]) + np.cos(modalEig[modeNo]))
        val = np.sinh(modalEig[modeNo]*x) + np.sin(modalEig[modeNo]*x) - sigma*(np.cosh(modalEig[modeNo]*x) - np.cos(modalEig[modeNo]*x))
        return modalEig[modeNo]*val/normalizationFactor[modeNo]
    
    ## Second derivative of modal shape function \f$\phi'' (x)\f$
    # @param x Spatial domain of the cantilevered pipe
    # @param modeNo Mode number (starts at 1)
    def d2Phi(x, modeNo):
        sigma = (np.sinh(modalEig[modeNo]) - np.sin(modalEig[modeNo]))/(np.cosh(modalEig[modeNo]) + np.cos(modalEig[modeNo]))
        val = np.cosh(modalEig[modeNo]*x) + np.cos(modalEig[modeNo]*x) - sigma*(np.sinh(modalEig[modeNo]*x) + np.sin(modalEig[modeNo]*x))
        return (modalEig[modeNo]**2)*val/normalizationFactor[modeNo]
   
    ## Third derivative of modal shape function \f$\phi' (x)\f$
    # @param x Spatial domain of the cantilevered pipe
    # @param modeNo Mode number (starts at 1)
    def d3Phi(x, modeNo):
        sigma = (np.sinh(modalEig[modeNo]) - np.sin(modalEig[modeNo]))/(np.cosh(modalEig[modeNo]) + np.cos(modalEig[modeNo]))
        val = np.sinh(modalEig[modeNo]*x) - np.sin(modalEig[modeNo]*x) - sigma*(np.cosh(modalEig[modeNo]*x) + np.cos(modalEig[modeNo]*x))
        return (modalEig[modeNo]**3)*val/normalizationFactor[modeNo]
    
    ## Fourth derivative of modal shape function \f$\phi' (x)\f$
    # @param x Spatial domain of the cantilevered pipe
    # @param modeNo Mode number (starts at 1)
    def d4Phi(x, modeNo):
        sigma = (np.sinh(modalEig[modeNo]) - np.sin(modalEig[modeNo]))/(np.cosh(modalEig[modeNo]) + np.cos(modalEig[modeNo]))
        val = np.cosh(modalEig[modeNo]*x) - np.cos(modalEig[modeNo]*x) - sigma*(np.sinh(modalEig[modeNo]*x) - np.sin(modalEig[modeNo]*x))
        return (modalEig[modeNo]**4)*val/normalizationFactor[modeNo]

    ## Identity function returns the input
    def identity(x, modeNo=1):
        return x

## Non-dimensional modal eigenvalues for a cantilever beam: \f$\lambda\f$
modalEig = np.zeros(16)
modalEig[1] = 1.8751; modalEig[2] = 4.69409; modalEig[3] = 7.85476; modalEig[4] = 10.9955; modalEig[5] = 14.1372;
modalEig[6] = 17.2788; modalEig[7] = 20.4204; modalEig[8] = 23.5619; modalEig[9] = 26.7035; modalEig[10] = 29.8451;
modalEig[11] = 32.9867; modalEig[12] = 36.1283; modalEig[13] = 39.2699; modalEig[14] = 42.4115; modalEig[15] = 45.5531;

normalizationFactor = np.zeros(modalEig.size)
normalizationFactor = np.array([0., 0.99999892, 0.99999988, 1.00000016, 0.99999815, 1.00000112, 1.00000117, 1.00000117, 
                                0.99999906, 0.99999903, 1.00000049, 1.00016218, 1.00232238, 0.9922316,  0.96260839, 0.91879675])



## Class for time-domain response of the cantilevered pipe system
# Contains the following functions
#  \li Superposition of modal responses
#  \li Complete response
#  \li Tip displacement history and kinetic energy dynamics of the complete system
class Response:
    
    ## Superposition of modal responses
    def superpose(q):
        
        global xi # Domain description
        global N 
        
        res = np.zeros(xi.shape)
        for k in range(N):
           res += q[k]*ModeShape.phi(xi, k+1)
           
        return res
    
    ## Plot and record animation of the complete response
    # @param t Time array for which simulation has to be done
    # @param displacement_ic Displacement initial conditions
    # @param velocity_ic Velocity initial conditions
    def plotResponse(t, displacement_ic, velocity_ic, fps=10, velocity_hammer_input = 'No', velocity_hammer_magnitude = 0.1, record_video = 'No'):  # fps = frames per second
        
        global N  # No of modes used for approximation
        global xi  # Domain description
        global beta_     # Mass ratio = (fluid)/(fluid + solid)
        global gamma_    # Non-dimensionalized acceleration due to gravity
        global alpha_    # Kelvin-Voigt viscoelasticity factor
        global sigma_    # Non-dimensionalized external dissipation constant
        global U 

        ### Projecting initial conditions on the Galerkin modes
        x0 = np.zeros((2*N, 1)) 

        # Projecting displacement
        for n in range(N):
            if (velocity_hammer_input == 'Yes'):
                x0[n] = 0 
            else:
                x0[n] = Integrator.numericalIntegration(displacement_ic*ModeShape.phi(xi, n+1), xi, scheme = 'simps')

        # Projecting velocity
        for n in range(N):
            if (velocity_hammer_input == 'Yes'):
                x0[n + N] = velocity_hammer_magnitude*ModeShape.phi(1.0, n+1)
            else:
                x0[n + N] = Integrator.numericalIntegration(velocity_ic*ModeShape.phi(xi, n+1), xi, scheme = 'simps')

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
    
        A = np.zeros((2*N, 2*N)) # State or system matrix
        A[0:N, N:2*N] = np.eye(N)
        A[N:2*N, 0:N]  = -K
        A[N:2*N, N:2*N] = -C
        
        B = np.zeros((2*N, 1))   # Control or input matrix
        
        C_ = np.eye(2*N)         # Output matrix

        D = np.zeros((2*N,1))    # Direct transmission or feed through matrix
        
        # State space model
        sys = control.matlab.ss(A, B, C_, D)
        #print(sys)
        
        ### Response to initial conditions: 
        q, t = control.matlab.initial(sys, t, x0)
        
        ### Solution reconstruction and plotting
        plt.ion()
        fig, ax = plt.subplots(2, 1, figsize=(10,15))

        if (record_video == 'Yes'):
            camera = Camera(fig) # initializing the camera

        for k in range(len(t)): 
                       
            ### Superposition of modes for displacement and velocity
            # Displacement
            ax[0].plot(xi, Response.superpose(q[k, 0:N]), color=colour[0], label='Displacement')
            ax[0].set_xlim((0,1)); ax[0].set_ylim((-0.2,0.2))
            ax[0].set_ylabel(r'Displacement, $\eta $')
            ax[0].text(0.15, 1.25, r'Dynamics of cantilevered pipe conveying fluid', transform=ax[0].transAxes, fontsize=MEDIUM_SIZE-3)
            ax[0].text(0.0, 1.15, r'Parameters: $\beta$ = %.3f, $U$ = %.3f, $\alpha$ = %.3f, $\gamma$ = %.3f, $\sigma$ = %.3f'% (beta_, U, alpha_, gamma_, sigma_), \
                       transform=ax[0].transAxes, fontsize=MEDIUM_SIZE-3)
            ax[0].text(0.4, 1.05, 'Time: %.5f'% t[k], transform=ax[0].transAxes, fontsize=MEDIUM_SIZE)
            ax[0].grid('on')

            # Velocity
            ax[1].plot(xi, Response.superpose(q[k, N:2*N]), color=colour[1], label='Velocity')
            ax[1].set_xlim((0,1)); ax[1].set_ylim((-1.0,1.0))
            ax[1].set_xlabel(r'$\xi$')
            ax[1].set_ylabel(r'Velocity, $\dot{\eta} $')
            ax[1].grid('on')

            # plt.legend(loc='best')
            # plt.cla()
            
            # Capture the snapshot of the figure
            if (record_video == 'Yes'):
                camera.snap()

            # Pause for the prescribed fps\
            plt.pause(1.0/fps)

            # Clear axes (comment it if camera is being used)
            if (record_video != 'Yes'):
                ax[0].clear(); ax[1].clear()
        
        # Create and save the animation to a file
        if (record_video == 'Yes'):
            animation = camera.animate()
            animation.save('test.mp4')
         
        #plt.show()
        plt.close()

        return None

    ## Record and plot displacement time history at a point and the kinet
    # @param t Time array for which simulation has to be done
    # @param displacement_ic Displacement initial conditions
    # @param velocity_ic Velocity initial conditions
    def pointHistoryAndEnergy(t, displacement_ic, velocity_ic, point = -1, save_plot = 'No', velocity_hammer_input = 'No', velocity_hammer_magnitude = 0.1): 
        
        global N  # No of modes used for approximation
        global xi  # Domain description
        global beta_     # Mass ratio = (fluid)/(fluid + solid)
        global gamma_    # Non-dimensionalized acceleration due to gravity
        global alpha_    # Kelvin-Voigt viscoelasticity factor
        global sigma_    # Non-dimensionalized external dissipation constant
        global U 

        ### Projecting initial conditions on the Galerkin modes
        x0 = np.zeros((2*N, 1)) 

        # Projecting displacement
        for n in range(N):
            if (velocity_hammer_input == 'Yes'):
                x0[n] = 0 
            else:
                x0[n] = Integrator.numericalIntegration(displacement_ic*ModeShape.phi(xi, n+1), xi, scheme = 'simps')

        # Projecting velocity
        for n in range(N):
            if (velocity_hammer_input == 'Yes'):
                x0[n + N] = velocity_hammer_magnitude*ModeShape.phi(1.0, n+1)
            else:
                x0[n + N] = Integrator.numericalIntegration(velocity_ic*ModeShape.phi(xi, n+1), xi, scheme = 'simps')

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
        C_ = np.eye(2*N)
        D = np.zeros((2*N,1))
        
        # State space model
        sys = control.matlab.ss(A, B, C_, D)
        #print(sys)
        
        ### Response to initial conditions: 
        q, t = control.matlab.initial(sys, t, x0)

        ### Initialize variables
        pointDisplacement = np.zeros(t.size)
        kineticEnergy = np.zeros(t.size)

        ### Record
        for k in range(len(t)):

            kineticEnergy[k] = 0.5*float(q[k, N:2*N].T @ (np.eye(N) @ q[k, N:2*N]))

            # Superpose
            for l in range(N):
                pointDisplacement[k] += q[k, l]*ModeShape.phi(xi[point], l+1)

        ### Plot point history 
        plt.ion()
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(t, pointDisplacement, color=colour[0], label='Displacement')
        ax.set_xlim((t[0],t[-1])); ax.set_ylim((-0.2,0.2))
        ax.set_ylabel(r'Point displacement, $\eta $')
        ax.set_xlabel(r'Time, $\tau$')
        #ax.text(0.25, 1.1, r'Dynamics of cantilevered pipe conveying fluid', transform=ax.transAxes, fontsize=MEDIUM_SIZE-3)
        ax.text(0.15, 1.05, r'Parameters: $\beta$ = %.3f, $U$ = %.3f, $\alpha$ = %.3f, $\gamma$ = %.3f, $\sigma$ = %.3f'% (beta_, U, alpha_, gamma_, sigma_), \
                       transform=ax.transAxes, fontsize=MEDIUM_SIZE-2)
        ax.text(0.4, 1.01, r'Probe location: $\xi$ = %.2f'% xi[point], transform=ax.transAxes, fontsize=MEDIUM_SIZE - 2)
        ax.grid('on')

        figName = "pointDisplacement_U_" + str(U) + "_beta_" + str(beta_) + "_alpha_" + str(alpha_) + "_gamma_" + str(gamma_) + "_sigma_" + str(sigma_) + ".png"    
        if (save_plot == 'Yes'):
            plt.savefig(figName, transparent = False, bbox_inches = 'tight', pad_inches = 0)
            os.system('shotwell ' + figName + ' &')
        else:
            plt.pause(10)

        plt.close()

        ### Plot rate of change of Total Energy
        plt.ion()
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(t, kineticEnergy, color=colour[0], label='kineticEnergy')
        ax.set_xlim((t[0], t[-1])); ax.set_ylim((0, 0.6))
        ax.set_ylabel(r'Kinetic energy, $T$')
        ax.set_xlabel(r'Time, $\tau$')
        #ax.text(0.25, 1.1, r'Dynamics of cantilevered pipe conveying fluid', transform=ax.transAxes, fontsize=MEDIUM_SIZE-3)
        ax.text(0.11, 1.1, r'Parameters: $\beta$ = %.3f, $U$ = %.3f, $\alpha$ = %.3f, $\gamma$ = %.3f, $\sigma$ = %.3f'% (beta_, U, alpha_, gamma_, sigma_), \
                       transform=ax.transAxes, fontsize=MEDIUM_SIZE-2)
        ax.text(0.3, 1.03, r'Kinetic energy, $T = \frac{1}{2} \vec{\dot{q}}^T~M~\vec{\dot{q}}$', transform=ax.transAxes, fontsize=MEDIUM_SIZE - 2)
        ax.grid('on')

        figName = "kineticEnergy_U_" + str(U) + "_beta_" + str(beta_) + "_alpha_" + str(alpha_) + "_gamma_" + str(gamma_) + "_sigma_" + str(sigma_) + ".png"    
        if (save_plot == 'Yes'):
            plt.savefig(figName, transparent = False, bbox_inches = 'tight', pad_inches = 0)
            os.system('shotwell ' + figName + ' &')
        else:
            plt.pause(10)

        plt.close()


        return None



## Class for root locus 
# To determine variation of complex frequencies (poles) for a given set of parameters \f$\beta,~\sigma,~\alpha,~\gamma\f$

class RootLocus:
    
    ## Function for determining and plottting root locus
    def rootLocus(U_array, modes_to_plot = 4, save_plot = 'No', returnEigenvalues = 'No'):  # modes_to_plot: Number of modes to plot
        
        global N         # No of modes used for approximation
        global xi        # Domain description
        global beta_     # Mass ratio = (fluid)/(fluid + solid)
        global gamma_    # Non-dimensionalized acceleration due to gravity
        global alpha_    # Kelvin-Voigt viscoelasticity factor
        global sigma_    # Non-dimensionalized external dissipation constant
        global U         # Non-dimensional flow velocity
        

        ### Print what is being done
        print("*** Determining  variation of complex frequency with flow velocity ***")    
        print('Parameters: beta = %.5f, gamma = %.5f, alpha = %.5f, sigma = %.5f'% (beta_, gamma_, alpha_, sigma_)) 
        print("   ")
        
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
                if(alpha_ == 0.0):
                    frequencyArray[:, n] = np.sort(eigvals[::2]) #np.reshape(eigvals[::2], (N,1))
                else:
                    frequencyArray[:, n] = np.sort(eigvals[::2]) #np.reshape(eigvals[::2], (N,1))
                
            else:
                # Order the eigenvalues according to nearest neighbours 
                frequencyArray[:, n] = closestNeighbourMapping(frequencyArray[:, n-1], np.sort(eigvals[::2]))
                
        ### Plotting variation of complex frequency of lower modes for a particular value of beta_        
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
        ax.set_title(r'Parameters: $\beta$ = %.5f, $\gamma$ = %.5f, $\alpha$ = %.5f, $\sigma$ = %.5f'% (beta_, gamma_, alpha_, sigma_))
        ax.grid()
        
        # Adding flutter U to the plot
        (flutterMode, flutterIndices, flutterVelocity) = RootLocus.determineFlutterSpeed(frequencyArray, U_array)
        markers = ['*', 'd', '+', 'x']
    
        if (len(flutterMode) != 0):
            for k in range(len(flutterMode)):
                legend += ["$U_{f," + str(flutterMode[k]) +"}$ = " + str("%.4f"%flutterVelocity[k])]
                ax.scatter([frequencyArray[flutterMode[k] - 1, flutterIndices[k]].real], [0], s=150, marker=markers[k], color = 'red')
    
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
        figName = "rootLocus_beta_" + str(beta_) + "_alpha_" + str(alpha_) + "_gamma_" + str(gamma_) + "_sigma_" + str(sigma_) + ".png"
    
        if (save_plot == 'Yes'):
            plt.savefig(figName, transparent = False, bbox_inches = 'tight', pad_inches = 0)
            os.system('shotwell ' + figName + ' &')
        else:
            plt.pause(10)

        plt.close()

        if (returnEigenvalues == 'Yes'):
            return frequencyArray
        else:  
            return None

    ## CRUDE function for determining and root locus scatter plot
    def rootLocusScatter(U_array, modes_to_plot = 4, save_plot = 'No', returnEigenvalues = 'No'):  # modes_to_plot: Number of modes to plot
        
        global N         # No of modes used for approximation
        global xi        # Domain description
        global beta_     # Mass ratio = (fluid)/(fluid + solid)
        global gamma_    # Non-dimensionalized acceleration due to gravity
        global alpha_    # Kelvin-Voigt viscoelasticity factor
        global sigma_    # Non-dimensionalized external dissipation constant
        global U         # Non-dimensional flow velocity
        

        ### Print what is being done
        print("*** Determining  variation of complex frequency with flow velocity ***")    
        print('Parameters: beta = %.3f, gamma = %.3f, alpha = %.3f, sigma = %.3f'% (beta_, gamma_, alpha_, sigma_)) 
        print("   ")
        
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
                if(alpha_ == 0.0):
                    frequencyArray[:, n] = np.sort(eigvals[::2]) #np.reshape(eigvals[::2], (N,1))
                else:
                    frequencyArray[:, n] = np.sort(eigvals[::2]) #np.reshape(eigvals[::2], (N,1))
                
            else:
                # Order the eigenvalues according to nearest neighbours 
                frequencyArray[:, n] = closestNeighbourMapping(frequencyArray[:, n-1], np.sort(eigvals[::2]))
                
        ### Plotting variation of complex frequency of lower modes for a particular value of beta_        
        legend = []          # Dynamically create legend
        
        plt.ion() # interactive mode on
        fig, ax = plt.subplots(figsize=(15,10))

        markers = ['*', 'd', '+']
        for row in range(modes_to_plot):
            ax.scatter(np.real(frequencyArray[row, :]), np.imag(frequencyArray[row, :]),  s=30, marker=markers[row], color = 'red')
            legend += ["Coupled mode "+ str(row + 1)]
                
        ax.set_xlim((0,130)); ax.set_ylim((-20,40))
        ax.set_xlabel(r'Real $(\omega)$')
        ax.set_ylabel(r'Imag $(\omega)$')
        ax.set_title(r'Parameters: $\beta$ = %.3f, $\gamma$ = %.3f, $\alpha$ = %.3f, $\sigma$ = %.3f'% (beta_, gamma_, alpha_, sigma_))
        ax.grid()
    
        plt.legend(legend,loc="best") #place legend below plot
        #plt.show()
        figName = "rootLocus_beta_" + str(beta_) + "_alpha_" + str(alpha_) + "_gamma_" + str(gamma_) + "_sigma_" + str(sigma_) + ".png"
    
        if (save_plot == 'Yes'):
            plt.savefig(figName, transparent = False, bbox_inches = 'tight', pad_inches = 0)
            os.system('shotwell ' + figName + ' &')

        else:
            plt.pause(30)

        plt.close()

        if (returnEigenvalues == 'Yes'):
            return print(frequencyArray)

        else:  
            return None

    
    ##  Function to determine flutter flow velocity: check when the imaginary part becomes negative and return the corresponding U
    def determineFlutterSpeed(freqArray, U_array):
        
        # Initialize
        nModes, nIntervals = freqArray.shape
        flutterVelocityArray = [-1]*nModes # Unphysical flow velocity
        flutterIndices = []   # Indices corresponding to flutter for different modes
        flutterMode = []
        flutterVelocity = []
        
        # Check for sign changes in imaginary part of frequency from positive to negative
        for i in range(nModes):
            for j in range(1, nIntervals-1):
                if (freqArray[i,j-1].imag >= 0 and freqArray[i,j+1].imag < 0):
                    flutterVelocityArray[i] = U_array[j]
                    flutterIndices += [j]
                    flutterMode += [i + 1]
                    flutterVelocity += [flutterVelocityArray[i]]
                    break
        
        # print flutter flow velocity for different modes
        if (len(flutterMode) != 0):
    
            print("*****************************************************")
            print("   Flutter flow velocities for different modes ")
            print("     ")
            for i in range(len(flutterMode)):
                print("   U_flutter = " + str(flutterVelocity[i]) + " for Mode " + str(flutterMode[i]))
            print("*****************************************************")
    
        return (flutterMode, flutterIndices, flutterVelocity)
        

##  Class for variation of critical flow velocity with mass ratio \f$\beta\f$ for a given set of parameters
class FlutterVelocityVsBeta:

    ##  Function to plot the variation of critical flow velocity with mass ratio \f$\beta\f$ for a given set of parameters
    def flutterVelocityVsBeta(U_array, save_plot = 'No', show_omega = 'No'):
        
        global N         # No of modes used for approximation
        global xi        # Domain description
        global beta_     # Mass ratio = (fluid)/(fluid + solid)
        global gamma_    # Non-dimensionalized acceleration due to gravity
        global alpha_    # Kelvin-Voigt viscoelasticity factor
        global sigma_    # Non-dimensionalized external dissipation constant
        global U         # Non-dimensional flow velocity
    

        ### Print what is being done
        print("*** Determining  variation of flutter flow velocity vs mass ratio ***")    
        print('Parameters: gamma = %.3f, alpha = %.3f, sigma = %.3f'% (gamma_, alpha_, sigma_)) 
        print("   ")

        ### Define beta array
        beta_array = np.linspace(0.00000,0.99,300) # excluding beta =  1 for ``stability'' sake
        
        ### Define flutter beta array
        flutterBeta_array = np.zeros(U_array.shape)

        ### Define flutter omega array
        flutterOmega_array = np.zeros(U_array.shape)
        
        ### Setting up damping matrix (C)  and stiffness matrix (K) 
        # Equation: [I] \ddot{\vb{q}} + [C] \dot{\vb{q}} + [K] \vb{q} = \vb{0}
        
        C = np.zeros((N,N))
        K = np.zeros((N,N))
        
        ### Real and imaginary parts of the eigenvalues
        frequencyArray = np.zeros((N,beta_array.size), dtype=np.complex64)
        
        ### State space form
        # \ddot{\vb{x}} = [A] \vb{x}  where A = [zeros(N,N), identity(N,N); -K, -C]
        # \vb{x} = {\vb{q} \\ \vb{z}},   \vb{z} = \dot{\vb{q}}
        
        A = np.zeros((2*N, 2*N))
        A[0:N, N:2*N] = np.eye(N)
        
        for n in range(U_array.size):
            
            U = U_array[n]     # Non-dimensionalized flow velocity
            
            print("Processing U = %f" % U)
            
            for m in range(beta_array.size): 
                
                beta_ = beta_array[m] # Mass ratio 

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
                if(m==0):
                    frequencyArray[:, m] = np.sort(eigvals[::2]) # 2 indicates taking only one of the complex conjugates                    
                else:
                    # Order the eigenvalues according to nearest neighbours 
                    frequencyArray[:, m] = closestNeighbourMapping(frequencyArray[:, m-1], np.sort(eigvals[::2]))
                    
            # beta_ for flutter
            (flutterBeta, flutterOmega) = FlutterVelocityVsBeta.determineFlutterBeta(frequencyArray, beta_array)
    
            # Assign flutterBeta to flutterBeta_array
            flutterBeta_array[n] = flutterBeta[0]

            # Assign flutterOmega to flutterOmega_array
            flutterOmega_array[n] = flutterOmega[0]
    
        ### Plotting variation of complex frequency of lower modes for a particular value of beta_        
        legend = []          # Dynamically create legend
        
        plt.ion() # interactive mode on
        fig, ax = plt.subplots(figsize=(9,18))

        plt.title(r'$\gamma$ = %.3f, $\alpha$ = %.3f, $\sigma$ = %.3f'% (gamma_, alpha_, sigma_))       
        ax.set_xlim((0,1)); ax.set_ylim((0,np.ceil(U_array[-1]) + 1))
        ax.set_xlabel(r'Mass ratio, $\beta$')
        ax.set_ylabel(r'Flutter flow velocity, $U_{f}$')
        ax.plot(flutterBeta_array, U_array, color='black', lw=line_width) #, label=r"$\alpha = \gamma = \sigma = 0$")
        ax.tick_params(axis='y')
        ax.grid(True)
        ax.set_xticks(0.1*np.arange(11))

        ## Plot critical omega on the same figure
        if (show_omega == 'Yes'):
            ax.grid(False)
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel(r'Flutter frequency, $\omega_{f}$', color=colour[1])
            ax2.plot(flutterBeta_array, flutterOmega_array, color=colour[1], lw=line_width) #, label=r"$\alpha = \gamma = \sigma = 0$")
            ax2.tick_params(axis='y', labelcolor=colour[1])
        
        #ax.set_title(r'Variation of dimensionless c $U_{cf}$ with $\beta$')
        #plt.legend(loc='best')

        fig.tight_layout() # otherwise the right y-label is slightly clipped

        plt.show()
        plt.pause(5)

        # Writing data to file and saving image
        fileName = "flutterVelocityVsBeta_alpha_" + str(alpha_) + "_gamma_" + str(gamma_) + "_sigma_" + str(sigma_)
        
        with open(fileName + ".dat", 'w') as f1:
            f1.write('# U_f flutter_beta   flutter_omega\n' )
            for k in range(len(U_array)):
                f1.write(str(U_array[k]) + "    " + str(flutterBeta_array[k]) + "    " + str(flutterOmega_array[k]) + "\n")
    
        if (save_plot == 'Yes'):
            plt.savefig(fileName  + ".png", transparent = False, bbox_inches = 'tight', pad_inches = 0)
            os.system('shotwell ' + fileName + '.png' + ' &')
        else:
            plt.pause(10)

        plt.close()

        # # Flutter Omega
        # plt.ion() # interactive mode on
        # fig, ax = plt.subplots(figsize=(9,18))
        
        # plt.plot(flutterOmega_array, U_array, color=colour[0], lw=line_width) #, label=r"$\alpha = \gamma = \sigma = 0$")
        # plt.title(r'$\gamma$ = %.3f, $\alpha$ = %.3f, $\sigma$ = %.3f'% (gamma_, alpha_, sigma_))       
        # #ax.set_xlim((0,1)); 
        # ax.set_ylim((0,np.ceil(U_array[-1]) + 1))
        # ax.set_xlabel(r'Critical frequency, $\omega_c$')
        # ax.set_ylabel(r'Critical flow velocity, $U_{cf}$')
        # #ax.set_title(r'Variation of dimensionless c $U_{cf}$ with $\beta$')
        # ax.grid()
        # #ax.set_xticks(0.1*np.arange(11))
        # #plt.legend(loc='best')

        # plt.show()
        # plt.pause(5)
        # # Writing data to file and saving image
        # fileName = "flutterVelocityVsOmega_alpha_" + str(alpha_) + "_gamma_" + str(gamma_) + "_sigma_" + str(sigma_)
        # plt.savefig(fileName  + ".png", transparent = False, bbox_inches = 'tight', pad_inches = 0)
        # plt.close()
    
        # if (show_plot == 'Yes'):
        #    os.system('shotwell ' + fileName + '.png' + ' &')
       
        return None

    ## Function to determine flutter flow velocity : check when the imaginary part becomes negative and return the corresponding U
    def determineFlutterBeta(freqArray, beta_array):
        
        # Initialize 
        nModes, nIntervals = freqArray.shape
        #flutterBetaArray = [-1]*nModes # Unphysical mass ratio
        #flutterIndices = []   # Indices corresponding to flutter for different modes
        #flutterMode = []
        flutterBeta = []
        flutterOmega = []
        
        # Check for sign changes in imaginary part of frequency from positive to negative
        for i in range(nModes):
            for j in range(1, nIntervals-1):
                if ((freqArray[i,j-1].imag >= 0 and freqArray[i,j+1].imag < 0) or (freqArray[i,j-1].imag < 0 and freqArray[i,j+1].imag >= 0)) :
                    #flutterIndices += [j]
                    #flutterMode += [i + 1]
                    flutterBeta += [beta_array[j]]
                    flutterOmega += [freqArray[i,j].real]
                    break
        
        # print flutter flow velocity for different modes
        if (len(flutterBeta) != 0):
            for i in range(1):
                print("   beta = " + str(flutterBeta[i]) + " for U_flutter = " + str(U))
            #print("*****************************************************")
    
        return (flutterBeta, flutterOmega)


## UTILITY FUNCTIONS

##  Function for step-wise "ascending" order for a mode
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



##  Function to plot variation of \f$\beta\f$ v/s flutter flow velocity \f$U\f$ 
def plotBetaVsVelocity():

    #global sigma_
    global alpha_
    global gamma_

    plt.ion() # interactive mode on
    fig, ax = plt.subplots(figsize=(9,18))

    #gamma_ = [0.0, 10.0, 100.0]
    #alpha_ = [0.0, 0.001, 0.002, 0.003]
    sigma_ = [0.0, 1.0, 2.0, 5.0, 8.0]

    for i in range(len(sigma_)):
        fileName = "flutterVelocityVsBeta_alpha_" + str(alpha_) + "_gamma_" + str(gamma_) + "_sigma_" + str(sigma_[i])
        data = np.loadtxt(fileName + '.dat', skiprows=1, delimiter='    ')
        #data = np.loadtxt(fileName, skiprows=1, delimiter=',')
        ax.plot(data[:,1], data[:,0], color=colour[i], lw=line_width, label=r"$\sigma = %.3f$"%sigma_[i])
    
    #plt.title(r'$\alpha$ = %.3f, $\sigma$ = %.3f'% (alpha_, sigma_))    
    #plt.title(r'$\gamma$ = %.3f, $\sigma$ = %.3f'% (gamma_, sigma_))  
    plt.title(r'$\alpha$ = %.3f, $\gamma$ = %.3f'% (alpha_, gamma_))   
    ax.set_xlim((0,1)); ax.set_ylim((0,18))
    ax.set_xlabel(r'Mass ratio, $\beta$')
    ax.set_ylabel(r'Flutter flow velocity, $U_{f}$')
    plt.legend(loc='best')
    ax.grid(True)
    ax.set_xticks(0.1*np.arange(11))
    plt.pause(2)

    #plt.savefig("flutterVelocityVsBeta_various_gamma.png", transparent = False, bbox_inches = 'tight', pad_inches = 0)
    #plt.savefig("flutterVelocityVsBeta_various_alpha.png", transparent = False, bbox_inches = 'tight', pad_inches = 0)
    plt.savefig("flutterVelocityVsBeta_various_sigma.png", transparent = False, bbox_inches = 'tight', pad_inches = 0)

    plt.close()

    return None


'''
   ********************************************************************************************************************** 

                           MAIN PROGRAM :  LINEAR ANALYSIS OF A CANTILEVERED PIPE CONVEYING FLUID

   **********************************************************************************************************************  
'''

### Parameters
beta_   = 0.2     # Mass ratio = (fluid)/(fluid + solid)
gamma_  = 0.0     # Non-dimensionalized acceleration due to gravity
alpha_  = 0.0   # Kelvin-Voigt viscoelasticity factor
sigma_  = 0.0       # Non-dimensionalized external dissipation constant
U       = 5.57      # Non-dimensional flow velocity

### Domain discretization
xi = np.linspace(0, 1, 501)

### Number of modes for Galerkin approximation
N = 10

### Non-dimensionalized flow velocity array
U_array = np.linspace(0,20,301)
#U_array = np.zeros((1,))

### Variation of complex frequencies with flow velocity U
#locus = RootLocus.rootLocus(U_array, modes_to_plot = 4, save_plot ='No', returnEigenvalues='Yes')
#print(locus[:,0])
#RootLocus.rootLocusScatter(U_array, modes_to_plot = 3, save_plot ='No')

### Variation of flutter flow velocity with mass ratio for a given set of parameters
U_array = np.linspace(4.9, 13.0, 101)
#FlutterVelocityVsBeta.flutterVelocityVsBeta(U_array[:], save_plot='Yes', show_omega = 'No')


### Variation of flutter flow velocity with mass ratio for different gamma_
#plotBetaVsVelocity()


'''
   Plotting COMPLETE response for a set of initial conditions
   
'''

# Non-dimensional flow velocity
U = 5.57

# Time array
t = np.linspace(0, 10, 501)

# Defining initial conditions
displacement_ic = 0.1*ModeShape.phi(xi, 1) 
velocity_ic = np.zeros(xi.shape)

# Plot dynamic response
#Response.plotResponse(t, displacement_ic, velocity_ic, fps=5, velocity_hammer_input = 'Yes', velocity_hammer_magnitude = 0.2, record_video = 'Yes')

# Plot point displacement history
Response.pointHistoryAndEnergy(t, displacement_ic, velocity_ic, point = -1, save_plot = 'Yes',  velocity_hammer_input = 'Yes', velocity_hammer_magnitude = 0.2)

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
#     plt.plot(x, 0.01*np.cos(np.sqrt(modalEig[1])*t)*ModeShape.phi(x, 1))
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
    8. https://github.com/jwkvam/celluloid (animation)
    9. https://matplotlib.org/gallery/api/two_scales.html (two vertical axes for one horizontal axis)
    
'''