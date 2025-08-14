"""
 * Module Name: liouville_space_integration
 * Description: numerical integration of the Lindblad-master equation subject to an optical Raman scheme
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from numba import jit
from scipy.integrate import odeint
    

class liouville_space_integration:
    
    '''
    
    * Class: liouville_space_integration
    
    * Purpose: solution to the Lindblad-master equation 
               for modeling both the control imperfections 
               and photonic as well as phononic processes
    
    '''
    
    def __init__(self,w1,w2,H_0,a_1,a_2,mu_1,mu_2,mu_3,E1,E2,sigma,T,L,rho0,phi_p):
        
        '''
        
        * Parameters:
            
            w1 (float): frequency of laser 1 
            w2 (float): frequency of laser 2
            H_0 (array of floats): diagonal Hamiltonian containing the energy eigenvalues
            a_1 (array of floats): polarization of laser 1
            a_2 (array of floats): polarization of laser 2
            mu_1 (array of floats): dipole operator 1 in the eigenbasis of the SnV Hamiltonian
            mu_2 (array of floats): dipole operator 2 in the eigenbasis of the SnV Hamiltonian
            mu_3 (array of floats): dipole operator 3 in the eigenbasis of the SnV Hamiltonian
            E1 (float): amplitude of laser 1
            E2 (float): amplitude of laser 2
            sigma (float): standard deviation of laser 1 and 2
            T (float): gate duration
            L (array of floats): dissipator
            rho0 (array of floats): initial state
            phi_p (float): phase difference between laser 1 and 2
        
        '''
        
        self.w1=w1
        self.w2=w2
        self.H_0=H_0
        self.a_1=a_1
        self.a_2=a_2
        self.mu_1=mu_1
        self.mu_2=mu_2
        self.mu_3=mu_3
        self.E1=E1
        self.E2=E2
        self.sigma=sigma
        self.T=T
        self.L=L
        self.rho0=rho0
        self.phi_p=phi_p
    
    @staticmethod
    @jit
    def dL(y, t, w1, w2, a_1, a_2, mu_1, mu_2, mu_3, H0, E1, E2, sigma, L, T, phi_p):
        
        '''
        
        * Function: dL
        
        * Purpose: define ODE dydt=dL(y,t,params)
        
        * Returns:
            
            (array of complex): ODE
        
        '''
        
        H1=np.zeros((8,8),dtype=np.complex128)
        H2=np.zeros((8,8),dtype=np.complex128)
        
        H1[0,4]=a_1[0]*mu_1[0,4]+a_1[1]*mu_2[0,4]+a_1[2]*mu_3[0,4]
        H1[0,5]=a_1[0]*mu_1[0,5]+a_1[1]*mu_2[0,5]+a_1[2]*mu_3[0,5]
        H1[0,6]=a_1[0]*mu_1[0,6]+a_1[1]*mu_2[0,6]+a_1[2]*mu_3[0,6]
        H1[0,7]=a_1[0]*mu_1[0,7]+a_1[1]*mu_2[0,7]+a_1[2]*mu_3[0,7]
        
        
        H1[2,4]=a_1[0]*mu_1[2,4]+a_1[1]*mu_2[2,4]+a_1[2]*mu_3[2,4]
        H1[2,5]=a_1[0]*mu_1[2,5]+a_1[1]*mu_2[2,5]+a_1[2]*mu_3[2,5]
        H1[2,6]=a_1[0]*mu_1[2,6]+a_1[1]*mu_2[2,6]+a_1[2]*mu_3[2,6]
        H1[2,7]=a_1[0]*mu_1[2,7]+a_1[1]*mu_2[2,7]+a_1[2]*mu_3[2,7]
        
        
        H1[1,4]=np.exp(+1j*(w1-w2)*t)*(a_1[0]*mu_1[1,4]+a_1[1]*mu_2[1,4]+a_1[2]*mu_3[1,4])
        H1[1,5]=np.exp(+1j*(w1-w2)*t)*(a_1[0]*mu_1[1,5]+a_1[1]*mu_2[1,5]+a_1[2]*mu_3[1,5])
        H1[1,6]=np.exp(+1j*(w1-w2)*t)*(a_1[0]*mu_1[1,6]+a_1[1]*mu_2[1,6]+a_1[2]*mu_3[1,6])
        H1[1,7]=np.exp(+1j*(w1-w2)*t)*(a_1[0]*mu_1[1,7]+a_1[1]*mu_2[1,7]+a_1[2]*mu_3[1,7])
        
        
        
        H1[3,4]=np.exp(+1j*(w1-w2)*t)*(a_1[0]*mu_1[3,4]+a_1[1]*mu_2[3,4]+a_1[2]*mu_3[3,4])
        H1[3,5]=np.exp(+1j*(w1-w2)*t)*(a_1[0]*mu_1[3,5]+a_1[1]*mu_2[3,5]+a_1[2]*mu_3[3,5])
        H1[3,6]=np.exp(+1j*(w1-w2)*t)*(a_1[0]*mu_1[3,6]+a_1[1]*mu_2[3,6]+a_1[2]*mu_3[3,6])
        H1[3,7]=np.exp(+1j*(w1-w2)*t)*(a_1[0]*mu_1[3,7]+a_1[1]*mu_2[3,7]+a_1[2]*mu_3[3,7])
        
        
        
        H2[0,4]=np.exp(-1j*(w1-w2)*t)*(a_2[0]*mu_1[0,4]+a_2[1]*mu_2[0,4]+a_2[2]*mu_3[0,4])
        H2[0,5]=np.exp(-1j*(w1-w2)*t)*(a_2[0]*mu_1[0,5]+a_2[1]*mu_2[0,5]+a_2[2]*mu_3[0,5])
        H2[0,6]=np.exp(-1j*(w1-w2)*t)*(a_2[0]*mu_1[0,6]+a_2[1]*mu_2[0,6]+a_2[2]*mu_3[0,6])
        H2[0,7]=np.exp(-1j*(w1-w2)*t)*(a_2[0]*mu_1[0,7]+a_2[1]*mu_2[0,7]+a_2[2]*mu_3[0,7])
          
        
        H2[2,4]=np.exp(-1j*(w1-w2)*t)*(a_2[0]*mu_1[2,4]+a_2[1]*mu_2[2,4]+a_2[2]*mu_3[2,4])
        H2[2,5]=np.exp(-1j*(w1-w2)*t)*(a_2[0]*mu_1[2,5]+a_2[1]*mu_2[2,5]+a_2[2]*mu_3[2,5])
        H2[2,6]=np.exp(-1j*(w1-w2)*t)*(a_2[0]*mu_1[2,6]+a_2[1]*mu_2[2,6]+a_2[2]*mu_3[2,6])
        H2[2,7]=np.exp(-1j*(w1-w2)*t)*(a_2[0]*mu_1[2,7]+a_2[1]*mu_2[2,7]+a_2[2]*mu_3[2,7])
        
        
        
        H2[1,4]=a_2[0]*mu_1[1,4]+a_2[1]*mu_2[1,4]+a_2[2]*mu_3[1,4]
        H2[1,5]=a_2[0]*mu_1[1,5]+a_2[1]*mu_2[1,5]+a_2[2]*mu_3[1,5]
        H2[1,6]=a_2[0]*mu_1[1,6]+a_2[1]*mu_2[1,6]+a_2[2]*mu_3[1,6]
        H2[1,7]=a_2[0]*mu_1[1,7]+a_2[1]*mu_2[1,7]+a_2[2]*mu_3[1,7]
        
        
        H2[3,4]=a_2[0]*mu_1[3,4]+a_2[1]*mu_2[3,4]+a_2[2]*mu_3[3,4]
        H2[3,5]=a_2[0]*mu_1[3,5]+a_2[1]*mu_2[3,5]+a_2[2]*mu_3[3,5]
        H2[3,6]=a_2[0]*mu_1[3,6]+a_2[1]*mu_2[3,6]+a_2[2]*mu_3[3,6]
        H2[3,7]=a_2[0]*mu_1[3,7]+a_2[1]*mu_2[3,7]+a_2[2]*mu_3[3,7]
        
        H2=H2*np.exp(1j*phi_p)
        H2dag=np.conjugate(H2.T)
        H2=H2+H2dag
        
        H1dag=np.conjugate(H1.T)
        H1=H1+H1dag
        
        
        H=H0+E1*np.exp(-(t-T/2)**2/(4*sigma**2))*H1+E2*np.exp(-(t-T/2)**2/(4*sigma**2))*H2
        
        I=np.eye(8,dtype=np.complex128)
        H_=-1j*(np.kron(I,H)-np.kron(H.T,I))+L
        
        Hre=np.real(H_)
        Him=np.imag(H_)
        Hh1=np.hstack((Hre,-Him))
        Hh2=np.hstack((Him,Hre))
        Htot=np.vstack((Hh1,Hh2))
        
        return np.dot(Htot,y)
    
    def reshape_to_list(self,arr):
        
        '''
        
        * Function: reshape_to_list
        
        * Purpose: reshape m x n array to mn x 1 array
        
        * Parameters:
            
            arr (array of complex)
        
        * Returns:
            
            (array of complex): reshaped array
        
        '''
        
        [m,n]=np.shape(arr)
        a=np.zeros((int(m*n),),dtype=np.complex128)
        for i in range(n):
            a[i*m:(i+1)*m]=arr[:,i]
        
        return a
    
    def reshape_(self,arr):
        
        '''
        
        * Function: reshape_
        
        * Purpose: reshape mn x 1 array to 2 x 1 array
        
        * Parameters:
            
            arr (array of complex)
        
        * Returns:
            
            (array of complex): reshaped array
        
        '''
        
        m=len(arr)
        n=int(m/8)
        a=np.zeros((n,n),dtype=np.complex128)
        for i in range(n):
            a[:,i]=arr[i*n:(i+1)*n]
    
        return a
    
    def integrate(self):
        
        '''
        
        * Function: integrate
        
        * Purpose: solve ODE
        
        * Returns:
            
            (array of complex): final state of integration
            (array of complex): states of integration at every time step
            (float): approximation error
        
        '''
        
        rho_0=np.zeros((8,8),dtype=np.complex128)
        rho_0[0:2,0:2]=self.rho0
        y_0=self.reshape_to_list(rho_0)
        y_0=np.hstack((np.real(y_0),np.zeros((64,))))
        t = np.linspace(0, self.T, int((self.T)/1e-2))
        N = 100 * int((self.T) / (2 * np.pi / (self.w1)))
        data=odeint(self.dL, y_0, t, args = (self.w1, self.w2, self.a_1, self.a_2, self.mu_1, self.mu_2, self.mu_3, self.H_0, self.E1, self.E2, self.sigma, self.L, self.T, self.phi_p), rtol = 10**(-10), atol = 10**(-12), mxstep = N)
        state=data[-1]                      
        rho_T=state[0:64]+1j*state[64:128]
        rho_T=self.reshape_(rho_T)
        rhoT=rho_T[0:2,0:2]
        rhoT_proj=np.zeros((8,8),dtype=np.complex128)
        rhoT_proj[0:2,0:2]=rhoT
        e=np.linalg.norm(rhoT_proj-rho_T, ord=1)
        
        return rhoT,data,e