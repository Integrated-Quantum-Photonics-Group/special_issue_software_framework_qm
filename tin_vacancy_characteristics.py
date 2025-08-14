"""
 * Module Name: tin_vacancy_characteristics
 * Description: Hamiltonian model for the SnV
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import numpy as np
from scipy.linalg import block_diag
import numpy.linalg as linalg

class SnV:
    
    '''
    
    * Class: SnV
    
    * Purpose: Model SnV 
    
    * Note: this class computes the diagonalized SnV hamiltonian
    and the dipole operators in the eigenbasis of the hamiltonian
    
    '''
    
    def __init__(self,THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u,B,angle):
        
        '''
        
        * Parameters:
            
            THz (float): unit Terahertz
            A (float): unit Ampere
            kg (float): unit kilograms
            m (float): unit meters
            alpha_g (float): strain component ground state
            alpha_u (float): strain component excited state
            beta_g (float): strain component ground state
            beta_u (float): strain component excited state
            B (float): DC field orientation

        '''
        
        self.THz=THz
        self.A=A
        self.kg=kg
        self.m=m
        self.alpha_g=alpha_g
        self.alpha_u=alpha_u
        self.beta_g=beta_g
        self.beta_u=beta_u
        self.B=B
        self.angle=angle
    
    def ham(self,delta,la,Y_x,Y_y,alpha,beta,q,ga_L,ga_S,B):
        
        '''
        
        * Function: ham
        
        * Purpose: model SnV
        
        * Parameters:
            
            delta (float): ZPL SnV
            Y_x (float): Jahn-Teller coefficient
            Y_y (float): Jahn-Teller coefficient
            alpha (float): strain component
            beta (float): strain component
            q (float): coefficient for Zeeman term
            ga_L (float): coefficient for Zeeman term
            ga_S (float): coefficient for Zeeman term
            B (array of floats): DC magnetic field vector
            
        * Returns:
            
            (array of complex): Hamiltonian
        
        '''
        
        # magnetic field components
        Bx=B[0]
        By=B[1]
        Bz=B[2]
        
        # hamiltonian content for zero-phonon line
        H_A=delta*np.eye(4,dtype=np.complex128)
        
        # spin orbit coupling
        H_SO=np.array([[0,0,-1j*la,0],
                       [0,0,0,1j*la],
                       [1j*la,0,0,0],
                       [0,-1j*la,0,0]],dtype=np.complex128)
    
        # jahn-teller
        H_JT=np.array([[Y_x,0,Y_y,0],
                       [0,Y_x,0,Y_y],
                       [Y_y,0,-Y_x,0],
                       [0,Y_y,0,-Y_x]],dtype=np.complex128) 
        
        # extrinsic strain
        H_ST=np.array([[alpha,0,beta,0],
                       [0,alpha,0,beta],
                       [beta,0,-alpha,0],
                       [0,beta,0,-alpha]],dtype=np.complex128) 
        
        # zeeman term
        H_Z1=q*ga_L*np.array([[0,0,1j*Bz,0],
                                  [0,0,0,1j*Bz],
                                  [-1j*Bz,0,0,0],
                                  [0,-1j*Bz,0,0]],dtype=np.complex128)
        
        # zeeman term
        H_Z2=ga_S*np.array([[Bz,Bx-1j*By,0,0],
                                [Bx+1j*By,-Bz,0,0],
                                [0,0,Bz,Bx-1j*By],
                                [0,0,Bx+1j*By,-Bz]],dtype=np.complex128)
        
        # zeeman term
        H_Z=H_Z1+H_Z2
        
        # total hamiltonian              
        H=H_A+H_SO+H_JT+H_ST+H_Z
            
        return H
    
    def interaction(self):
        
        
        '''
        
        * Function: interaction
        
        * Purpose: interaction Hamiltonian
        
        * Returns:
            
            (array of floats): interaction Hamiltonian 1
            (array of floats): interaction Hamiltonian 2
            (array of floats): interaction Hamiltonian 3
        
        '''
        # unitless interaction hamiltonian 
        H_1=-np.array([[0,0,1,0],
                      [0,0,0,-1],
                      [1,0,0,0],
                      [0,-1,0,0]],dtype=np.complex128)
        
        H_1=np.kron(H_1,np.eye(2,dtype=np.complex128))
        
        H_2=-np.array([[0,0,0,-1],
                      [0,0,-1,0],
                      [0,-1,0,0],
                      [-1,0,0,0]],dtype=np.complex128)
        
        H_2=np.kron(H_2,np.eye(2,dtype=np.complex128))
        
        H_3=-2*np.array([[0,0,1,0],
                         [0,0,0,1],
                         [1,0,0,0],
                         [0,1,0,0]],dtype=np.complex128)
        
        H_3=np.kron(H_3,np.eye(2,dtype=np.complex128))
        
        return H_1,H_2,H_3
    
    def hamiltonian(self):
        
        '''
        
        * Function: hamiltonian
        
        * Purpose: compute the diagonalized SnV hamiltonian
        and the dipole operators in the eigenbasis of the hamiltonian
        
        * Returns:
            
            (array of floats): diagonal Hamiltonian
            (array of complex): dipole operator 1
            (array of complex): dipole operator 2
            (array of complex): dipole operator 3
            (array of complex): eigenbasis of the SnV

        '''
        
        # parameters
        pi=np.pi
        delta_g=0*self.THz*2*pi
        la_g=815/2*1e-3*self.THz*2*pi
        Y_x_g=65*1e-3*self.THz*2*pi
        Y_y_g=0*self.THz

        q_g=0.15
        me=9.109*1e-31*self.kg
        e=1.602*1e-7*self.A*(1/self.THz)
        ga_L=e/(2*me)
        ga_S=ga_L # attention, this is a new result (it was ga_S=2ga_L)

        delta_u=484.32*self.THz*2*pi
        la_u=2355/2*1e-3*self.THz*2*pi
        Y_x_u=855*1e-3*self.THz*2*pi
        Y_y_u=0*self.THz
        q_u=q_g
        
        alpha_g=self.alpha_g
        beta_g=self.beta_g
        alpha_u=self.alpha_u
        beta_u=self.beta_u
        
        B=self.B*np.array([np.sin(self.angle),0,np.cos(self.angle)])
        H_g=self.ham(delta_g,la_g,Y_x_g,Y_y_g,alpha_g,beta_g,q_g,ga_L,ga_S,B)
        H_u=self.ham(delta_u,la_u,Y_x_u,Y_y_u,alpha_u,beta_u,q_u,ga_L,ga_S,B)
        H=block_diag(H_g,H_u)
        
        # diagonalization and dipole operator in new basis
        evals,evecs = linalg.eig(H)
        evals=np.real(evals)
        ind=np.argsort(evals)
        evals=evals[ind]
        emin=evals[0]
        evals=evals-emin
        S=evecs[:,ind]
        Sdag=np.conjugate(S.T)
        
        H_0=np.diag(evals)
        
        H_1,H_2,H_3=self.interaction()

        mu_1=np.dot(Sdag,np.dot(H_1,S))
        mu_2=np.dot(Sdag,np.dot(H_2,S))
        mu_3=np.dot(Sdag,np.dot(H_3,S))
        
        return H_0,mu_1,mu_2,mu_3,S
    
    


        