# -*- coding: utf-8 -*-
"""
 * Module Name: atomic_factor
 * Description: computation of a constant
                for the interaction Hamiltonian
                and decay rates according to Fermi's golden rule
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import numpy as np
from tin_vacancy_characteristics import SnV


class atomic_factor:
    
    '''
    * Class: atomic_factor
    
    * Purpose: compute the factor a in the definition
    of the dipole operator d_i=ae*H_i,i=1,2,3 where H_i denotes
    the unitless interaction hamiltonian from the SnV class
    '''
    
    def __init__(self,THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u):
        self.THz=THz
        self.A=A
        self.kg=kg
        self.m=m
        self.alpha_g=alpha_g
        self.alpha_u=alpha_u
        self.beta_g=beta_g
        self.beta_u=beta_u
        
    def get_atomic_a(self):
        
        '''
        * Function: get_atomic_a
        
        Purpose: compute the factor a in the definition
        of the dipole operator d_i=ae*H_i,i=1,2,3 where H_i denotes
        the unitless interaction hamiltonian from the SnV class
        
        * Parameters: 
        
            THz (float): unit terahertz
            A (float): unit ampere
            kg (float): unit kilograms
            m (float): unit meter
            alpha_g (float): strain component ground state
            alpha_u (float): strain component excited state
            beta_g (float): strain component ground state
            beta_u (float): strain component excited state
        
        * Returns:
            
            (float): constant for the interaction Hamiltonian
                     and decay rates according to Fermi's golden rule
        '''
        
        H_0,mu_1,mu_2,mu_3,S=SnV(self.THz,self.A,self.kg,self.m,self.alpha_g,self.alpha_u,self.beta_g,self.beta_u,0,0).hamiltonian()
        
        # parameters: debye walley factor, speed of light, fine structure constant, refraction index, life time
        DW=0.6
        c=3*10**(-4)*self.m*self.THz
        alpha=1/137
        n=2.47
        T1=4.5*10**3*1/self.THz
        
        # unitless interaction hamiltonian
        H1,H2,H3=SnV(0,0,0,0,0,0,0,0,0,0).interaction()
        
        # computation of a
        evals=np.diag(H_0)
        S=S[:,::2]
        evals=evals[::2]        
        
        bra_1=np.conjugate(S[:,0].reshape(1,8))
        bra_2=np.conjugate(S[:,1].reshape(1,8))
        ket_3=S[:,2].reshape(8,1)
        val32=np.abs(np.dot(bra_2,np.dot(H1,ket_3)))**2+np.abs(np.dot(bra_2,np.dot(H2,ket_3)))**2+np.abs(np.dot(bra_2,np.dot(H3,ket_3)))**2
        val31=np.abs(np.dot(bra_1,np.dot(H1,ket_3)))**2+np.abs(np.dot(bra_1,np.dot(H2,ket_3)))**2+np.abs(np.dot(bra_1,np.dot(H3,ket_3)))**2
        
        w32=evals[2]-evals[1]
        w31=evals[2]-evals[0]
        
        fac=(3*c**2*DW)/(4*alpha*n*T1)
        
        denom=w32**3*val32+w31**3*val31
        
        a=np.sqrt(fac/denom)
        
        return a[0,0]