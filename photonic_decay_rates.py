"""
 * Module Name: photonic_decay_rates
 * Description: evaluation of the photonic decay rates with Fermis golden rule
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import numpy as np

class photonic_decay_rates:
    
    '''
    
    * Class: photonic_decay_rates
    
    * Purpose: evaluation of the photonic decay rates with Fermis golden rule
        
    this class computes the photonic decay rates in the SnV with fermi's golden rule
    
    unit (THz)
    energy values of the SnV (evals)
    eigenbasis of the hamiltonian (S)
    atomic displacement factor (a)
    '''
    
    def __init__(self,THz,evals,S,a):
        
        '''
        
        * Paramters:
            
            THz (float): unit
            evals (array of floats): eigenvalues of the SnV Hamiltonian
            S (array of complex): eigenbasis of the SnV Hamiltonian
            a (float): constant for computing the decay rates which is evaluated in
                       the module atomic_displacment_factor
            
        '''
        
        self.THz=THz
        self.evals=evals
        self.S=S
        self.a=a
        
    def rates(self,H1,H2,H3):
        
        '''
        
        * Function: rates
        
        * Purpose: computation of photonic decay rates
        
        * Parameters:
            
            H1 (array of floats): interaction Hamiltonian 1 in standard basis
            H2 (array of floats): interaction Hamiltonian 2 in standard basis
            H3 (array of floats): interaction Hamiltonian 3 in standard basis

        * Returns:
            
            (list of floats): decay rates 
        
        '''
        
        c=3*10**(-4)*self.THz
        alpha=1/137
        n=2.47
        
        mm=0
        gas=[]
        for i in range(0,4):
            for j in range(4,8):
                w_ji=self.evals[j]-self.evals[i]
                bra_j=self.S[:,j].conjugate().reshape(1,8)
                ket_i=self.S[:,i].reshape(8,1)
                ga_ij=4*alpha*n*self.a**2*w_ji**3/(3*c**2)*(np.abs(np.dot(bra_j,np.dot(H1,ket_i)))**2+np.abs(np.dot(bra_j,np.dot(H2,ket_i)))**2+np.abs(np.dot(bra_j,np.dot(H3,ket_i)))**2)
                ga_ij=ga_ij[0,0]
                
                gas.append(ga_ij)
                
                mm=mm+1
                        
        return gas