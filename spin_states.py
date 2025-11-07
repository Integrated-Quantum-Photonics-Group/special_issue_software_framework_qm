"""
 * Module Name: spin_states
 * Description: computation of the propagated spin states s.t. an optical pi/2 pulse
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: October 30, 2025
 * Version: 1.0
"""

import os
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
})
import numpy as np
from tin_vacancy_characteristics import SnV
from liouville_space_integration import liouville_space_integration
import scipy

from atomic_factor_g4v import atomic_factor
from dissipator import dissipator,hooke,phononic_absorption


def pi8_rotated_states(ps,alpha_g,alpha_u,beta_g,beta_u,Delta2,theta,B,x,C1A,C2B,C2A,C1B,TT):
    
    '''
    
    * Function: pi8_rotated_states
    
    * Purpose: propagate subject to optimized pi/8 gate parameters
    
    * Parameters:
        
        ps (float): unit picoseconds
        alpha_g (float): strain component ground state
        alpha_u (float): strain component excited state
        beta_g (float): strain component ground state
        beta_u (float): strain component excited state
        Delta2 (float): detuning of second pulse to level 2 (set to zero)
        theta (float): rotation angle in bloch-sphere (here pi/8)
        B (float): DC magnetic field orientation
        x (array of floats): pulse length (tau),
                             deuning of lasers to state 5 (Delta5),
                             DC field orientation (angle),
                             pulse polarization pulse 1 (phi1),
                             pulse polarization pulse 2 (phi2),
                             pulse polarization pulse 1 (theta1),
                             pulse polarization pulse 2 (theta2),
                             phase difference between laser 1 and 2 (phi_p),
                             pulse amplitude laser 1 (E1),
                             pulse amplitude laser 2 (E2),
        C (float): cooperativity
        TT (float): temperature
        
        * Returns:
            
            (array of complex): propagated state of |1><1|,
            (array of complex): propagated state of |1><2|,
            (array of complex): propagated state of |2><1|,
            (array of complex): propagated state of |2><2|,
            (float): approximation error

    '''
    
    THz=1
    ps=1/THz
    A=1
    kg=1
    m=1
    
    phi1,phi2,angle,theta1,theta2,tau,Delta5,phi_p,E1,E2=x

    H_0,mu_1,mu_2,mu_3,S=SnV(THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u,B,angle).hamiltonian()

    evals,_=np.linalg.eig(H_0)

    w1=evals[4]-Delta5
    w2=w1-evals[1]+Delta2

    Delta4=evals[3]-(w1-w2)

    H_0=np.diag([0,Delta2,evals[2],Delta4,Delta5,
                 evals[5]-w1,evals[6]-w1,evals[7]-w1])

    _ex = (1. / np.sqrt(6)) * np.array([1, -2, 1])
    _ey = (1. / np.sqrt(2)) * np.array([1, 0, -1])
    _ez = (1. / np.sqrt(3)) * np.array([1, 1, 1])
    _lattice_basis = np.array([_ex, _ey, _ez])
    a_1=np.array([np.sin(theta1)*np.cos(phi1),np.sin(theta1)*np.sin(phi1),np.cos(theta1)])
    a_2=np.array([np.sin(theta2)*np.cos(phi2),np.sin(theta2)*np.sin(phi2),np.cos(theta2)])

    a_1=np.dot(_lattice_basis.T,a_1)
    a_2=np.dot(_lattice_basis.T,a_2)

    sigma=tau/(2*np.sqrt(2*np.log(2)))
    T=10*sigma+100*ps

    aa=atomic_factor(THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u,'SnV').get_atomic_a()

    C_d=hooke(THz,kg,m).get_tensor()
    Rs=['gx','gy','ux','uy']
    chis=[]
    for R in Rs:
        chis.append(phononic_absorption(THz,kg,m,R,C_d).get_chi_R())
    
    L=dissipator(THz,m,kg,evals,S,aa,C1A,C2B,C2A,C1B,A,alpha_g,alpha_u,beta_g,beta_u,B,angle,chis,TT).vectorized_dissipator()

    rho01=np.array([[1,0],[0,0]],dtype=np.complex128)
    rho02=np.array([[0,1],[0,0]],dtype=np.complex128)
    rho03=np.array([[0,0],[1,0]],dtype=np.complex128)
    rho04=np.array([[0,0],[0,1]],dtype=np.complex128)

    rhoT1,data1,e1=liouville_space_integration(w1,w2,H_0,a_1,a_2,mu_1,mu_2,mu_3,E1,E2,sigma,T,L,rho01,phi_p).integrate()

    rhoT2,data2,e2=liouville_space_integration(w1,w2,H_0,a_1,a_2,mu_1,mu_2,mu_3,E1,E2,sigma,T,L,rho02,phi_p).integrate()

    rhoT3,data3,e3=liouville_space_integration(w1,w2,H_0,a_1,a_2,mu_1,mu_2,mu_3,E1,E2,sigma,T,L,rho03,phi_p).integrate()
    rhoT4,data4,e4=liouville_space_integration(w1,w2,H_0,a_1,a_2,mu_1,mu_2,mu_3,E1,E2,sigma,T,L,rho04,phi_p).integrate()
             
    return rhoT1,rhoT2,rhoT3,rhoT4,e1

def get_mixed_state_fidelity(rho,sigma):
    
    F=np.trace(scipy.linalg.sqrtm(np.dot(scipy.linalg.sqrtm(rho),np.dot(sigma,scipy.linalg.sqrtm(rho)))))**2
    
    return F

def continue_rotating(rhoT1,rhoT2,rhoT3,rhoT4):
    
    '''
    
    * Function: continue_rotating
    
    * Purpose: using the pi/8 rotated states construct the pi/2 rotated states
    
    * Parameters:
        
        rhoT1 (array of complex): propagated state of |1><1|,
        rhoT2 (array of complex): propagated state of |1><2|,
        rhoT3 (array of complex): propagated state of |2><1|,
        rhoT4 (array of complex): propagated state of |2><2|,
        
    * Returns:
        
        (list of arrays of complex): pi/2 rotated states
    
    '''
    
    rhos=[rhoT1,rhoT2,rhoT3,rhoT4]
    rhos_=[rhoT1,rhoT2,rhoT3,rhoT4]
    for i in range(3):
        [rhoT1n,rhoT2n,rhoT3n,rhoT4n]=rhos_
        rhoTT1=rhoT1n[0,0]*rhos[0]+rhoT1n[0,1]*rhos[1]+rhoT1n[1,0]*rhos[2]+rhoT1n[1,1]*rhos[3]
        rhoTT2=rhoT2n[0,0]*rhos[0]+rhoT2n[0,1]*rhos[1]+rhoT2n[1,0]*rhos[2]+rhoT2n[1,1]*rhos[3]
        rhoTT3=rhoT3n[0,0]*rhos[0]+rhoT3n[0,1]*rhos[1]+rhoT3n[1,0]*rhos[2]+rhoT3n[1,1]*rhos[3]
        rhoTT4=rhoT4n[0,0]*rhos[0]+rhoT4n[0,1]*rhos[1]+rhoT4n[1,0]*rhos[2]+rhoT4n[1,1]*rhos[3]
        states=np.vstack((rhoTT1,rhoTT2,rhoTT3,rhoTT4))
        rhos_=[rhoTT1,rhoTT2,rhoTT3,rhoTT4]

    return states


def get_spin_states(B,C1A,C2B,C2A,C1B,TT):
    
    '''
    
    * Function: get_spin_states
    
    * Purpose: propagated liouville basis and compute approximation error
    
    * Parameters:
        
        B (float): magnetic field strength ** to be removed **
        C (float): cooperativity
        flag (string): choice between B=3,1,0.3 T
        TT (float): temperature
    
    * Returns:
        
        (list of arrays of complex): pi/2 
    
    '''
    
    # units
    THz=1
    ps=1/THz

    alpha_g=0
    alpha_u=0
    beta_g=0
    beta_u=0

    Delta2=0
    theta=np.pi/8
    
    
    if np.abs(B*1e+24-3)<1e-10:

        tau=353.32/4
        Delta5=99.66*1e-3*2*np.pi
        angle=43.11/360*2*np.pi
        phi1=100.70/360*2*np.pi
        phi2=100.63/360*2*np.pi
        theta1=104.49/360*2*np.pi
        theta2=104.58/360*2*np.pi
        phi_p=90.03/360*2*np.pi
        E1=68.44*1e-3
        E2=72.33*1e-3
    
    elif np.abs(B*1e+24-1)<1e-10:
        tau,Delta5,angle,phi1,phi2,theta1,theta2,phi_p,E1,E2=64.23/4 ,110.53*1e-3*2*np.pi, 64.62/360*2*np.pi, 97.66/360*2*np.pi, 105.31/360*2*np.pi, 108.93/360*2*np.pi, 103.88/360*2*np.pi, 167.31/360*2*np.pi, 220.44*1e-3, 223.05*1e-3 #B1T
    elif np.abs(B*1e+24-0.3)<1e-10:
        phi1,phi2,angle,theta1,theta2,tau,Delta5,phi_p,E1,E2=1.47472268,  2.03073063,  1.42067733,  2.07416781,1.61893809, 37.91212733,  0.22445809,  2.45553716, 54.4*1e-3, 56.02*1e-3
    
    x=np.array([phi1,phi2,angle,theta1,theta2,tau,Delta5,phi_p,E1,E2])
    rhoT1,rhoT2,rhoT3,rhoT4,e1=pi8_rotated_states(ps,alpha_g,alpha_u,beta_g,beta_u,Delta2,theta,B,x,C1A,C2B,C2A,C1B,TT)
    states=continue_rotating(rhoT1,rhoT2,rhoT3,rhoT4)
    
    return states,e1
    
