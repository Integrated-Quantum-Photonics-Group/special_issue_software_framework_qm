"""
 * Module Name: optimize_entanglement_fidelity
 * Description: optimizing spin-photon entanglement 
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: October 6, 2025
 * Version: 2.0
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from scipy import integrate
from tin_vacancy_characteristics import SnV
from silicon_vacancy_characteristics import SiV
from atomic_factor_g4v import atomic_factor
from photonic_decay_rates import photonic_decay_rates
from scipy.optimize import shgo
import scipy

def objective(x,w15,w26,ga15,ga26,g15,g26,ga1,flag):
    
    '''
    * Function: objective
    
    * Purpose: objective function model for spin-photon entanglement optimization
    
    * Parameters:
        
        x (array of floats): cavity loss rate, cavity mode detuning, central frequency detuning
        w15 (float): atomaic transition frequency 15
        w26 (float): atomic transition frequency 26
        ga15 (float): atomic decay rate 15
        ga26 (float): atomic decay rate 26
        g15 (complex): coupling strength 15
        g26 (complex): coupling strength 26
        ga1 (float): bandwidth of incoming photon 
    
    * Returns:
        
        (float): spin-photon entanglement infidelity
    '''
    C,d,dw0=x
    
    k=np.abs(g15)**2/(4*C*ga15)
    
    w0=w15-dw0
    wc=w15-d
    kl=k
    rdown=lambda w:1-2*kl*(1j*(w-w15)+ga15)/((1j*(w-wc)+k)*(1j*(w-w15)+ga15)+np.abs(g15)**2)
    rup=lambda w:1-2*kl*(1j*(w-w26)+ga26)/((1j*(w-wc)+k)*(1j*(w-w26)+ga26)+np.abs(g26)**2)
    
    # photon spectra of the early and late time bin photon
    SX_=lambda w: 1/(w**2+0.25*ga1**2)
    
    
    # normalization constants
    
    bound=int(1e+4)*ga1
    NX=(integrate.quad(lambda w: SX_(w),-bound,bound))[0]**(-1/2)
    
    SX=lambda w: NX**2*SX_(w)
    
    alpha=1/np.sqrt(2)
    beta=alpha
    alpha_c=np.conjugate(alpha)
    beta_c=np.conjugate(beta)
    
    #bound=int(1e+4)*ga1
    func1=lambda w: SX(w-w0)*np.abs(rdown(w))**2
    func2_re=lambda w: SX(w-w0)*np.real(rdown(w)*np.conjugate(rup(w)))
    func2_im=lambda w: SX(w-w0)*np.imag(rdown(w)*np.conjugate(rup(w)))
    func3=lambda w: SX(w-w0)*np.abs(rup(w))**2
    
    I1,_=integrate.quad(func1,w0-bound,w0+bound)
    I2_re,_=integrate.quad(func2_re,w0-bound,w0+bound)
    I2_im,_=integrate.quad(func2_im,w0-bound,w0+bound)
    I3,_=integrate.quad(func3,w0-bound,w0+bound)
    I2=I2_re+1j*I2_im
    I2_c=np.conjugate(I2)
    
    A=np.abs(alpha/2+beta/2)**2*I1
    B=1/4*alpha_c*(alpha+beta)*I1+1/4*beta_c*(alpha+beta)*I2
    B_c=np.conjugate(B)
    C=1/4*np.abs(alpha)**2+1/4*alpha_c*beta*I2_c+1/4*alpha*beta_c*I2+1/4*np.abs(beta)**2*I3
    
    
    rho_p=np.array([[A,B],
                   [B_c,C]],dtype=complex)
    
    beta=-beta
    beta_c=-beta_c
    
    A=np.abs(alpha/2+beta/2)**2*I1
    B=1/4*alpha_c*(alpha+beta)*I1+1/4*beta_c*(alpha+beta)*I2
    B_c=np.conjugate(B)
    C=1/4*np.abs(alpha)**2+1/4*alpha_c*beta*I2_c+1/4*alpha*beta_c*I2+1/4*np.abs(beta)**2*I3
    
    
    rho_m=np.array([[A,B],
                   [B_c,C]],dtype=complex)
    
    sz=np.array([[-1,0],
                 [0,1]])
    
    H=1/np.sqrt(2)*np.array([[1,-1],[1,1]])
    rho_p=np.dot(H,np.dot(rho_p,H.T))
    rho_m=np.dot(H,np.dot(rho_m,H.T))
    
    rho=rho_p+np.dot(sz,np.dot(rho_m,sz))
    
    eta=np.trace(rho)
    rho=rho/eta
    
    sigma=np.array([[0.5,0.5],
                      [0.5,0.5]])
    
    F=mixed_state_fidelity(sigma,rho)
    
    if flag=='optimize':
        return 1-np.abs(F)
    else:
        return 1-np.abs(F),eta

def mixed_state_fidelity(sigma,rho):
    
    '''
    * Function: mixed_state_fidelity
    
    * Purpose: fidelity calculation
    
    * Parameters:
        
       sigma (array of complex): target state
       rho (array of complex): simulated state
    
    * Returns:
        
        (float): spin-photon entanglement infidelity
    '''
    
    A=np.dot(scipy.linalg.sqrtm(rho),np.dot(sigma,scipy.linalg.sqrtm(rho)))
    X=scipy.linalg.sqrtm(A)
    F=np.trace(X)**2
    
    return F


def doe(center,a1,a2,angle,B,Ex,eps_xy):
    
    '''
    * Function: doe
    
    * Purpose: design of experiment
    
    * Parameters:
        
        a1 (float): mode orientation polar angle
        a2 (float): mode orientation azimuthal angle
        angle (float): magnetic field orientation
        B (float): magnetic field strength
        Ex (float): axial strain
        eps_xy (float): shear strain
    
    * Returns:
        
        g15 (complex): coupling strength 15
        g16 (complex): coupling strength 16
        g25 (complex): coupling strength 25
        g26 (complex): coupling strength 26
        ga15 (float): atomic decay rate 15
        ga16 (float): atomic decay rate 16
        ga25 (float): atomic decay rate 25
        ga26 (float): atomic decay rate 26
        w15 (float): atomaic transition frequency 15
        splitting (float): spin splitting
        e6 (float): atomic transition frequency 16
        w26 (float): atomic transition frequency 26
        
    '''
    
    THz=1


    [THz,A,kg,m]=[1,1,1,1]
    ps=1/THz
    s=1e+12*ps
    J=kg*m**2/(s**2)
    V=J/(A*s)
    
    if center=='SiV':
        
        dg=2*np.pi*1.3*1e+3
        de=2*np.pi*1.8*1e+3
        
    elif center=='SnV':
        
        dg=2*np.pi*0.787*1e+3
        de=2*np.pi*0.956*1e+3

    alpha_g,alpha_u,beta_g,beta_u=[dg*Ex,de*Ex,2*dg*eps_xy,2*de*eps_xy]

    n=2.417
    eps_0=8.854*1e-12*A*s/(V*m)
    eps_r=5.7
    h=6.626*1e-34*J*s


    # hamiltonian, spin splitting and atomic displacement
    if center=='SnV':
        H_0,mu_1,mu_2,mu_3,S=SnV(THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u,B,angle).hamiltonian()
        H1,H2,H3=SnV(0,0,0,0,0,0,0,0,0,0).interaction()
    elif center=='SiV':
        H_0,mu_1,mu_2,mu_3,S=SiV(THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u,B,angle).hamiltonian()
        H1,H2,H3=SiV(0,0,0,0,0,0,0,0,0,0).interaction()

    splitting=H_0[1,1]-H_0[0,0]

    w26=H_0[5,5]-H_0[1,1]
    evals=np.diag(H_0)
    a=atomic_factor(THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u,center).get_atomic_a()

    # photonic decay rates in the SnV and the relevant rate ga51
    gas=photonic_decay_rates(THz,evals,S,a).rates(H1,H2,H3)
    gas=np.array(gas)
    gas=np.reshape(gas,(4,4))

    ga15=gas[0,0]
    ga25=gas[1,0]
    ga16=gas[0,1]
    ga26=gas[1,1]
    

    # atomic transition frequency and spin splitting
    w0_down=H_0[4,4]-H_0[0,0]
    splitting=H_0[1,1]-H_0[0,0]
    e6=H_0[5,5]-H_0[0,0]

    e=1.602*1e-7*A*(1/THz)

    G=np.array([[1/np.sqrt(6),-2/np.sqrt(6),1/np.sqrt(6)],
           [1/np.sqrt(2),0,-1/np.sqrt(2)],
           [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]])
    
    
    eps=np.array([np.cos(a1)*np.sin(a2),np.sin(a1)*np.sin(a2),np.cos(a2)])
    eps=np.dot(G.T,eps)
    d_el=a*e*(eps[0]*mu_1+eps[1]*mu_2+eps[2]*mu_3)


    wc=H_0[4,4]
    c=3*1e+8*m/s
    hbar=h/(2*np.pi)
    la_ph=2*np.pi*c/wc
    V_c=1.8*la_ph**3/(2*n**3)
    fac=np.sqrt(wc/(2*hbar*eps_0*eps_r*V_c))
    
    g15=1j*fac*d_el[0,4]
    g16=1j*fac*d_el[0,5]
    
    g25=1j*fac*d_el[1,4]
    
    g26=1j*fac*d_el[1,5]
    
    w15=w0_down
    
    return g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,splitting,e6,w26

def get_params(center,B,angle,ga,C_max,Ex,eps_xy):
    
    '''
    * Function: get_params
    
    * Purpose: solve optimization problem
    
    * Parameters:
        
        B (float): magnetic field strength
        angle (float): magnetic field orientation
        ga (float): bandwidth of incoming photon
        Ex (float): axial strain
        eps_xy (float): shear strain
    
    * Returns:
        
        (array of floats): cavity loss rate, cavity mode detuning, central frequency detuning
        (float): spin-photon entanglement fidelity
        (float): spin-photon entanglement efficiency
        (float): cooperativity of transition 1A
        (float): cooperativity of transition 2B
        (float): cooperativity of transition 2A
        (float): cooperativity of transition 1B
        
    '''
    
    g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,splitting,e6,w26=doe(center,0,0,angle,B,Ex,eps_xy)
    
    ga15=ga15/2
    ga16=ga16/2
    ga25=ga25/2
    ga26=ga26/2
    
    f=lambda x: objective(x,w15,w26,ga15,ga26,g15,g26,ga,'optimize')
    res=shgo(f,bounds=[(0.1,C_max),(-0.1,0.1),(-0.1,0.1)],sampling_method='sobol',iters=4,n=64)
    
    y=res.x
    I,eta=objective(y,w15,w26,ga15,ga26,g15,g26,ga,'evaluate')
    y=(np.abs(g15)**2/(4*y[0]*ga15),y[1],y[2])
    
    C1A=np.abs(g15)**2/(y[0]*ga15)
    C2B=np.abs(g26)**2/(y[0]*ga26)
    C2A=np.abs(g25)**2/(y[0]*ga25)
    C1B=np.abs(g16)**2/(y[0]*ga16)
    
    
    return y,I,eta,C1A,C2B,C2A,C1B