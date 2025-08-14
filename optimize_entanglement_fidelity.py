"""
 * Module Name: optimize_entanglement_fidelity
 * Description: optimizing spin-photon entanglement 
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import numpy as np
from scipy import integrate
from tin_vacancy_characteristics import SnV
from atomic_factor import atomic_factor
from photonic_decay_rates import photonic_decay_rates
from scipy.optimize import shgo
import scipy
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def main2(x,w15,w26,ga15,ga26,g15,g26,ga1):
    
    '''
    * Function: main2
    
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
    k,d,dw0=x
      
    w0=w15+dw0
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
    
    # rho_p
    func1=lambda w: 0.5*np.abs(beta)**2/4*SX(w-w0)*np.abs(rdown(w)-rup(w))**2
    func2=lambda w: 0.5*SX(w-w0)*np.abs(alpha*rdown(w)+beta/2*(rdown(w)+rup(w)))**2
    func3_re=lambda w: np.real(0.5*beta/2*SX(w-w0)*(rdown(w)-rup(w))*(np.conjugate(alpha)*np.conjugate(rdown(w))
                                                                     +np.conjugate(beta)/2*(np.conjugate(rdown(w))
                                                                                            +np.conjugate(rup(w)))))
    func3_im=lambda w: np.imag(0.5*beta/2*SX(w-w0)*(rdown(w)-rup(w))*(np.conjugate(alpha)*np.conjugate(rdown(w))
                                                                     +np.conjugate(beta)/2*(np.conjugate(rdown(w))
                                                                                            +np.conjugate(rup(w)))))
    
    I1,_=integrate.quad(func1,w0-bound,w0+bound)
    I2,_=integrate.quad(func2,w0-bound,w0+bound)
    cc,_=integrate.quad(func3_re,w0-bound,w0+bound)
    dd,_=integrate.quad(func3_im,w0-bound,w0+bound)
    
    I3=cc+1j*dd
    
    rho_p=np.array([[I1,I3],
                  [np.conjugate(I3),I2]])
    
    # rho_m
    func2=lambda w: 0.5*SX(w-w0)*np.abs(alpha*rdown(w)-beta/2*(rdown(w)+rup(w)))**2
    func3_re=lambda w: -np.real(0.5*beta/2*SX(w-w0)*(rdown(w)-rup(w))*(np.conjugate(alpha)*np.conjugate(rdown(w))
                                                                     -np.conjugate(beta)/2*(np.conjugate(rdown(w))+np.conjugate(rup(w)))))
    func3_im=lambda w: -np.imag(0.5*beta/2*SX(w-w0)*(rdown(w)-rup(w))*(np.conjugate(alpha)*np.conjugate(rdown(w))
                                                                     -np.conjugate(beta)/2*(np.conjugate(rdown(w))+np.conjugate(rup(w)))))
    I2,_=integrate.quad(func2,w0-bound,w0+bound)
    cc,_=integrate.quad(func3_re,w0-bound,w0+bound)
    dd,_=integrate.quad(func3_im,w0-bound,w0+bound)
    
    I3=cc+1j*dd
    rho_m=np.array([[I1,I3],
                [np.conjugate(I3),I2]])
    
    sz=np.array([[-1,0],
                 [0,1]])
    
    rho=rho_p+np.dot(sz,np.dot(rho_m,sz))
    
    rho=rho/np.trace(rho)
    
    sigma=np.array([[0.5,0.5],
                      [0.5,0.5]])
    
    F=mixed_state_fidelity(sigma,rho)
    
    return 1-np.abs(F)

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

def get_sol(x,w15,w26,ga15,ga26,g15,g26,ga1):
    
    '''
    * Function: get_sol
    
    * Purpose: evaluation spin-photon entanglement fidelity and efficiency
    
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
        
        (float): spin-photon entanglement fidelity
        (float): spin-photon entanglement efficiency
    '''
    k,d,dw0=x
     
    w0=w15+dw0
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
    func1=lambda w: 0.5*np.abs(beta)**2/4*SX(w-w0)*np.abs(rdown(w)-rup(w))**2
    func2=lambda w: 0.5*SX(w-w0)*np.abs(alpha*rdown(w)+beta/2*(rdown(w)+rup(w)))**2
    func3_re=lambda w: np.real(0.5*beta/2*SX(w-w0)*(rdown(w)-rup(w))*(np.conjugate(alpha)*np.conjugate(rdown(w))
                                                                     +np.conjugate(beta)/2*(np.conjugate(rdown(w))+np.conjugate(rup(w)))))
    func3_im=lambda w: np.imag(0.5*beta/2*SX(w-w0)*(rdown(w)-rup(w))*(np.conjugate(alpha)*np.conjugate(rdown(w))
                                                                     +np.conjugate(beta)/2*(np.conjugate(rdown(w))+np.conjugate(rup(w)))))
    
    I1,_=integrate.quad(func1,w0-bound,w0+bound)
    I2,_=integrate.quad(func2,w0-bound,w0+bound)
    cc,_=integrate.quad(func3_re,w0-bound,w0+bound)
    dd,_=integrate.quad(func3_im,w0-bound,w0+bound)
    
    I3=cc+1j*dd
    
    rho_p=np.array([[I1,I3],
                  [np.conjugate(I3),I2]])
    
    # rho_m
    func2=lambda w: 0.5*SX(w-w0)*np.abs(alpha*rdown(w)-beta/2*(rdown(w)+rup(w)))**2
    func3_re=lambda w: -np.real(0.5*beta/2*SX(w-w0)*(rdown(w)-rup(w))*(np.conjugate(alpha)*np.conjugate(rdown(w))
                                                                     -np.conjugate(beta)/2*(np.conjugate(rdown(w))+np.conjugate(rup(w)))))
    func3_im=lambda w: -np.imag(0.5*beta/2*SX(w-w0)*(rdown(w)-rup(w))*(np.conjugate(alpha)*np.conjugate(rdown(w))
                                                                     -np.conjugate(beta)/2*(np.conjugate(rdown(w))+np.conjugate(rup(w)))))
    I2,_=integrate.quad(func2,w0-bound,w0+bound)
    cc,_=integrate.quad(func3_re,w0-bound,w0+bound)
    dd,_=integrate.quad(func3_im,w0-bound,w0+bound)
    
    I3=cc+1j*dd
    rho_m=np.array([[I1,I3],
                [np.conjugate(I3),I2]])
    
    sz=np.array([[-1,0],
                 [0,1]])
    
    rho=rho_p+np.dot(sz,np.dot(rho_m,sz))
    
    eta=np.trace(rho)
    rho=rho/eta
    
    sigma=np.array([[0.5,0.5],
                      [0.5,0.5]])
    
    F=mixed_state_fidelity(sigma,rho)
    
    return np.abs(F),np.abs(eta)


def doe(a1,a2,angle,B,Ex,eps_xy):
    
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

    dg=2*np.pi*0.787*1e+3
    de=2*np.pi*0.956*1e+3

    alpha_g,alpha_u,beta_g,beta_u=[dg*Ex,de*Ex,2*dg*eps_xy,2*de*eps_xy]

    n=2.417
    eps_0=8.854*1e-12*A*s/(V*m)
    eps_r=5.7
    h=6.626*1e-34*J*s


    # hamiltonian, spin splitting and atomic displacement
    H_0,mu_1,mu_2,mu_3,S=SnV(THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u,B,angle).hamiltonian()
    H1,H2,H3=SnV(0,0,0,0,0,0,0,0,0,0).interaction()

    splitting=H_0[1,1]-H_0[0,0]

    w26=H_0[5,5]-H_0[1,1]
    evals=np.diag(H_0)
    a=atomic_factor(THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u).get_atomic_a()

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

def get_params(B,angle,ga,Ex,eps_xy):
    
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

    g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,splitting,e6,w26=doe(0,0,angle,B,Ex,eps_xy)
    f=lambda x: main2(x,w15,w26,ga15,ga26,g15,g26,ga)
    res=shgo(f,bounds=[(0.001,0.1),(-0.1,0.1),(-0.1,0.1)],sampling_method='sobol',iters=2,n=64)
    
    y=res.x
    F,eta=get_sol(y,w15,w26,ga15,ga26,g15,g26,ga)
    
    C1A=np.abs(g15)**2/(y[0]*ga15)
    C2B=np.abs(g26)**2/(y[0]*ga26)
    C2A=np.abs(g25)**2/(y[0]*ga25)
    C1B=np.abs(g16)**2/(y[0]*ga16)
    
    
    return y,F,eta,C1A,C2B,C2A,C1B