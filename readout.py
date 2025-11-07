"""
 * Module Name: readout
 * Description: simulate readout back to the photon
 * Author: Yannick Strocka
 * Created On: October 30, 2025
 * Last Modified: October 30, 2025
 * Version: 2.0
"""

import os

# Muss VOR dem Import von numpy/scipy gesetzt werden!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from silicon_vacancy_characteristics import SiV
from tin_vacancy_characteristics import SnV
from atomic_factor_g4v import atomic_factor
from photonic_decay_rates import photonic_decay_rates
from scipy.integrate import odeint
import warnings
from scipy.integrate import simpson
warnings.filterwarnings("ignore")


def f(y,t,d,wc,k,g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,e2,e6,ga_d,e0,flag):
    
    '''
    * Function: f
    
    * Purpose: model Heisenberg-Langevin equations dydt=f(y,t,params) and solve numerically
    
    * Parameters:
        
        y (array of floats): y=(a,s15,s25,s16,s26,s11,s22,s55,s66)
        t (float): time
        d (float): incoming mode central frequency
        wc (float): cavity mode central frequency
        k (float): cavity loss rate
        g15 (complex): coupling strength 15
        g16 (complex): coupling strength 16
        g25 (complex): coupling strength 25
        g26 (complex): coupling strength 26
        ga15 (float): decay rate 15
        ga16 (float): decay rate 16
        ga25 (float): decay rate 26
        w15 (float): transition
        e2 (float): spin splitting
        e6 (float): transition frequency 16
        ga_d (float): bandwidth of incoming photon
        e0 (float): field amplitude of incoming photon
        flag (string): cross or ideal 
        
    * Returns:
        
        (array of floats): right hand side of ordinary differential equation
        
    '''
    
    if flag=='ideal':
        g25=0
        g16=0
    
    wd=d
    e5=w15
    dc=e5-wc
    ain=e0*np.exp((1j*wd-ga_d/2)*t)
    
    a=y[0]+1j*y[1]
    s15=y[2]+1j*y[3]
    s25=y[4]+1j*y[5]
    s16=y[6]+1j*y[7]
    s26=y[8]+1j*y[9]
    s11=y[10]+1j*y[11]
    s22=y[12]+1j*y[13]
    s55=y[14]+1j*y[15]
    s66=y[16]+1j*y[17]
    
    ac=np.conjugate(a)
    s15c=np.conjugate(s15)
    s25c=np.conjugate(s25)
    s16c=np.conjugate(s16)
    s26c=np.conjugate(s26)
    
    g15c=np.conjugate(g15)
    g25c=np.conjugate(g25)
    g16c=np.conjugate(g16)
    g26c=np.conjugate(g26)
    
    
    f_t=np.exp(1j*e2*t)
    f_tc=np.conjugate(f_t)
    
    expr1=-1j*(g15*s15+f_t*g25*s25+f_tc*g16*s16+g26*s26)-k*a+np.sqrt(2*k)*ain
    
    expr2=-1j*(-dc*s15+f_tc*g25c*s15*s25c*a-f_t*g16c*s16c*s15*a+g15c*a*(s11-s55))-0.5*(ga15+ga25)*s15
    
    expr3=-1j*(-dc*s25+g15c*s25*s15c*a-g26c*s26c*s25*a+f_tc*g25c*a*(s22-s55))-0.5*(ga15+ga25)*s25
    
    expr4=-1j*((e2-e6+e5-dc)*s16+g15c*s15c*s16*a+g26c*s16*s26c*a+f_t*g16c*a*(s11-s66))-0.5*(ga16+ga26)*s16
    
    expr5=-1j*((e2-e6+e5-dc)*s26-f_tc*g25c*s25c*s26*a+f_t*g16c*s26*s16c*a+g26c*a*(s22-s66))-0.5*(ga16+ga26)*s26
    
    expr6=-1j*(g15*s15*ac-g15c*s15c*a+f_tc*g16*s16*ac-f_t*g16c*s16c*a)+ga15*s55+ga16*s66
    
    expr7=-1j*(f_t*g25*s25*ac-f_tc*g25c*s25c*a+g26*s26*ac-g26c*s26c*a)+ga25*s55+ga26*s66
    
    expr8=-1j*(-g15*s15*ac+g15c*s15c*a-f_t*g25*s25*ac+f_tc*g25c*s25c*a)-(ga15+ga25)*s55
        
    expr9=-expr6-expr7-expr8
    
    f1=np.real(expr1)
    f2=np.imag(expr1)
    f3=np.real(expr2)
    f4=np.imag(expr2)
    f5=np.real(expr3)
    f6=np.imag(expr3)
    f7=np.real(expr4)
    f8=np.imag(expr4)
    f9=np.real(expr5)
    f10=np.imag(expr5)
    f11=np.real(expr6)
    f12=np.imag(expr6)
    f13=np.real(expr7)
    f14=np.imag(expr7)
    f15=np.real(expr8)
    f16=np.imag(expr8)
    f17=np.real(expr9)
    f18=np.imag(expr9)
    
    
    rhs=np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,
                 f11,f12,f13,f14,f15,f16,f17,f18])
    
    return rhs

def doe(center,a1,a2,k,delta,angle,B,Ex,eps_xy):
    
    '''
    * Function: doe
    
    * Purpose: design of experiment
    
    * Parameters:
        
        a1 (float): mode orientation polar angle
        a2 (float): mode orientation azimuthal angle
        k (float): cavity loss rate
        delta (float): mode detuning from atomic transition frequency
        angle (float): magnetic field orientation
        B (float): magnetic field strength
        Ex (float): axial strain
        eps_xy (float): shear strain
    
    * Returns:
        
        k (float): cavity loss rate
        wc (float): cavity mode central frequency
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

    wc=H_0[4,4]-delta
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

    
    return k,wc,g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,splitting,e6


def get_output_signal(init,dw0,wc,k,g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,e2,e6,ga,e0,dt,T,flag):
    
    '''
    
    * Function: get_output_signal
    
    * Purpose: evaluate output mode in time domain including crosstalk
    
    * Parameters: 
        
        init (float): initialize spin in 0 or 1
        dw0 (float): central frequency detuning
        wc (float): cavity mode central frequency
        k (float): cavity loss rate
        g15 (complex): coupling strength 15
        g16 (complex): coupling strength 16
        g25 (complex): coupling strength 25
        g26 (complex): coupling strength 26
        ga15 (float): atomic decay rate 15
        ga16 (float): atomic decay rate 16
        ga25 (float): atomic decay rate 25
        ga26 (float): atomic decay rate 26
        w15 (float): transition frequency 15
        e2 (float): spin splitting
        e6 (float): transition frequency 16
        ga (float): bandwidth of incoming photon
        e0 (float): amplitude of incoming mode
        dt (float): time step size
        T (float): integration time
        flag (string): 'ideal' or 'cross'
        
    * Returns:
        
        (array of complex): output mode in time domain
    
    '''
    
    if init==0:
        y_0=np.zeros((18,))
        y_0[10]=1

    if init==1:
        y_0=np.zeros((18,))
        y_0[12]=1
        
    N=int(T/dt)
    t=np.linspace(0,T,N)
    
    w0=w15-dw0-wc
    data=odeint(f, y_0, t, args=(w0,wc,k,g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,e2,e6,ga,e0,flag),rtol = 10**(-10), atol = 10**(-12))
    y=np.array([data[i][0] +1j*data[i][1] for i in range(len(data))])
    ain=e0*np.exp((1j*w0-ga/2)*t)
    y1=np.sqrt(2*k)*y-ain
    
    return y1

def get_integrals(y1,y2,T):
    
    t=np.linspace(0,T,len(y1))
    
    I1=simpson(np.abs(y1)**2,t)
    I2_re=simpson(np.real(y1*np.conjugate(y2)),t)
    I2_im=simpson(np.imag(y1*np.conjugate(y2)),t)
    I2=I2_re+1j*I2_im
    I3=simpson(np.abs(y2)**2,t)
    
    return I1,I2,I3


def get_state(I1X,I2X,I3X,m,n,Las,flag,F_aux):
    
    '''
    * Function: get_state
    
    * Purpose: evaluate saved state including all imperfections
    
    * Parameters:
        
        Fph (float): photon generation fidelity
        I1X (float): integral 1
        I2X (complex): integral 2
        I3X (float): integral 3
        aa (float): entry 00 of photonic state
        bb (float): entry 01 of photonic state
        cc (float): entry 10 of photonic state
        dd (float): entry 11 of photonic state
        La (array of complex): propagated state |1> subject to pi/2 pulse
        flag (string): measurement + or -
    
    * Returns:
        
        (array of complex): photonic state after readout 
    
    '''
    
    rho_ph=np.array([[0.5,0.5],[0.5,0.5]])
    eps=2*(1-F_aux)
    rho_ph=(1-eps)*rho_ph+eps*np.eye(2)/2
        
    La=Las[2*m+n]
    IsX=[I1X,I2X,np.conjugate(I2X),I3X]
    rho=np.zeros((2,2),dtype=complex)
    
    if flag=='down':
        i=0
        rho[0,0]=La[i,i]*rho_ph[0,0]*IsX[2*m+n]
        rho[0,1]=La[i,i]*rho_ph[0,1]*IsX[2*m+i]
        rho[1,0]=La[i,i]*rho_ph[1,0]*IsX[2*i+n]
        rho[1,1]=La[i,i]*rho_ph[1,1]*IsX[3*i]
    
    elif flag=='up':
        i=1
        rho[0,0]=La[i,i]*rho_ph[0,0]*IsX[2*m+n]
        rho[0,1]=La[i,i]*rho_ph[0,1]*IsX[2*m+i]
        rho[1,0]=La[i,i]*rho_ph[1,0]*IsX[2*i+n]
        rho[1,1]=La[i,i]*rho_ph[1,1]*IsX[3*i]
    
    return rho


def readout(center,mm,F_aux,B,angle,k,delta,dw0,gaX,Las,Ex,eps_xy):
    
    '''
    * Function: readout
    
    * Purpose: simulate photonic states after reflection scheme for full basis of spin qubit qubit
    
    * Parameters:
        
        center (string): SiV or SnV
        mm (float): measurement + or -
        F_aux (float): fidelity of photon from auxiliary photon source
        B (float): magnetic field strength
        angle (float): magnetic field orientation
        k (float): cavity loss rate
        delta (float): cavity mode detuning
        dw0 (float): central frequency detuning
        gaX (float): bandwidth of incoming photon
        La (array or complex): propagated spin subject to pi/2 pulse
        Ex (float): axial strain
        eps_xy (float): shear strain
        
    * Returns:
        
        (list of arrays of complex): readout states for full basis of spin qubits
    
    '''
    
    flag='cross'
    a1=0
    a2=0
    k,wc,g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,e2,e6=doe(center,a1,a2,k,delta,angle,B,Ex,eps_xy)


    dt=1e+0
    T=1e+6
    init=0
    e0=1e-4
    y1=get_output_signal(init,dw0,wc,k,g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,e2,e6,gaX,e0,dt,T,flag)
    init=1
    y2=get_output_signal(init,dw0,wc,k,g15,g16,g25,g26,ga15,ga16,ga25,ga26,w15,e2,e6,gaX,e0,dt,T,flag)

    I1,I2,I3=get_integrals(y1,y2,T) 
    
    I1X=I1*gaX/(e0**2)
    I2X=I2*gaX/(e0**2)
    I3X=I3*gaX/(e0**2)
    
    state1=get_state(I1X,I2X,I3X,0,0,Las,mm,F_aux)
    state2=get_state(I1X,I2X,I3X,0,1,Las,mm,F_aux)
    state3=get_state(I1X,I2X,I3X,1,0,Las,mm,F_aux)
    state4=get_state(I1X,I2X,I3X,1,1,Las,mm,F_aux)
    states=[state1,state2,state3,state4]
    
    return states
