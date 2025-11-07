"""
 * Module Name: mw_control
 * Description: microwave control of the SnV for the spin pi/2 rotation
 * Author: Mohamed Belhassen
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import numpy as np
from scipy.constants import hbar, e, m_e, Boltzmann
from scipy import integrate
from scipy.integrate import ode


pi = np.pi

# Bohr magneton
mu_B = .5 * e * hbar / m_e

# Gyromagnetic ratios in GHz/T
Gamma_L =(10**-9)*mu_B/(2*pi*hbar)
Gamma_S = Gamma_L

# Diamond density kg / m^3
density = 3501


# Diamond a0 m
a0 = 0.356e-9

# Stiffness Tensor kg Hz^2 / m
C11 = 1079.26e9
C12 = 126.73e9
C44 = 578.16e9


# Ground State SnV
Lambda = 815/2 # Spin-Orbital coupling GHz
f = 0.15 # Zeeman orbital coupling 
d = 0.787e6 # GHz/Strain
fs = -0.562e6 # GHz/Strain


Dx = np.array([[d,0,fs/2],
                   [0,-d,0],
                   [fs/2,0,0]])*1e9*2*pi # Hz Rad
Dy = np.array([[0,-d,0],
               [-d,0,fs/2],
               [0,fs/2,0]])*1e9*2*pi

Id = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0], 
               [0, 0, 0, 1]])

# To be computed only once
Factor = hbar * 1e9 / Boltzmann

def Hso(Lambda):
    return np.array([[0, 0, - Lambda * 1j, 0],
                     [0, 0, 0, Lambda * 1j],
                     [Lambda * 1j, 0, 0, 0],
                     [0, -Lambda * 1j, 0, 0]]) * 2 * pi # Spin-Orbital interaction Rad.Ghz
def Strain(Ex,Epsilon_xy):
    # Strain interaction in Rad.Ghz
    return np.array([[d*Ex, 0, 2*d*Epsilon_xy, 0],
                     [0, d*Ex, 0, 2*d*Epsilon_xy],
                     [2*d*Epsilon_xy, 0, -d*Ex, 0], 
                     [0, 2*d*Epsilon_xy, 0, -d*Ex]]) * 2 * pi # Strain interaction in Rad.Ghz
def Bpar(Bz):
    # Zeeman Effect with Bz in Rad.Ghz
    return np.array([[Gamma_S * Bz, 0, f * Gamma_L * Bz * 1j, 0], 
                     [0, - Gamma_S * Bz, 0,  f * Gamma_L * Bz * 1j], 
                     [- f * Gamma_L * Bz * 1j, 0, Gamma_S * Bz, 0], 
                     [0, - f * Gamma_L * Bz * 1j, 0, - Gamma_S * Bz]]) * 2 * pi 
def Bper(Bx,By):
    # Zeeman Effect with Bx and By in Rad.Ghz
    return np.array([[0, Gamma_S * Bx - Gamma_S * By * 1j, 0, 0], 
                     [Gamma_S * Bx + Gamma_S * By * 1j, 0, 0, 0], 
                     [0, 0, 0, Gamma_S * Bx - Gamma_S * By * 1j], 
                     [0, 0, Gamma_S * Bx + Gamma_S * By * 1j, 0]]) * 2 * pi 

# Diagonalization of a matrix with ordering from lowest to highest eigenvalues
def Diag(H_tot):
    # Diagonalise H_tot
    EigVa, P = np.linalg.eig(H_tot)

    sorted_indexes = np.argsort(EigVa)
    
    EigVa = EigVa[sorted_indexes]
    P = P[:,sorted_indexes]
    P_dag = np.transpose(np.conjugate(P))
    return EigVa,P,P_dag

# Compute Rabi oscilations in GHz
def Rabi(B,b,x):
       
    H_tot= Hso(Lambda) + Bpar(B*np.cos(x[0])) + Bper(B*np.sin(x[0])*np.cos(x[1]),B*np.sin(x[0])*np.sin(x[1])) + Strain(x[4],x[5])
    
    _, P, P_dag = Diag(H_tot)
    
    H_ac = P_dag @ (Bpar(b*np.cos(x[2])) + Bper(b*np.sin(x[2])*np.cos(x[3]),b*np.sin(x[2])*np.sin(x[3]))) @ P
    
    return np.abs(H_ac[0,1])/(2*pi)

# Transforming Phonon coupling matrix to eigenvector basis
def h_x(P,P_dag):
    return P_dag @ np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, -1, 0], 
                             [0, 0, 0, -1]]) @ P # 2pi in Dx
def h_y(P,P_dag):
    return P_dag @ np.array([[0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [1, 0, 0, 0], 
                             [0, 1, 0, 0]]) @ P # 2pi in Dx

# Extracting wavemodes
def Wavemodes(k, Stiffness):
    
    Matrix = np.einsum("ijkl, j, k -> il", Stiffness,k,k)
    
    EigVa, q = np.linalg.eig(Matrix/density)

    sorted_indexes = np.argsort(EigVa)
    
    EigVa = EigVa[sorted_indexes]
    q = q[:,sorted_indexes]
    return EigVa,q

# Bose-Einstein distribution
def n_average(w, T):
    if w == 0:
        return 0
    elif T == 0:
        return 0
    else:
        exp_term = np.exp(Factor * w / T)
        return 1 / (exp_term - 1)

# Create a zero matrix with element 1 in raw i, column j
def sigma(i, j):
    matrix = np.zeros((4, 4))
  
    # Set the element at position [i, j] to 1
    matrix[i, j] = 1
    
    return matrix

# Frame rotation
def U(E, t):
        return np.diag(np.array([np.exp(1j*E[0]*t),np.exp(1j*E[1]*t),np.exp(1j*E[2]*t),np.exp(1j*E[3]*t)]))
    
#
C_diamond = np.zeros((3,3,3,3))
C_diamond[0,0,0,0] = C11
C_diamond[1,1,1,1] = C11
C_diamond[2,2,2,2] = C11

C_diamond[0,0,1,1] = C12
C_diamond[1,1,0,0] = C12

C_diamond[1,1,2,2] = C12
C_diamond[2,2,1,1] = C12

C_diamond[0,0,2,2] = C12
C_diamond[2,2,0,0] = C12

C_diamond[1,2,1,2] = C44
C_diamond[2,1,1,2] = C44
C_diamond[1,2,2,1] = C44
C_diamond[2,1,2,1] = C44

C_diamond[2,0,2,0] = C44
C_diamond[0,2,2,0] = C44
C_diamond[2,0,0,2] = C44
C_diamond[0,2,0,2] = C44

C_diamond[0,1,0,1] = C44
C_diamond[1,0,0,1] = C44
C_diamond[0,1,1,0] = C44
C_diamond[1,0,1,0] = C44

# Rotating from the diamond lattice coordinate to the defect coordinates 
# Z rotation
theta = -3*pi/4
Rotation = np.array([[np.cos(theta),-np.sin(theta),0],
                   [np.sin(theta),np.cos(theta),0],
                   [0,0,1]])
C_d = np.einsum('ab,cd,ef,gh,...bdfh->...aceg', Rotation, Rotation, Rotation, Rotation, C_diamond*1e-9)


# Y rotation
theta = -np.arccos(1/np.sqrt(3))
Rotation = np.array([[np.cos(theta),0,np.sin(theta)],
                    [0,1,0],
                    [-np.sin(theta),0,np.cos(theta)]])
C_d = np.einsum('ab,cd,ef,gh,...bdfh->...aceg', Rotation, Rotation, Rotation, Rotation, C_d)*1e9

# Integrated function to compute Xi_e_xx and Xi_e_yy and Xi_e_xy

def func_x(theta,phi):
        
    k = np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])
    
    EiV , qs = Wavemodes(k,C_d)  
   
    val=0
    for i in range(3):
        q = qs[:,i]
        c = np.sqrt(EiV[i])
        val = val + np.einsum("ij,i,j->",Dx,k,q)**2/(c**5)

    return val*np.sin(theta)

def func_y(theta,phi):
        
    k = np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])
    
    EiV , qs = Wavemodes(k,C_d)  
   
    val=0
    for i in range(3):
        q = qs[:,i]
        c = np.sqrt(EiV[i])
        val = val + np.einsum("ij,i,j->",Dy,k,q)**2/(c**5)

    return val*np.sin(theta)

def func_xy(theta,phi):
        
    k = np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])
    
    EiV , qs = Wavemodes(k,C_d)  
   
    val=0
    for i in range(3):
        q = qs[:,i]
        c = np.sqrt(EiV[i])
        val = val + np.einsum("ij,i,j->",Dy,k,q)*np.einsum("ij,i,j->",Dx,k,q)/(c**5)

    return val*np.sin(theta) * hbar/(16*np.pi**3*density)

def Chi():
    cx, errx = integrate.nquad(func_x, [[0,pi],[0,2*pi]])
    cy, erry = integrate.nquad(func_y, [[0,pi],[0,2*pi]])
    
    return np.array([cx, errx, cy, erry])*hbar/(16*np.pi**3*density) * 1e18 # GHz

# Check that Xi_e_xy is zero
cxy, errxy = integrate.nquad(func_xy, [[0,pi],[0,2*pi]])

chi = Chi() # Xi_ex, error, Ei-ey, error

#

# Decay rates
def gamma(i, j, E, hx, hy, T):
    return 2 * pi * (np.abs(hx[i, j])**2 * chi[0] + np.abs(hy[i, j])**2 * chi[2]) * (E[i] - E[j])**3 * n_average(E[i] - E[j], T)

# Differential equation
def master_eq(t, rho_flat, H_ac, diss, E):
    
    # Microwave frequency
    Omega_ac = (E[1].real-E[0].real)
    
    # Frame Rotation
    H_rot_t = U(E,t) @ H_ac @ U(-E,t) * np.cos(Omega_ac * t-np.pi/2)
    
    # Vectorization of the matrix product
    H_real = (np.kron(np.eye(4), np.real(H_rot_t)) - np.kron(np.real(H_rot_t.T), np.eye(4)))
    H_imag = (np.kron(np.eye(4), np.imag(H_rot_t)) - np.kron(np.imag(H_rot_t.T), np.eye(4)))
    
    # Separation of Real and Imaginary part
    H_AC = np.zeros((32,32))

    H_AC[0:16, 0:16] = H_imag
    H_AC[0:16, 16:32] = H_real
    H_AC[16:32, 0:16] = - H_real
    H_AC[16:32, 16:32] = H_imag   

    return (H_AC + diss) @ rho_flat

# Vectorization of Lindblad operator
def superoperator_lindblad(L):
    """
    Create the superoperator corresponding to a single Lindblad operator.
    """
    dim = L.shape[0]
    L_dag_L = np.dot(L.conj().T, L)
    term1 = np.kron(L, L.conj())
    term2 = 0.5 * np.kron(np.eye(dim), L_dag_L) + 0.5 * np.kron(L_dag_L.T, np.eye(dim))
    return term1 - term2

def Markovian(B,b, x , Duration, T, rho0_flat, num = 10000):
    
    # Total Hamiltonian        
    H_tot = Hso(Lambda) + Bpar(B*np.cos(x[0])) + Bper(B*np.sin(x[0])*np.cos(x[1]),B*np.sin(x[0])*np.sin(x[1])) + Strain(x[4],x[5])
    
    # Diagonalization of the Hamiltonian and extracting the energy levels and eigenbasis transformation
    E, P, P_dag = Diag(H_tot)

    # Transformation into the eigenbasis
    H_ac = P_dag @ (Bpar(b*np.cos(x[2])) + Bper(b*np.sin(x[2])*np.cos(x[3]),b*np.sin(x[2])*np.sin(x[3]))) @ P

    hx = h_x(P,P_dag)
    hy = h_y(P,P_dag)
    
    # Initial and end time
    t_start = 0.0
    t_end = Duration  
     
    t = np.linspace(t_start, t_end, num) # The algorithm may diverge for num = 1000 when we take gammas=0
    
    diss = np.zeros((32,32)) # Dissipation Matrix
    
    for i in range(4):
        for j in range(4):
            gamma_val = gamma(i, j, E, hx, hy, T).real # real to remove the 0j
            sig = sigma(i, j)         
            diss += gamma_val * np.kron(np.eye(2),superoperator_lindblad(sig))
    
    # Free Evolution
    # Create an 'ode' object
    solver = ode(master_eq)
    solver.set_integrator('vode', method='bdf', atol=1e-10, rtol=1e-8, nsteps= 1e8)
    solver.set_initial_value(rho0_flat, t_start)
    # precompute parameters
    params = (np.zeros((4,4)), diss, E.real)
    solver.set_f_params(*params)

    # Integrate the ODEs using t_eval
    solution = [rho0_flat]  # Start with the initial state
    for t_eval in t[1:]:
        solver.integrate(t_eval)
        solution.append(solver.y)
    
    # Density vector, vectorized density matrix by columns
    Data = np.array(solution)
    Rho_F = Data[:,:16] + 1j * Data[:,16:]
    
    # Microwave control evolution
    # Create an 'ode' object
    solver = ode(master_eq)
    solver.set_integrator('vode', method='bdf', atol=1e-10, rtol=1e-8, nsteps= 1e8)
    solver.set_initial_value(rho0_flat, t_start)
    # precompute parameters
    params = (H_ac, diss, E.real)
    solver.set_f_params(*params)

    
    # Integrate the ODEs using t_eval
    solution = [rho0_flat]  # Start with the initial state
    for t_eval in t[1:]:
        solver.integrate(t_eval)
        solution.append(solver.y)
    
    # Density vector, vectorized density matrix by columns
    Data = np.array(solution)
    Rho_D = Data[:,:16] + 1j * Data[:,16:]
       
    return Rho_F, Rho_D, t


def states(ind,x_par,x_strain,y_strain,B,b,T):
    
    #####
    x = x_par + x_strain + y_strain
    #####

    Period = 1 / Rabi(B,b,x)

    # Duration of the evolution
    Duration = Period/4
    
    rho0s=[np.array([[1,0,0,0],
              [0,0,0,0],
              [0,0,0,0],
              [0,0,0,0]]),
          np.array([[0,1,0,0],
              [0,0,0,0],
              [0,0,0,0],
              [0,0,0,0]]),
          np.array([[0,0,0,0],
              [1,0,0,0],
              [0,0,0,0],
              [0,0,0,0]]),
          np.array([[0,0,0,0],
              [0,1,0,0],
              [0,0,0,0],
              [0,0,0,0]])]
    
    rho0=rho0s[ind]
    
    rho0_flat = rho0.flatten('F')
    rho0_flat = np.concatenate((rho0_flat.real, rho0_flat.imag))

    _, Rho_D, t = Markovian(B,b, x , Duration, T, rho0_flat)

    rho_T=Rho_D[-1,:].reshape((4,4),order='F')
    rhoT=rho_T[0:2,0:2]
    rhoT_proj=np.zeros((4,4),dtype=np.complex128)
    rhoT_proj[0:2,0:2]=rhoT
    e=np.linalg.norm(rhoT_proj-rho_T, ord=1)
    return rhoT,e

def get_spin_states_mw_SnV(theta_dc,theta_ac,Ex,Epsilon_xy,B,b,T,data_transfer):
    x_strain=np.array([0,0,0,0,Ex,0])
    y_strain=np.array([0,0,0,0,0,Epsilon_xy])
    d= 0.787e6
    x_par=np.array([theta_dc,0,theta_ac,0,65/d,0])
    rhoTs=[]
    if data_transfer=='readin':
        rhoT,e=states(0,x_par,x_strain,y_strain,B,b,T)
        
        return rhoT,e
    
    else:
        for i in range(4):
            rhoT,e=states(i,x_par,x_strain,y_strain,B,b,T)
            rhoTs.append(rhoT)
        
        return np.vstack((rhoTs[0],rhoTs[1],rhoTs[2],rhoTs[3])),e
