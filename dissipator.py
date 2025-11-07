"""
 * Module Name: dissipator
 * Description: computation of the dissipator L in the Lindblad-master equation
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: October 6, 2025
 * Version: 2.0
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
from scipy import integrate

class dissipator:
    
    '''
    * Class: dissipator
    
    * Purpose: computation of the dissipator L in the Lindblad-master equation including photonic and
               phononic processes
    
    '''
    
    def __init__(self,THz,m,kg,evals,S,a,C1A,C2B,C2A,C1B,A,alpha_g,alpha_u,beta_g,beta_u,B,angle,chis,TT):
        
        '''
        * Parameters:
            
            THz (float): unit terahertz
            A (float): unit ampere
            kg (float): unit kilograms
            m (float): unit meter
            evals (array of floats): energy levels of the SnV
            S (array of complex): eigenvectors of the SnV Hamiltonian
            a (float): a constant for the interaction Hamiltonian
                       and decay rates according to Fermi's golden rule
            C (float): cooperativity
            alpha_g (float): strain component ground state
            alpha_u (float): strain component excited state
            beta_g (float): strain component ground state
            beta_u (float): strain component excited state
            B (float): DC magnetic field strength
            angle (float): DC magnetic field orientation
            chis (list of floats): phononic absorption cross section
            TT (float): temperature
        '''
        
        self.THz=THz
        self.m=m
        self.kg=kg
        self.evals=evals
        self.S=S
        self.a=a
        self.C1A=C1A
        self.C2B=C2B
        self.C2A=C2A
        self.C1B=C1B
        self.A=A
        self.alpha_g=alpha_g
        self.alpha_u=alpha_u
        self.beta_g=beta_g
        self.beta_u=beta_u
        self.B=B
        self.angle=angle
        self.chis=chis
        self.TT=TT
    
    
    def phononic_decay(self):
        
        '''
        * Function: phononic_decay
        
        * Purpose: description of relaxation and excitation rates due to phonons
        
        * Returns:
            
            (array): Lindblad-operators describing phononic decay according to I. Harris et al. (2023)'
        '''
        
        
        C_d=hooke(self.THz,self.kg,self.m).get_tensor()
        Rs=['gx','gy','ux','uy']
        chis=[]
        for R in Rs:
            chis.append(phononic_absorption(self.THz,self.kg,self.m,R,C_d).get_chi_R())
        
        gas_g,gas_e=phononic_decay_rates(self.THz,self.A,self.kg,self.m,self.alpha_g,self.alpha_u,self.beta_g,self.beta_u,self.B,self.angle,self.chis,self.TT).get_decay_rates()
        
        
        I=np.eye(8)
        Ls=[]
        for i in range(4):
            ket_i=I[:,i].reshape(8,1)
            bra_i=I[:,i].reshape(1,8)
            for j in range(i+1,4):
                bra_j=I[:,j].reshape(1,8)
                ket_j=I[:,j].reshape(8,1)
                
                Ls.append(np.sqrt(gas_g[j,i])*np.dot(ket_i,bra_j))
                Ls.append(np.sqrt(gas_g[i,j])*np.dot(ket_j,bra_i))
        
        for i in range(4,8):
            ket_i=I[:,i].reshape(8,1)
            bra_i=I[:,i].reshape(1,8)
            for j in range(i+1,8):
                ket_j=I[:,j].reshape(8,1)
                bra_j=I[:,j].reshape(1,8)
                Ls.append(np.sqrt(gas_e[j-4,i-4])*np.dot(ket_i,bra_j))
                Ls.append(np.sqrt(gas_e[i-4,j-4])*np.dot(ket_j,bra_i))
                
        mm=len(Ls)
        Ls_=np.zeros((8*mm,8))
        for i in range(mm):
            Ls_[8*i:8*(i+1),:]=Ls[i]
        
        return Ls_
    
    def photonic_decay(self,H1,H2,H3):
        
        '''
        * Function: photonic_decay
        
        * Purpose: description of decay rates due to spontaneous emission
        
        * Returns:
            
            (array): Lindblad-operators describing photonic decay according to Fermi's golden rule'
        '''
        
        c=3*10**(-4)*self.THz
        alpha=1/137
        n=2.47
            
        L=np.zeros((16*8,8),dtype=np.complex128)
        mm=0
        I=np.eye(8)
        gas=[]
        for i in range(0,4):
            for j in range(4,8):
                w_ji=self.evals[j]-self.evals[i]
                bra_j=np.conjugate(self.S[:,j].reshape(1,8))
                ket_i=self.S[:,i].reshape(8,1)
                ga_ij=4*alpha*n*self.a**2*w_ji**3/(3*c**2)*(np.abs(np.dot(bra_j,np.dot(H1,ket_i)))**2+np.abs(np.dot(bra_j,np.dot(H2,ket_i)))**2+np.abs(np.dot(bra_j,np.dot(H3,ket_i)))**2)
                if i==0 and j==4:
                    ga_ij=ga_ij*(1+self.C1A)
                elif i==1 and j==5:
                    ga_ij=ga_ij*(1+self.C2B)
                elif i==1 and j==4:
                    ga_ij=ga_ij*(1+self.C2A)
                elif i==0 and j==5:
                    ga_ij=ga_ij*(1+self.C1B)

                ga_ij=ga_ij[0,0]
                
                gas.append(ga_ij)
                bra_j=I[:,j].reshape(1,8)
                ket_i=I[:,i].reshape(8,1)
                L_ij=np.sqrt(ga_ij)*np.dot(ket_i,bra_j)
                L[8*mm:8*(mm+1),:]=L_ij
                mm=mm+1
        
        
                
                        
        return L   
    
    def vectorized_dissipator(self):
        
        '''
        * Function: vectorized_dissipator
        
        * Purpose: time integration 
        
        * Returns:
            
            (array): Lindblad-operators describing photonic and phononic decay in the form Lρ in vectorized form'
        '''
        
        H1,H2,H3=SnV(0,0,0,0,0,0,0,0,0,0).interaction()
        L_photon=self.photonic_decay(H1,H2,H3)
        L_phonon=self.phononic_decay()
        L_tot=np.vstack((L_photon,L_phonon))

        
        [p,q]=np.shape(L_tot)
        num_lind=int(p/8)
        I=np.eye(8,dtype=np.complex128)
        L=np.zeros((64,64),dtype=np.complex128)
        for k in range(num_lind):
            Lk=L_tot[8*k:8*(k+1),:]
            Lk_c=np.conjugate(Lk)
            L=L+np.kron(Lk_c,Lk)-0.5*np.kron(I,np.dot(Lk_c.T,Lk))-0.5*np.kron(np.dot(Lk.T,Lk_c),I)

        return L

class hooke:
    
    '''
    * Class: hooke
    
    * Purpose: computation of Hookes stiffness tensor for Hookes law for diamond
    
    '''
    
    
    def __init__(self,THz,kg,m):
        
        '''
        * Parameters:
            
            THz (float): unit terahertz
            kg (float): unit kilograms
            m (float): unit meter
        '''
        
        self.THz=THz
        self.kg=kg
        self.m=m
    
    def get_tensor(self):
        
        '''
        
        * Function: get_tensor
        
        * Purpose: computation of phononic decay rates
        
        * Returns:
            
            Hookes stiffness tensor for diamond
        
        '''
        
        Hz=1e-12*self.THz
        s=1/Hz
        N=self.kg*self.m/s**2
        Pa=N/self.m**2
        GPa=1e+9*Pa
        C11 = 1079.26*GPa
        C12 = 126.73*GPa
        C44 = 578.16*GPa
        
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

        # Z rotation
        theta = -3*np.pi/4
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
        
        return C_d
    
    
class phononic_absorption:
    
    '''
    
    * Class: phononic_absorption
    
    * Purpose: ingredient for computing phononic decay rates
    
    '''
    
    
    def __init__(self,THz,kg,m,R,C_d):
        
        '''
        * Parameters:
            
            THz (float): unit terahertz
            kg (float): unit kilograms
            m (float): unit meter
            R (string): orbital
            C_d (array of floats): Hookes stiffness tensor
        '''
        
        self.THz=THz
        self.kg=kg
        self.m=m
        self.R=R
        self.C_d=C_d
        
    def Wavemodes(self,k,Stiffness,rho):
        
        '''
        
        * Function: Wavemodes
        
        * Purpose: computation of phononic decay rates
        
        * Returns: (list of floats): eigenvalues of C_{ijkl}k_j k_k
                   (array of complex): eigenvectors of C_{ijkl}k_j k_k
            
        '''

        Matrix = np.einsum("ijkl, j, k -> il", Stiffness,k,k)
        
        EigVa, q = np.linalg.eig(Matrix/rho)

        sorted_indexes = np.argsort(EigVa)
        
        EigVa = EigVa[sorted_indexes]
        q = q[:,sorted_indexes]
            
        return EigVa,q
                
    def func(self,theta,phi):
            
        '''
        * Function: func
        
        * Purpose: computation of phononic absorption cross section
        
        * Returns: (float):  the function to be integrated 
        in spherical coordinates for getting the phononic absorption cross section
        
        * Note: 
        
        theta,phi: spherical cooradinates
        k=(cos(phi)sin(theta),sin(phi)sin(theta),cos(theta))
        la,mu: constants in Hookes law
        A: result of simplifying Hookes law for isotropic diamond material
        '''
        g=1e-3*self.kg
        cm=1e-2*self.m
        Hz=1e-12*self.THz
        dg=0.787*1e+15*Hz*2*np.pi
        fg=-0.562*1e+15*Hz*2*np.pi
        du=0.956*1e+15*Hz*2*np.pi
        fu=-2.555*1e+15*Hz*2*np.pi
        rho=3.51*g/(cm**3)
        
        if self.R=='gx':
            D=np.array([[dg,0,fg/2],
                        [0,-dg,0],
                        [fg/2,0,0]])
        elif self.R=='gy':
            D=np.array([[0,-dg,0],
                        [-dg,0,fg/2],
                        [0,fg/2,0]])
        
        elif self.R=='ux':
            D=np.array([[du,0,fu/2],
                        [0,-du,0],
                        [fu/2,0,0]])
        
        elif self.R=='uy':
            D=np.array([[0,-du,0],
                        [-du,0,fu/2],
                        [0,fu/2,0]])
        
        
        k = np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])
        
        EiV , qs = self.Wavemodes(k,self.C_d,rho)  
       
        val=0
        for i in range(3):
            q = qs[:,i]
            c = np.sqrt(EiV[i])
            val = val + np.einsum("ij,i,j->",D,k,q)**2/(c**5)
        
            
        return val*np.sin(theta)
    
    def get_chi_R(self):
        
        '''
        * Function: get_chi_R
        
        * Purpose: phononic absorption cross section
        
        * Returns: (float): phononic absorption cross section
        
        * Note: chi_R: phononic absorption cross-sections s.t. ga=2pi chi_R w^3(n+1)
        '''
        val, error = integrate.nquad(self.func, [[0,np.pi],[0,2*np.pi]])

        Hz=1e-12*self.THz
        kg=1
        g=1e-3*kg
        m=1
        cm=1e-2*m
        s=1/Hz
        N=kg*m**2/(s**2)
        J=N*m
        rho=3.51*g/(cm**3)
        h=6.62607015e-34*J*s
        hbar=h/(2*np.pi)
        
        return val*hbar/(16*np.pi**3*rho)
    
class phononic_decay_rates:
    
    '''
    
    * Class: phononic_decay_rates
    
    * Purpose: computation of the phononic decay rates
    
    '''
    
    def __init__(self,THz,A,kg,m,alpha_g,alpha_u,beta_g,beta_u,B,angle,chis,TT):
        
        '''
        * Parameters:
            
            THz (float): unit terahertz
            A (float): unit ampere
            kg (float): unit kilograms
            m (float): unit meter
            evals (array of floats): energy levels of the SnV
            S (array of complex): eigenvectors of the SnV Hamiltonian
            a (float): a constant for the interaction Hamiltonian
                       and decay rates according to Fermi's golden rule
            C (float): cooperativity
            alpha_g (float): strain component ground state
            alpha_u (float): strain component excited state
            beta_g (float): strain component ground state
            beta_u (float): strain component excited state
            B (float): DC magnetic field strength
            angle (float): DC magnetic field orientation
            chis (list of floats): phononic absorption cross section
            TT (float): temperature
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
        self.chis=chis
        self.TT=TT
        
    def get_decay_rates(self):
        
        '''
        
        * Function: get_decay_rates
        
        * Purpose: computation of phononic decay rates
        
        * Returns: (array of floats): decay rates in the ground state manifold
                   (array of floats): decay rates in the excited state manifold
        
        '''
        
        Hz=1e-12*self.THz
        s=1/Hz
        N=self.kg*self.m/(s**2)
        J=N*self.m
        h=6.626*1e-34*J*s
        K=1
        kB=1.380649*1e-23*J/K
        TT=self.TT
        hbar=h/(2*np.pi)
        
        H_0,mu_1,mu_2,mu_3,S=SnV(self.THz,self.A,self.kg,self.m,self.alpha_g,
                                 self.alpha_u,self.beta_g,self.beta_u,self.B,
                                 self.angle).hamiltonian()
        
        ws=np.diag(H_0)
        
        hx=np.kron(np.array([[0,1],[1,0]]),np.eye(2))
        hy=np.kron(np.array([[0,-1j],[1j,0]],dtype=complex),np.eye(2))
        
        Sg=S[0:4,0:4]
        Sgdag=np.conjugate(Sg.T)
        Se=S[4:8,4:8]
        Sedag=np.conjugate(Se.T)
        
        hgx=np.dot(Sgdag,np.dot(hx,Sg))
        hgy=np.dot(Sgdag,np.dot(hy,Sg))
        
        hex=np.dot(Sedag,np.dot(hx,Se))
        hey=np.dot(Sedag,np.dot(hy,Se))
                
        chigx=self.chis[0]
        chigy=self.chis[1]
        chiex=self.chis[2]
        chiey=self.chis[3]

        
        gas_g=np.zeros((4,4))
        
        for i in range(4):
            for j in range(i+1,4):
                wij=np.abs(H_0[i,i]-H_0[j,j])
                nn=1/(np.exp(hbar*wij/(kB*TT))-1)
                ga=2*np.pi*(nn+1)*(np.abs(hgx[i,j])**2*chigx*np.abs(ws[i]-ws[j])**3+np.abs(hgy[i,j])**2*chigy*np.abs(ws[i]-ws[j])**3)
                gas_g[j,i]=ga
                
                ga=2*np.pi*nn*(np.abs(hgx[i,j])**2*chigx*np.abs(ws[i]-ws[j])**3+np.abs(hgy[i,j])**2*chigy*np.abs(ws[i]-ws[j])**3)
                gas_g[i,j]=ga
        
        gas_e=np.zeros((4,4))
        
        for i in range(4,8):
            for j in range(i+1,8):
                wij=np.abs(H_0[i,i]-H_0[j,j])
                nn=1/(np.exp(hbar*wij/(kB*TT))-1)
 
                ga=2*np.pi*(nn+1)*(np.abs(hex[i-4,j-4])**2*chiex*np.abs(ws[i]-ws[j])**3+np.abs(hey[i-4,j-4])**2*chiey*np.abs(ws[i]-ws[j])**3)
                gas_e[j-4,i-4]=ga
                
                ga=2*np.pi*nn*(np.abs(hex[i-4,j-4])**2*chiex*np.abs(ws[i]-ws[j])**3+np.abs(hey[i-4,j-4])**2*chiey*np.abs(ws[i]-ws[j])**3)
                gas_e[i-4,j-4]=ga
        
        return gas_g,gas_e