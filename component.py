"""
 * Module Name: component
 * Description: computation of Kraus-operators and approximation error
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import numpy as np
from optimize_entanglement_fidelity import get_params
from spin_states import get_spin_states
from mw_control import get_spin_states_mw
from saved_state import get_saved_spins
from kraus_operators import get_kraus_single
from readout import readout

class component:
    
    '''
    * Class: component
    
    * Purpose: computation of Kraus-operators and approximation error depending on system parameters
             control technique (either optical or microwave) and the data transfer direction (readin or readout)
    '''
    
    def __init__(self,Fgen,TT,ga,Ex,eps_xy,B_dc,B_ac,theta_dc,theta_ac,control,data_transfer):
        self.Fgen=Fgen
        self.TT=TT
        self.ga=ga
        self.Ex=Ex
        self.eps_xy=eps_xy
        self.B_dc=B_dc
        self.B_ac=B_ac
        self.theta_dc=theta_dc
        self.theta_ac=theta_ac
        self.control=control
        self.data_transfer=data_transfer
    
    def channel(self):
        
        '''
        * Function: channel
        
        * Purpose: computation of Kraus-operators and approximation error depending on system parameters
                 control technique (either optical or microwave) and the data transfer direction (readin or readout)
                 
        * Parameters:
            
            F_gen (float): fidelity of the incoming photon
            bw (float): bandwidth of the incoming photon
            temp (float): temperature
            Ex_g (float): strain ground state
            Ex_u (float): strain excited state
            eps_xy_g (float): shear strain ground state
            eps_xy_u (float): shear strain excited state
            B_dc (float): DC magnetic field strength
            B_ac (float): AC magnetic field strength
            theta_dc (float): orientation of the DC field
            theta_ac (float): orientation of the AC field
            control (string): opt (optical control) or mw (microwave control)
            data_transfer (string): readin or readout
    
        * Returns:
            
            (complex): list of the Kraus-operators
            (float): approximation error
        '''
        
        if self.data_transfer=='readin':
            
            if self.control=='opt':
                
                # read-in with optical control
                
                # define the DC magnetic field
                B=3*1e-24
                angle=43.11/360*2*np.pi
                
                # optimize the phase gate by tuning cavity loss rate k, detuning dc and central frequency w0
                # evaluate phase gate fidelity F and cooperativity C
                # cooperativity C influences the spin states
                y,I,eta,C1A,C2B,C2A,C1B=get_params(B,angle,self.ga,0,0)
                k,dc,dw0=y
                
                # for the given temperature, magnetic field and cooperativity evaluate spin states (states) and the
                # approximation error (e)
                states,e=get_spin_states(B,C1A,C2B,C2A,C1B,'3',self.TT)
                La=states[0:2,0:2]
    
                # using all these parameters and evaluate the spin states after read-in
                single_spins=get_saved_spins('+',B,angle,self.Fgen,k,dc,dw0,self.ga,La,0,0)
                #print(np.round(single_spins,3))
                
                # compute Kraus-operators for the read-in process
                Ks=get_kraus_single(single_spins)
            
            elif self.control=='mw':
                
                # read-in with microwave control
                
                # optimize the phase gate by tuning cavity loss rate k, detuning dc and central frequency w0
                
                y,I,eta,C1A,C2B,C2A,C1B=get_params(self.B_dc*1e-24,self.theta_dc,self.ga,self.Ex,self.eps_xy)
                
                k,dc,dw0=y
                
                # for the given temperature, magnetic field and strain evaluate spin states (states) and the
                # approximation error (e)
                states,e=get_spin_states_mw(self.theta_dc,self.theta_ac,self.Ex*1e-3,self.eps_xy*1e-3,self.B_dc,self.B_ac,self.TT)
                La=states[0:2,0:2]
                
                # using all these parameters and evaluate the spin states after read-in
                single_spins=get_saved_spins('+',self.B_dc*1e-24,self.theta_dc,self.Fgen,k,dc,dw0,self.ga,La,self.Ex,self.eps_xy)  
                
                # compute Kraus-operators for the read-in process
                Ks=get_kraus_single(single_spins)
        
        elif self.data_transfer=='readout':
            
            if self.control=='opt':
                
                # read-out with optical control
                
                # define the DC magnetic field
                B=3*1e-24
                angle=43.11/360*2*np.pi
                
                # optimize the phase gate by tuning cavity loss rate k, detuning dc and central frequency w0
                # evaluate phase gate fidelity F and cooperativity C
                # cooperativity C influences the spin states
                y,I,eta,C1A,C2B,C2A,C1B=get_params(B,angle,self.ga,0,0)

                k,dc,dw0=y
                
                # for the given temperature, magnetic field and cooperativity evaluate spin states (states) and the
                # approximation error (e)
                states,e=get_spin_states(B,C1A,C2B,C2A,C1B,'3',self.TT)
                states=[states[2*i:2*(i+1),:] for i in range(4)]
                
                # evaluate Kraus-operators for the pi/2 rotation
                Ks=get_kraus_single(states)
                
                # evaluate Kraus-operators for the read-out process
                Ks=readout(Ks)
            
            elif self.control=='mw':
                
                # readout with microwave control
                
                # for the given temperature, magnetic field and strain evaluate spin states (states) and the
                # approximation error (e)
                states,e=get_spin_states_mw(self.theta_dc,self.theta_ac,self.Ex*1e-3,self.eps_xy*1e-3,self.B_dc,self.B_ac,self.TT)
                states=[states[2*i:2*(i+1),:] for i in range(4)]
                
                # evaluate Kraus-operators for the pi/2 rotation
                Ks_rot=get_kraus_single(states)
                
                # evaluate Kraus-operators for the read-out process
                Ks=readout(Ks_rot)
            
        return Ks,e