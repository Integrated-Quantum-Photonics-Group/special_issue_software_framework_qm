"""
 * Module Name: component
 * Description: computation of Kraus-operators and approximation error
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: October 31, 2025
 * Version: 2.0
"""

import numpy as np
from optimize_entanglement_fidelity import get_params
from spin_states import get_spin_states
from mw_control import get_spin_states_mw
from saved_state import get_saved_spins
from kraus_operators import get_kraus_single
from readout import readout
import json
from json_coding import decode_from_json

class component:
    
    '''
    * Class: component
    
    * Purpose: computation of Kraus-operators and approximation error depending on system parameters
             control technique (either optical or microwave) and the data transfer direction (readin or readout)
    '''
    
    def __init__(self,center,spin_gate,phase_gate,k,dc,dw0,Cs,spin_states,TT,ga_in,ga_out,Ex,eps_xy,B_dc,B_ac,theta_dc,theta_ac,C_max,F_aux,measurement,control,data_transfer):
        
        self.center=center
        self.spin_gate=spin_gate
        self.phase_gate=phase_gate
        self.k=k
        self.dc=dc
        self.dw0=dw0
        self.Cs=Cs
        self.spin_states=spin_states
        self.TT=TT
        self.ga_in=ga_in
        self.ga_out=ga_out
        self.Ex=Ex
        self.eps_xy=eps_xy
        self.B_dc=B_dc
        self.B_ac=B_ac
        self.theta_dc=theta_dc
        self.theta_ac=theta_ac
        self.C_max=C_max
        self.F_aux=F_aux
        self.measurement=measurement
        self.control=control
        self.data_transfer=data_transfer
    
    def channel(self):
        
        '''
        * Function: channel
        
        * Purpose: computation of Kraus-operators and approximation error depending on system parameters
                 control technique (either optical or microwave) and the data transfer direction (readin or readout)
                 
        * Parameters:
            
        center (string): color center ("SiV" or "SnV")
        spin_gate (string): choose states from implemented model or input them yourself ("optimized" or "experimental")
        phase_gate (string): choose optimized cavity parameters or input them yourself ("optimized" or "experimental")
        k (float): HWHM cavity loss rate (GHz)
        dc (float): cavity mode frequency detuning from the defect center's transition frequency (GHz)
        dw0 (float): incoming mode frequency detuning from the defect center's transition frequency (GHz)
        spin_states (string): json files of the propagated states if you chose spin_gate=experimental
        Cs (string): json files of the cooperativities if you chose control=opt and phase_gate=experimental
        TT (float): temperature (K)
        ga_in (float): bandwidth of the photon for read-in
        ga_out (float): bandwidth of the photon for read-out
        Ex (float): axial strain
        eps_xy (float): shear strain
        B_dc (float): DC magnetic field strength in Tesla
        B_ac (float): AC magnetic field strength in Tesla
        theta_dc (float): orientation of the DC field in rad
        theta_ac (float): orientation of the AC field in rad
        F_aux (float): fidelity of photon from auxiliary photon source for read-out
        C_max (float): cooperativity bound for phase gate optimization with definition using FWHM for cavity loss rate and decay rate (<25)
        measurement (string): +/- for read-in and down/up for read-out
        control (string): opt (optical control) or mw (microwave control)
        data_transfer (string): readin or readout
    
        * Returns:
            
            (complex): list of the Kraus-operators
            (float): approximation error
        '''
        
        if self.data_transfer=='readin':
            if self.phase_gate=='optimized':
                if self.spin_gate=='optimized':
                    if self.control=='opt':
                        
                        # read-in with optical control
                        
                        # define the DC magnetic field
                        if self.B_dc==3:
                            angle=43.11/360*2*np.pi
                        elif self.B_dc==1:
                            angle=64.62/360*2*np.pi
                        elif self.B_dc==0.3:
                            angle=81.4/360*2*np.pi
                        
                        # optimize the phase gate by tuning cavity loss rate k, detuning dc and central frequency w0
                        # evaluate phase gate fidelity F and cooperativity C
                        # cooperativity C influences the spin states
                        ga=np.max([self.ga_in,self.ga_out])
                        
                        y,I,eta,C1A,C2B,C2A,C1B=get_params('SnV',self.B_dc*1e-24,angle,ga,self.C_max,0,0)
                        k,dc,dw0=y
                        
                        # for the given temperature, magnetic field and cooperativity evaluate spin states (states) and the
                        # approximation error (e)
                        states,e=get_spin_states(self.B_dc*1e-24,C1A,C2B,C2A,C1B,self.TT)
                        La=states[0:2,0:2]
                        
                        # using all these parameters and evaluate the spin states after read-in
                        single_spins=get_saved_spins('SnV',self.measurement,self.B_dc*1e-24,angle,k,dc,dw0,self.ga_in,La,0,0)
                        #print(np.round(single_spins,3))
                        
                        # compute Kraus-operators for the read-in process
                        Ks=get_kraus_single(single_spins)
                    
                    elif self.control=='mw':
                        
                        # read-in with microwave control
                        
                        # optimize the phase gate by tuning cavity loss rate k, detuning dc and central frequency w0
                        
                        ga=np.max([self.ga_in,self.ga_out])
        
                        y,I,eta,C1A,C2B,C2A,C1B=get_params(self.center,self.B_dc*1e-24,self.theta_dc,ga,self.C_max,self.Ex,self.eps_xy)
                        k,dc,dw0=y
                        # for the given temperature, magnetic field and strain evaluate spin states (states) and the
                        # approximation error (e)
                        La,e=get_spin_states_mw(self.center,self.theta_dc,self.theta_ac,self.Ex,self.eps_xy,self.B_dc,self.B_ac,self.TT,self.data_transfer)
                        # using all these parameters and evaluate the spin states after read-in
                        single_spins=get_saved_spins(self.center,self.measurement,self.B_dc*1e-24,self.theta_dc,k,dc,dw0,self.ga_in,La,self.Ex,self.eps_xy)  
                        
                        # compute Kraus-operators for the read-in process
                        Ks=get_kraus_single(single_spins)
                        
                elif self.spin_gate=='experimental':
                    
                    ga=np.max([self.ga_in,self.ga_out])
                    y,I,eta,C1A,C2B,C2A,C1B=get_params(self.center,self.B_dc*1e-24,self.theta_dc,ga,self.C_max,self.Ex,self.eps_xy)
                    k,dc,dw0=y
                    
                    e=0
                    
                    rhos_loaded_raw = json.loads(self.spin_states)
                    Las = decode_from_json(rhos_loaded_raw)
                    La = Las["rho00"]
                    
                    single_spins=get_saved_spins(self.center,self.measurement,self.B_dc*1e-24,self.theta_dc,k,dc,dw0,self.ga_in,La,self.Ex,self.eps_xy)  
                    Ks=get_kraus_single(single_spins)
                
            elif self.phase_gate=='experimental':
                if self.spin_gate=='optimized':
                    if self.control=='opt':
                        # read-in with optical control
                        
                        # define the DC magnetic field
                        if self.B_dc==3:
                            angle=43.11/360*2*np.pi
                        elif self.B_dc==1:
                            angle=64.62/360*2*np.pi
                        elif self.B_dc==0.3:
                            angle=81.4/360*2*np.pi
                        
                        # optimize the phase gate by tuning cavity loss rate k, detuning dc and central frequency w0
                        # evaluate phase gate fidelity F and cooperativity C
                        # cooperativity C influences the spin states
                        ga=np.max([self.ga_in,self.ga_out])
                        
                        # for the given temperature, magnetic field and cooperativity evaluate spin states (states) and the
                        # approximation error (e)
                        Cs_loaded_raw=json.loads(self.Cs)
                        Cs=decode_from_json(Cs_loaded_raw)
                        C1A=Cs["C1A"]
                        C2B=Cs["C2B"]
                        C2A=Cs["C2A"]
                        C1B=Cs["C1B"]
                        states,e=get_spin_states(self.B_dc*1e-24,C1A,C2B,C2A,C1B,self.TT)
                        La=states[0:2,0:2]
                        
                        # using all these parameters and evaluate the spin states after read-in
                        single_spins=get_saved_spins('SnV',self.measurement,self.B_dc*1e-24,angle,self.k,self.dc,self.dw0,self.ga_in,La,0,0)
                        #print(np.round(single_spins,3))
                        
                        # compute Kraus-operators for the read-in process
                        Ks=get_kraus_single(single_spins)
                    
                    elif self.control=='mw':
                        
                        # read-in with microwave control
                        
                        # optimize the phase gate by tuning cavity loss rate k, detuning dc and central frequency w0
                        
                        ga=np.max([self.ga_in,self.ga_out])
        
                        # for the given temperature, magnetic field and strain evaluate spin states (states) and the
                        # approximation error (e)
                        La,e=get_spin_states_mw(self.center,self.theta_dc,self.theta_ac,self.Ex,self.eps_xy,self.B_dc,self.B_ac,self.TT,self.data_transfer)
                        # using all these parameters and evaluate the spin states after read-in
                        single_spins=get_saved_spins(self.center,self.measurement,self.B_dc*1e-24,self.theta_dc,self.k,self.dc,self.dw0,self.ga_in,La,self.Ex,self.eps_xy)  
                        
                        # compute Kraus-operators for the read-in process
                        Ks=get_kraus_single(single_spins)
                        
                elif self.spin_gate=='experimental':
                    
                    e=0
                    rhos_loaded_raw = json.loads(self.spin_states)
                    Las = decode_from_json(rhos_loaded_raw)
                    La = Las["rho00"]
                    single_spins=get_saved_spins(self.center,self.measurement,self.B_dc*1e-24,self.theta_dc,self.k,self.dc,self.dw0,self.ga_in,La,self.Ex,self.eps_xy)  
                    
                    # compute Kraus-operators for the read-in process
                    Ks=get_kraus_single(single_spins)
            
        
        elif self.data_transfer=='readout':
            if self.phase_gate=='optimized':
                if self.spin_gate=='optimized':
                    if self.control=='opt':
                        
                        # read-out with optical control
                        
                        # define the DC magnetic field
                        # define the DC magnetic field
                        if self.B_dc==3:
                            angle=43.11/360*2*np.pi
                        elif self.B_dc==1:
                            angle=64.62/360*2*np.pi
                        elif self.B_dc==0.3:
                            angle=81.4/360*2*np.pi
                        
                        # optimize the phase gate by tuning cavity loss rate k, detuning dc and central frequency w0
                        # evaluate phase gate fidelity F and cooperativity C
                        # cooperativity C influences the spin states
                        ga=np.max([self.ga_in,self.ga_out])
                        y,I,eta,C1A,C2B,C2A,C1B=get_params('SnV',self.B_dc*1e-24,angle,ga,self.C_max,0,0)
        
                        k,dc,dw0=y
                        
                        # for the given temperature, magnetic field and cooperativity evaluate spin states (states) and the
                        # approximation error (e)
                        states,e=get_spin_states(self.B_dc*1e-24,C1A,C2B,C2A,C1B,self.TT)
                        Las=[states[2*i:2*(i+1),:] for i in range(4)]
                        
                        states=readout(self.center,self.measurement,self.F_aux,self.B_dc*1e-24,angle,k,dc,dw0,self.ga_out,Las,0,0)
                        
                        # evaluate Kraus-operators for the pi/2 rotation
                        Ks=get_kraus_single(states)
                        
                    
                    elif self.control=='mw':
                        
                        # readout with microwave control
                        ga=np.max([self.ga_in,self.ga_out])
                        y,I,eta,C1A,C2B,C2A,C1B=get_params(self.center,self.B_dc*1e-24,self.theta_dc,ga,self.C_max,self.Ex,self.eps_xy)
                        k,dc,dw0=y
                        
                        # for the given temperature, magnetic field and strain evaluate spin states (states) and the
                        # approximation error (e)
                        states,e=get_spin_states_mw(self.center,self.theta_dc,self.theta_ac,self.Ex,self.eps_xy,self.B_dc,self.B_ac,self.TT,self.data_transfer)
                        Las=[states[2*i:2*(i+1),:] for i in range(4)]
                        
           
                        # evaluate Kraus-operators for the pi/2 rotation
                        states=readout(self.center,self.measurement,self.F_aux,self.B_dc*1e-24,self.theta_dc,k,dc,dw0,self.ga_out,Las,self.Ex,self.eps_xy)
                        
                        # evaluate Kraus-operators for the read-out process
                        Ks=get_kraus_single(states)
                
                elif self.spin_gate=='experimental':
                    
                    e=0
                    ga=np.max([self.ga_in,self.ga_out])
                    y,I,eta,C1A,C2B,C2A,C1B=get_params(self.center,self.B_dc*1e-24,self.theta_dc,ga,self.C_max,self.Ex,self.eps_xy)
    
                    k,dc,dw0=y
                    
                    # for the given temperature, magnetic field and cooperativity evaluate spin states (states) and the
                    # approximation error (e)
                    rhos_loaded_raw = json.loads(self.spin_states)
                    Las = decode_from_json(rhos_loaded_raw)
                    
                    
                    rho00=Las["rho00"]
                    rho11=Las["rho11"]
                    rho_p=Las["rho_p"]
                    rho_pi=Las["rho_pi"]
                    rho01=-1/2*(1+1j)*rho00-1/2*(1+1j)*rho11+rho_p+1j*rho_pi
                    rho10=1/2*(-1+1j)*rho00+1/2*(-1+1j)*rho11+rho_p-1j*rho_pi
                    
                    Las=[rho00,rho01,rho10,rho11]
                    
                    states=readout(self.center,self.measurement,self.F_aux,self.B_dc*1e-24,self.theta_dc,k,dc,dw0,self.ga_out,Las,self.Ex,self.eps_xy)
                    
                    # evaluate Kraus-operators for the pi/2 rotation
                    Ks=get_kraus_single(states)
            
            elif self.phase_gate=='experimental':
                if self.spin_gate=='optimized':
                    if self.control=='opt':
                        
                        # read-out with optical control
                        
                        # define the DC magnetic field
                        # define the DC magnetic field
                        if self.B_dc==3:
                            angle=43.11/360*2*np.pi
                        elif self.B_dc==1:
                            angle=64.62/360*2*np.pi
                        elif self.B_dc==0.3:
                            angle=81.4/360*2*np.pi
                        
                        # for the given temperature, magnetic field and cooperativity evaluate spin states (states) and the
                        # approximation error (e)
                        
                        Cs_loaded_raw=json.loads(self.Cs)
                        Cs=decode_from_json(Cs_loaded_raw)
                        C1A=Cs["C1A"]
                        C2B=Cs["C2B"]
                        C2A=Cs["C2A"]
                        C1B=Cs["C1B"]
                        states,e=get_spin_states(self.B_dc*1e-24,C1A,C2B,C2A,C1B,self.TT)
                        Las=[states[2*i:2*(i+1),:] for i in range(4)]
                        
                        states=readout(self.center,self.measurement,self.F_aux,self.B_dc*1e-24,angle,self.k,self.dc,self.dw0,self.ga_out,Las,0,0)
                        
                        # evaluate Kraus-operators for the pi/2 rotation
                        Ks=get_kraus_single(states)
                        
                    
                    elif self.control=='mw':
                        
                        # readout with microwave control
                        #ga=np.max([self.ga_in,self.ga_out])
                        
                        # for the given temperature, magnetic field and strain evaluate spin states (states) and the
                        # approximation error (e)
                        states,e=get_spin_states_mw(self.center,self.theta_dc,self.theta_ac,self.Ex,self.eps_xy,self.B_dc,self.B_ac,self.TT,self.data_transfer)
                        Las=[states[2*i:2*(i+1),:] for i in range(4)]
                        
           
                        # evaluate Kraus-operators for the pi/2 rotation
                        states=readout(self.center,self.measurement,self.F_aux,self.B_dc*1e-24,self.theta_dc,self.k,self.dc,self.dw0,self.ga_out,Las,self.Ex,self.eps_xy)
                        
                        # evaluate Kraus-operators for the read-out process
                        Ks=get_kraus_single(states)
                
                elif self.spin_gate=='experimental':
                    
                    e=0
                    rhos_loaded_raw = json.loads(self.spin_states)
                    Las = decode_from_json(rhos_loaded_raw)
                    
                    rho00=Las["rho00"]
                    rho11=Las["rho11"]
                    rho_p=Las["rho_p"]
                    rho_pi=Las["rho_pi"]
                    
                    rho01=-1/2*(1+1j)*rho00-1/2*(1+1j)*rho11+rho_p+1j*rho_pi
                    rho10=1/2*(-1+1j)*rho00+1/2*(-1+1j)*rho11+rho_p-1j*rho_pi
                    
                    Las=[rho00,rho01,rho10,rho11]
                    
                    states=readout(self.center,self.measurement,self.F_aux,self.B_dc*1e-24,self.theta_dc,self.k,self.dc,self.dw0,self.ga_out,Las,self.Ex,self.eps_xy)
                    
                    # evaluate Kraus-operators for the pi/2 rotation
                    Ks=get_kraus_single(states)
            
        return Ks,e