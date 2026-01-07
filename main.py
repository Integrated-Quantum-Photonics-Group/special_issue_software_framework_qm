"""
 * Module Name: main
 * Description: Kraus-operators and approximation error
                for read-in or read-out suject to control
                and defect center
                choice and parameters
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: October 31, 2025
 * Version: 2.0
"""

import numpy as np
from qsi.coordinator import Coordinator
import random
from parameter_query import parameter_query


# Create coordinator and start the modules:
coordinator = Coordinator(port=random.randint(1, 100))
# Coherent source
cc = coordinator.register_component(module="color_center.py", runtime="python")
# Run the coordinator process
coordinator.run()



'''

* Parameters:
    
    center (string): color center ("SiV" or "SnV")
    spin_gate (string): choose states from implemented model or input them yourself ("optimized" or "experimental")
    phase_gate (string): choose optimized cavity parameters or input them yourself ("optimized" or "experimental")
    k (float): HWHM cavity loss rate (GHz)
    dc (float): cavity mode frequency detuning from the defect center's transition frequency (GHz)
    d0 (float): incoming mode frequency detuning from the defect center's transition frequency (GHz)
    spin_states (string): json files of the propagated states if you chose spin_gate=experimental
    Cs (string): json files of the cooperativities if you chose control=opt and phase_gate=experimental
    temp (float): temperature (K)
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
    
* Note:
    
    control = 'opt': choice between B_dc = 0.3, 1.0, 3.0 T with optimized magnetic field orientations from Tab. III in
    Strocka et al. arXiv:2503.04985, Ex = eps_xy = 0 are set automatically
    optical control with a different magnetic field strength must be optimized seperately
    
'''

spin_gate = input('\n Optimized or experimental spin gate? ("optimized" or "experimental"): ').strip()
if spin_gate!="optimized" and spin_gate!="experimental":
    raise ValueError('Input must be "optimized" or "experimental" ')
phase_gate = input('\n Optimized or experimental phase gate? ("optimized" or "experimental"): ').strip()
if phase_gate!="optimized" and phase_gate!="experimental":
    raise ValueError('Input must be "optimized" or "experimental" ')
control = input('\n Microwave or optical control? ("mw" oder "opt"): ').strip()
if control!="mw" and control!="opt":
    raise ValueError('Input must be "mw" or "opt" ')

Cs,rhos,k,dc,d0,center,temp,ga_in,ga_out,Ex,eps_xy,B_dc,B_ac,theta_dc,theta_ac,F_aux,C_max,measurement,data_transfer=parameter_query(spin_gate,phase_gate,control)


cc.set_param("center", center)
cc.set_param("spin_gate", spin_gate)
cc.set_param("phase_gate", phase_gate)
cc.set_param("k", k*2*np.pi*1e-3)
cc.set_param("dc", dc*2*np.pi*1e-3)
cc.set_param("d0", d0*2*np.pi*1e-3)
cc.set_param("spin_states", rhos)
cc.set_param("Cs", Cs)
cc.set_param("temp", temp)
cc.set_param("ga_in", ga_in)
cc.set_param("ga_out", ga_out)
cc.set_param("Ex", Ex)
cc.set_param("eps_xy", eps_xy)
cc.set_param("B_dc", B_dc)
cc.set_param("B_ac", B_ac)
cc.set_param("theta_dc", theta_dc)
cc.set_param("theta_ac", theta_ac)
cc.set_param("F_aux", F_aux)
cc.set_param("C_max", C_max)
cc.set_param("measurement",measurement)
cc.set_param("control", control)
cc.set_param("data_transfer", data_transfer)


cc.send_params()



cc_state = cc.state_init()[0]

# operators (list of arrays of complex): Kraus-operators
response, operators = cc.channel_query(
    cc_state,
    {
        "state": cc_state.state_props[0].uuid
    },
    #signals = [asdict(sig)],
    time = 0.01
)

cc_state.apply_kraus_operators(
    operators,
    cc_state.get_all_props(response["kraus_state_indices"]))

print(cc_state.state)

#import os
#os._exit(0)