"""
 * Module Name: main
 * Description: Kraus-operators and approximation error
                for read-in or read-out suject to control
                choice and parameters
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import numpy as np
from qsi.coordinator import Coordinator
import random


# Create coordinator and start the modules:
coordinator = Coordinator(port=random.randint(1, 100))
# Coherent source
cc = coordinator.register_component(module="color_center.py", runtime="python")
# Run the coordinator process
coordinator.run()



'''

* Parameters:
    
    F_gen (float): fidelity of the incoming photon
    bw (float): bandwidth of the incoming photon in THz
    temp (float): temperature in K
    Ex (float): axial strain
    eps_xy (float): shear strain
    B_dc (float): DC magnetic field strength in Tesla
    B_ac (float): AC magnetic field strength in Tesla
    theta_dc (float): orientation of the DC field in rad
    theta_ac (float): orientation of the AC field in rad
    control (string): opt (optical control) or mw (microwave control)
    data_transfer (string): readin or readout
    
* Note:
    
    control = 'opt': B_dc = 3 T, theta_dc = 43.11 deg, strain = 0 are set automatically
    data_transfer = 'readout': F_gen and bw have no influence on the output because the SPS for
                               read-out is assumed to work perfectly
    optical control with a different magnetic field strength must be optimized seperately
    
'''


d = 0.787e3
cc.set_param("F_gen", 1.0)
cc.set_param("temp", 0.1)
cc.set_param("bw", 0.00318)
cc.set_param("Ex", 5/d)
cc.set_param("eps_xy", 2/d)
cc.set_param("B_dc", 3)
cc.set_param("B_ac", 1e-3)
cc.set_param("theta_dc", 0)
cc.set_param("theta_ac", np.pi/2)
cc.set_param("control", "opt")
cc.set_param("data_transfer", "readin")


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