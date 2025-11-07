"""
 * Module Name: mw_control
 * Description: microwave control of the SiV/SnV for the spin pi/2 rotation
 * Author: Mohamed Belhassen
 * Created On: October 30, 2025
 * Last Modified: October 30, 2025
 * Version: 1.0
"""

from mw_control_SiV import get_spin_states_mw_SiV
from mw_control_SnV import get_spin_states_mw_SnV

def get_spin_states_mw(center,theta_dc,theta_ac,Ex,Epsilon_xy,B,b,T,data_transfer):
    
    """
    
    * Function: get_spin_states_mw
    
    * Parameters:
        
        center (string): SiV or SnV
        theta_dc (float): DC magnetic field orientation (rad)
        theta_ac (float): AC magnetic field orientation (rad)
        Ex (float): compressive strain
        Epsilon_xy (float): shear strain
        B (float): DC magnetic field strength (T)
        b (float): AC magnetic field strength (T)
        T (float): temperature (K)
        data_transfer (string): readin or readout
        
    * Returns:
        
        rhosT (list of arrays of complex): propagated spin states {|1><1|,|1><2|,|2><1|,|2><2|}
        e (float): approximation error
    """
    
    if center=='SiV':
        rhosT,e=get_spin_states_mw_SiV(theta_dc,theta_ac,Ex,Epsilon_xy,B,b,T,data_transfer)
    elif center=='SnV':
        rhosT,e=get_spin_states_mw_SnV(theta_dc,theta_ac,Ex,Epsilon_xy,B,b,T,data_transfer)
    
    return rhosT,e