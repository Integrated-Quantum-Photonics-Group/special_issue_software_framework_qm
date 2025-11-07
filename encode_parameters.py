"""
 * Module Name: encode_parameters
 * Description: return json files
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: November 7, 2025
 * Version: 1.0
"""

import json
from json_coding import encode_for_json


def encode_parameters(C1A,C2B,C2A,C1B,rho00,rho11,rho_p,rho_pi):
    
    """
    
    * Function: encode_parameters
    
    * Parameters:
    
    C1A (float) : cooperativity for coupling 1A
    C2B (float): cooperativity for coupling 2B
    C2A (float): cooperativity for coupling 2A
    C1B (float): cooperativity for coupling 1B
    rho00 (array of complex): propagated state |1><1|
    rho11 (array of complex): propagated state |1><1|
    rho_p (array of complex): propagated state |+><+|
    rho_pi (array of complex): propagated state |R><R|

    * Returns:
    
    Cs : json encoded cooperativity list
    rhos : json encoded propagated states as a list

    """
    
    rhos = {
        "rho00": rho00,
        "rho11": rho11,
        "rho_p": rho_p,
        "rho_pi": rho_pi
    }
    
    Cs={
        "C1A": C1A,
        "C2B": C2B,
        "C2A": C2A,
        "C1B": C1B}

    rhos_encoded = encode_for_json(rhos)
    rhos = json.dumps(rhos_encoded)
    
    Cs_encoded = encode_for_json(Cs)
    Cs = json.dumps(Cs_encoded)
    
    
    return Cs,rhos
