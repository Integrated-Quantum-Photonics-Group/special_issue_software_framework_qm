"""
 * Module Name: color_center
 * Description: interaction interface between server and client
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: October 31, 2025
 * Version: 2.0
"""

from qsi.qsi import QSI
from qsi.helpers import numpy_to_json
from qsi.state import State, StateProp
from component import component
import uuid

"""
Implementation of the interaction interface
 """
if __name__ == "__main__":
    qsi = QSI()
    state_uuid = uuid.uuid4()
    previous_time = 0

    @qsi.on_message("state_init")
    def state_init(msg):
        state = State(StateProp(
            state_type="internal",
            truncation=2,
            uuid=state_uuid
            ))
        return {
            "msg_type": "state_init_response",
            "states": [state.to_message()]
        }

    @qsi.on_message("param_query")
    def param_query(msg):
        """
        Parameters:
            
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
         
        """
        return {
            "msg_type": "param_query_response",
            "params": {
                "center": "string",
                "spin_gate": "string",
                "phase_gate": "string",
                "k":"number",
                "dc":"number",
                "d0":"number",
                "spin_states":"string",
                "Cs":"number",
                "temp":"number",
                "ga_in": "number",
                "ga_out": "number",
                "Ex":"number",
                "eps_xy":"number",
                "B_dc":"number",
                "B_ac":"number",
                "theta_dc":"number",
                "theta_ac":"number",
                "C_max": "number",
                "F_aux": "number",
                "measurement":"string",
                "control":"string",
                "data_transfer":"string",
                }
            }

    @qsi.on_message("param_set")
    def param_sest(msg):
        
        global center
        global spin_gate
        global phase_gate
        global k
        global dc
        global d0
        global spin_states
        global Cs
        global temp
        global ga_in
        global ga_out
        global Ex
        global eps_xy
        global B_dc
        global B_ac
        global theta_dc
        global theta_ac
        global C_max
        global F_aux
        global measurement
        global control
        global data_transfer
        
        params = msg["params"]
        if "center" in params:
            center = params["center"]["value"]
        if "spin_gate" in params:
            spin_gate = params["spin_gate"]["value"]
        if "phase_gate" in params:
            phase_gate = params["phase_gate"]["value"]
        if "k" in params:
            k = params["k"]["value"]
        if "dc" in params:
            dc = params["dc"]["value"]
        if "d0" in params:
            d0 = params["d0"]["value"]
        if "spin_states" in params:
            spin_states = params["spin_states"]["value"]
        if "Cs" in params:
            Cs = params["Cs"]["value"]
        if "temp" in params:
            temp = float(params["temp"]["value"])
        if "ga_in" in params:
            ga_in = float(params["ga_in"]["value"])
        if "ga_out" in params:
            ga_out = float(params["ga_out"]["value"])
        if "Ex" in params:
            Ex = float(params["Ex"]["value"])
        if "eps_xy" in params:
            eps_xy = float(params["eps_xy"]["value"])
        if "B_dc" in params:
            B_dc = float(params["B_dc"]["value"])
        if "B_ac" in params:
            B_ac = float(params["B_ac"]["value"])
        if "theta_dc" in params:
            theta_dc = float(params["theta_dc"]["value"])
        if "theta_ac" in params:
            theta_ac = float(params["theta_ac"]["value"])
        if "C_max" in params:
            C_max = float(params["C_max"]["value"])
        if "F_aux" in params:
            F_aux = float(params["F_aux"]["value"])
        if "measurement" in params:
            measurement=params["measurement"]["value"]
        if "control" in params:
            control = params["control"]["value"]
        if "data_transfer" in params:
            data_transfer = params["data_transfer"]["value"]
        return {
            "msg_type": "param_set_response"
            }

    @qsi.on_message("channel_query")
    def channel_query(msg):
        
        global center
        global spin_gate
        global phase_gate
        global k
        global dc
        global d0
        global spin_states
        global Cs
        global temp
        global ga_in
        global ga_out
        global Ex
        global eps_xy
        global B_dc
        global B_ac
        global theta_dc
        global theta_ac
        global C_max
        global F_aux
        global measurement
        global control
        global data_transfer
        global previous_time
        
        
        # Kraus-operators and approximation error are evaluated by calling the function component
        operators,e=component(center,spin_gate,phase_gate,k,dc,d0,Cs,spin_states,temp,ga_in,ga_out,Ex,eps_xy,B_dc,B_ac,theta_dc,theta_ac,C_max,F_aux,measurement,control,data_transfer).channel()

        
        return {
            "msg_type" : "channel_query_response",
            "kraus_operators" : [ numpy_to_json(op) for op in operators],
            "kraus_state_indices" : [str(state_uuid)],
            "error" : e,
            "retrigger" : False
            }

    qsi.run()
