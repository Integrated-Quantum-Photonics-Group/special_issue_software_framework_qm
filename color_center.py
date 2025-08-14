"""
 * Module Name: color_center
 * Description: interaction interface between server and client
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
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
            
            F_gen (float): fidelity of the incoming photon
            bw (float): bandwidth of the incoming photon
            temp (float): temperature
            Ex (float): strain ground state
            eps_xy (float): shear strain
            B_dc (float): DC magnetic field strength
            B_ac (float): AC magnetic field strength
            theta_dc (float): orientation of the DC field
            theta_ac (float): orientation of the AC field
            control (string): opt (optical control) or mw (microwave control)
            data_transfer (string): readin or readout
         
        """
        return {
            "msg_type": "param_query_response",
            "params": {
                "F_gen": "number",
                "bw": "number",
                "temp": "number",
                "Ex":"number",
                "eps_xy":"number",
                "B_dc":"number",
                "B_ac":"number",
                "theta_dc":"number",
                "theta_ac":"number",
                "control":"string",
                "data_transfer":"string",
                }
            }

    @qsi.on_message("param_set")
    def param_sest(msg):
        
        global F_gen
        global temp
        global bw
        global Ex
        global eps_xy
        global B_dc
        global B_ac
        global theta_dc
        global theta_ac
        global control
        global data_transfer
        
        params = msg["params"]
        if "F_gen" in params:
            F_gen = float(params["F_gen"]["value"])
        if "temp" in params:
            temp = float(params["temp"]["value"])
        if "bw" in params:
            bw = float(params["bw"]["value"])
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
        if "control" in params:
            control = params["control"]["value"]
        if "data_transfer" in params:
            data_transfer = params["data_transfer"]["value"]
        return {
            "msg_type": "param_set_response"
            }

    @qsi.on_message("channel_query")
    def channel_query(msg):
        
        global F_gen
        global temp
        global bw
        global Ex
        global eps_xy
        global B_dc
        global B_ac
        global theta_dc
        global theta_ac
        global control
        global data_transfer
        global previous_time
        
        
        # Kraus-operators and approximation error are evaluated by calling the function component
        operators,e=component(F_gen,temp,bw,Ex,eps_xy,B_dc,B_ac,theta_dc,theta_ac,control,data_transfer).channel()

        
        return {
            "msg_type" : "channel_query_response",
            "kraus_operators" : [ numpy_to_json(op) for op in operators],
            "kraus_state_indices" : [str(state_uuid)],
            "error" : e,
            "retrigger" : False
            }

    qsi.run()
