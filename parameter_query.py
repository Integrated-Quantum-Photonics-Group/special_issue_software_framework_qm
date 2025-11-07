"""
 * Module Name: parameter_query
 * Description: make parameter choice interactive
 * Author: Yannick Strocka
 * Created On: October 30, 2025
 * Last Modified: October 31, 2025
 * Version: 1.0
"""
import numpy as np
from encode_parameters import encode_parameters

def parameter_query(spin_gate,phase_gate,control):
    
    """
    
    * Parameters:
    
    spin_gate (string) : experimental or optimized
    phase_gate (string): experimental or optimized
    control (string): mw or opt

    * Raises:

    ValueError
        DESCRIPTION.

    * Returns
    
    Cs (string): json files of the cooperativities if you chose control=opt and phase_gate=experimental
    rhos (string): json files of the propagated states if you chose spin_gate=experimental
    k (float): HWHM cavity loss rate (GHz)
    dc (float): cavity mode frequency detuning from the defect center's transition frequency (GHz)
    d0 (float): incoming mode frequency detuning from the defect center's transition frequency (GHz)
    center (string): color center ("SiV" or "SnV")
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
    data_transfer (string): readin or readout
    
    """
    
    if control=="mw":
        center = input('\n SiV or SnV? ("SiV" or "SnV"): ').strip()
        B_dc = float(input("\n B_dc/T = "))
        B_ac = float(input("\n B_ac/T = "))
        theta_dc = float(input("\n theta_dc/rad = "))
        theta_ac = float(input("\n theta_ac/rad = "))
        Ex = float(input("\n Ex = "))
        eps_xy = float(input("\n eps_xy = "))
        
    else:
        print('\n Current version of the software library only supports optical control for the SnV')
        center="SnV"
        B_dc = float(input("\n Choose between B_dc=0.3,1.0,3.0 T = "))
        B_ac=0
        theta_dc=0
        theta_ac=0
        Ex=0
        eps_xy=0
    
    
    if spin_gate == "optimized" and phase_gate=="experimental" and control=="opt":
        
        # Benutzer kann Parameter eingeben:
        print('\n')
        C1A = float(input("C1A = "))
        C2B = float(input("C2B = "))
        C2A = float(input("C2A = "))
        C1B = float(input("C1B = "))
        k=float(input("Cavity loss rate (HWHM) k/GHz= "))
        dc=float(input("Cavity mode detuning dc/GHz=(w_1A-w_c)/GHz= "))
        d0=float(input("Incoming mode detuning d0/GHz=(w0-w1A)/GHz= "))
        
        Cs, rhos = encode_parameters(C1A, C2B, C2A, C1B, np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    
    elif spin_gate == "optimized" and phase_gate=="experimental" and control=="mw":
        
        # Benutzer kann Parameter eingeben:
        k=float(input("Cavity loss rate (HWHM)k/GHz= "))
        dc=float(input("Cavity mode detuning dc/GHz=(w_1A-w_c)/GHz= "))
        d0=float(input("Incoming mode detuning d0/GHz=(w_1A-w0)/GHz= "))
        
        Cs, rhos = encode_parameters(0,0,0,0, np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    
    
    
    elif spin_gate=="experimental" and phase_gate=="optimized":
        
        k=0
        dc=0
        d0=0
        # Beispiel: Matrizenwerte (hier einfache Eingabe, könnte auch aus Datei oder vordefiniert kommen)
        print("\n Insert matrices as 1,0;0,1 for identity for example:")
        print("\n Initial states are rho00(0)=1,0;0,0, rho11(0)=0,0;0,1, rho_p(0)=0.5,0.5;0.5,0.5, rho_pi(0)=0.5,-0.5i,0.5i,0.5:")
        
        def eingabe_matrix(name):
            vals = input(f"{name} = ").replace(";", " ").replace(",", " ").split()
            vals = list(map(complex, vals))
            if len(vals) != 4:
                raise ValueError("Just type in 4 numbers (2x2 Matrix).")
            return np.array(vals).reshape(2, 2)
        
        rho00 = eingabe_matrix("rho00")
        rho11 = eingabe_matrix("rho11")
        rho_p = eingabe_matrix("rho_p")
        rho_pi = eingabe_matrix("rho_pi")
        
        Cs, rhos = encode_parameters(0,0,0,0, rho00, rho11, rho_p, rho_pi)
    
    elif spin_gate=="experimental" and phase_gate=="experimental":
        
        k=float(input("Cavity loss rate (HWHM) k/GHz= "))
        dc=float(input("Cavity mode detuning dc/GHz=(w_1A-w_c)/GHz= "))
        d0=float(input("Incoming mode detuning d0/GHz=(w_1A-w0)/GHz= "))
        
        print("\n Insert matrices as 1,0;0,1 for identity for example:")
        print("\n Initial states are rho00(0)=1,0;0,0, rho11(0)=0,0;0,1, rho_p(0)=0.5,0.5;0.5,0.5, rho_pi(0)=0.5,-0.5i,0.5i,0.5:\n")
        def eingabe_matrix(name):
            vals = input(f"{name} = ").replace(";", " ").replace(",", " ").split()
            vals = list(map(complex, vals))
            if len(vals) != 4:
                raise ValueError("Just type in 4 numbers (2x2 matrix).")
            return np.array(vals).reshape(2, 2)
        
        rho00 = eingabe_matrix("rho00")
        rho11 = eingabe_matrix("rho11")
        rho_p = eingabe_matrix("rho_p")
        rho_pi = eingabe_matrix("rho_pi")
        
        Cs, rhos = encode_parameters(0,0,0,0, rho00, rho11, rho_p, rho_pi)
    
    else:
            Cs, rhos = encode_parameters(0, 0, 0, 0, np.eye(2), np.eye(2), np.eye(2), np.eye(2))
            k=0
            dc=0
            d0=0
    
    
    temp = float(input("\n Choose a temperature between 0.1 K and 4 K (without a unit): "))
    ga_in = float(input("\n ga_in/THz = "))
    ga_out = float(input("\n ga_out/THz = "))
    F_aux = float(input("\n F_aux = "))
    C_max = float(input("\n C_max = "))
    data_transfer = input("\n data_transfer (readin or readout): ").strip()
    measurement=input("\n measurement (+/- for readin, down/up for readout): ").strip()
    
    
    
    
    
    return Cs,rhos,k,dc,d0,center,temp,ga_in,ga_out,Ex,eps_xy,B_dc,B_ac,theta_dc,theta_ac,F_aux,C_max,measurement,data_transfer