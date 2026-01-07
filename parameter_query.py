"""
 * Module Name: parameter_query
 * Description: make parameter choice interactive
 * Author: Yannick Strocka
 * Created On: October 30, 2025
 * Last Modified: October 31, 2025
 * Version: 1.0
"""
import numpy as np
import math
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
        if center!="SiV" and center!='SnV':
            raise ValueError("Input must be SiV or SnV")
            
        B_dc = float(input("\n B_dc/T = "))
        if not math.isfinite(B_dc) or B_dc < 0:
            raise ValueError("B_dc must be a finite, non-negative number (Tesla).")
        
        B_ac = float(input("\n B_ac/T = "))
        if not math.isfinite(B_ac) or B_ac < 0:
            raise ValueError("B_ac must be a finite, non-negative number (Tesla).")
        
        theta_dc = float(input("\n theta_dc/rad = "))
        if not math.isfinite(theta_dc) or not (0 <= theta_dc <= math.pi/2):
            raise ValueError("theta_dc must be between 0 and π/2 radians.")
        
        theta_ac = float(input("\n theta_ac/rad = "))
        if not math.isfinite(theta_ac) or not (0 <= theta_ac <= math.pi/2):
            raise ValueError("theta_ac must be between 0 and π/2 radians.")
        
        Ex = float(input("\n Ex = "))
        if not math.isfinite(Ex):
            raise ValueError("Ex must be a finite real number.")
        
        eps_xy = float(input("\n eps_xy = "))
        if not math.isfinite(eps_xy):
            raise ValueError("eps_xy must be finite.")
        
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
        
        
        print('\n')
        C1A = float(input("C1A = "))
        if not math.isfinite(C1A) or C1A < 0:
            raise ValueError("C1A must be a finite, non-negative real number.")
        
        C2B = float(input("C2B = "))
        if not math.isfinite(C2B) or C2B < 0:
            raise ValueError("C2B must be a finite, non-negative real number.")
        
        C2A = float(input("C2A = "))
        if not math.isfinite(C2A) or C2A < 0:
            raise ValueError("C2A must be a finite, non-negative real number.")
        
        C1B = float(input("C1B = "))
        if not math.isfinite(C1B) or C1B < 0:
            raise ValueError("C1B must be a finite, non-negative real number.")
        
        k = float(input("Cavity loss rate (HWHM) k/GHz = "))
        if not math.isfinite(k) or k <= 0:
            raise ValueError("k must be a finite, strictly positive number (loss rate).")
        
        dc = float(input("Cavity mode detuning dc/GHz = (w_1A - w_c)/GHz = "))
        if not math.isfinite(dc):
            raise ValueError("dc must be a finite real number (can be positive or negative).")
        
        d0 = float(input("Incoming mode detuning d0/GHz = (w0 - w1A)/GHz = "))
        if not math.isfinite(d0):
            raise ValueError("d0 must be a finite real number (can be positive or negative).")
        
        Cs, rhos = encode_parameters(C1A, C2B, C2A, C1B, np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    
    elif spin_gate == "optimized" and phase_gate=="experimental" and control=="mw":
        
        # Benutzer kann Parameter eingeben:
        k = float(input("Cavity loss rate (HWHM) k/GHz = "))
        if not math.isfinite(k) or k <= 0:
            raise ValueError("k must be a finite, strictly positive number (loss rate).")
        
        dc = float(input("Cavity mode detuning dc/GHz = (w_1A - w_c)/GHz = "))
        if not math.isfinite(dc):
            raise ValueError("dc must be a finite real number (can be positive or negative).")
        
        d0 = float(input("Incoming mode detuning d0/GHz = (w0 - w1A)/GHz = "))
        if not math.isfinite(d0):
            raise ValueError("d0 must be a finite real number (can be positive or negative).")
        
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
        
        k = float(input("Cavity loss rate (HWHM) k/GHz = "))
        if not math.isfinite(k) or k <= 0:
            raise ValueError("k must be a finite, strictly positive number (loss rate).")
        
        dc = float(input("Cavity mode detuning dc/GHz = (w_1A - w_c)/GHz = "))
        if not math.isfinite(dc):
            raise ValueError("dc must be a finite real number (can be positive or negative).")
        
        d0 = float(input("Incoming mode detuning d0/GHz = (w0 - w1A)/GHz = "))
        if not math.isfinite(d0):
            raise ValueError("d0 must be a finite real number (can be positive or negative).")
        
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
    if not math.isfinite(temp) or not (0.1 <= temp <= 4.0):
        raise ValueError("Temperature must be between 0.1 K and 4 K.")
    
    ga_in = float(input("\n ga_in/THz = "))
    if not math.isfinite(ga_in) or ga_in < 0:
        raise ValueError("ga_in must be a finite, non-negative number (THz).")
    
    ga_out = float(input("\n ga_out/THz = "))
    if not math.isfinite(ga_out) or ga_out < 0:
        raise ValueError("ga_out must be a finite, non-negative number (THz).")
    
    F_aux = float(input("\n F_aux = "))
    if F_aux > 1 or F_aux < 0:
        raise ValueError("F_aux must be below 1 and a non-negative number.")
    
    C_max = float(input("\n C_max = "))
    if not math.isfinite(C_max) or C_max < 0:
        raise ValueError("C_max must be a finite, non-negative number.")
    
    data_transfer = input("\n data_transfer (readin or readout): ").strip().lower()
    if data_transfer not in {"readin", "readout"}:
        raise ValueError("data_transfer must be either 'readin' or 'readout'.")
    
    measurement = input("\n measurement (+/- for readin, down/up for readout): ").strip().lower()
    
    if data_transfer == "readin":
        if measurement not in {"+", "-"}:
            raise ValueError("For readin, measurement must be '+' or '-'.")
    elif data_transfer == "readout":
        if measurement not in {"down", "up"}:
            raise ValueError("For readout, measurement must be 'down' or 'up'.")
    
    
    
    
    
    return Cs,rhos,k,dc,d0,center,temp,ga_in,ga_out,Ex,eps_xy,B_dc,B_ac,theta_dc,theta_ac,F_aux,C_max,measurement,data_transfer