"""
 * Module Name: readout
 * Description: Kraus-operators for readout process
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import numpy as np
from kraus_operators import get_kraus_single

def readout(Ks_rot):
    
    '''
    
    * Function: readout
    
    * Purpose: Kraus-operators for readout process
    
    * Parameters:
        
        Ks_rot (list of arrays of complex): Kraus-operators for the spin pi/2 rotation
    
    * Returns:
        
        (list of arrays of complex): Kraus-operators for readout process
    
    '''
    
    Ks_rot=[np.kron(np.eye(2),Ks_rot[i]) for i in range(len(Ks_rot))]
    
    I=np.eye(2)
    rhos=[]
    for i in range(2):
        ei=I[i,:].reshape(2,1)
        for j in range(2):
            ej=I[j,:].reshape(2,1)
            rho_s=np.dot(ei,ej.T)
            rho_ph=1/2*np.array([[1,1],[1,1]])
            Ue=np.diag([-1,1,1,1])
            Ul=np.diag([1,1,-1,1])
        
            rho0=np.kron(rho_ph,rho_s)
            rho_er=np.dot(Ue,np.dot(rho0,Ue))
            rho_rot=sum([np.dot(Ks_rot[i],np.dot(rho_er,Ks_rot[i].T.conj())) for i in range(4)])
            rho_lr=np.dot(Ul,np.dot(rho_rot,Ul))
        
            down=np.kron(np.eye(2),np.array([1,0]).reshape(2,1))

            rho_md=np.dot(down.T,np.dot(rho_lr,down))
            
            rhos.append(rho_md)
            
    
    Ks=get_kraus_single(rhos)

    return Ks