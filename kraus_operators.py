"""
 * Module Name: kraus_operators
 * Description: computation of kraus-operators 
 * Author: Yannick Strocka
 * Created On: May 19, 2025
 * Last Modified: May 19, 2025
 * Version: 1.0
"""

import numpy as np

def get_kraus_single(rhosT):
    
    '''
    
    * Function: get_kraus_single
    
    * Parameters:
        
        rhosT (list of arrays of complex): final states
        
    * Returns: 
        
        (list of arrays of complex): Kraus-operators
    
    '''

    rhos0=[]
    rhos0.append(np.array([[1,0],[0,0]],dtype=np.complex128))
    rhos0.append(np.array([[0,1],[0,0]],dtype=np.complex128))
    rhos0.append(np.array([[0,0],[1,0]],dtype=np.complex128))
    rhos0.append(np.array([[0,0],[0,1]],dtype=np.complex128))

    C=np.zeros((4,4),dtype=complex)
    for i in range(4):
        C=C+np.kron(rhosT[i],rhos0[i])

    la,v=np.linalg.eig(C)
    Ks=[]
    summe=np.zeros((2,2),dtype=complex)
    for i in range(4):
        K=np.sqrt(la[i])*np.reshape(v[:,i],(2,2))
        Ks.append(K)
        summe=summe+np.dot(K.T.conj(),K)

    for k in range(4):
        summe=np.zeros((2,2),dtype=complex)
        for i in range(4):
            summe=summe+np.dot(Ks[i],np.dot(rhos0[k],np.conjugate(Ks[i].T)))

    return Ks


