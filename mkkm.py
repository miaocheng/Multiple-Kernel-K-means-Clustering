# --------------------------------------------------------------------------------------------------
# This python file contains the implementation of MKKM algorithm for multi-view clustering
# 
# Reference:
# S. Jegelka, A. Gretton, B. Scholkopf, and et al., Generalized clustering via kernel embedding,
# in KI: Advances in Artificial Intelligence, Annual German Conference on AI, 2009.
# X. Liu, X. Zhu, M. Li, L. Wang, and et al., Multiple Kernel k-means with Incomplete Kernels,
# IEEE Trans. PAMI, 2018.
# 
# Coded by Miao Cheng
# Date: 2020-02-21
# All rights reserved
# --------------------------------------------------------------------------------------------------
import numpy as np
from numpy import linalg as la
from qpsolvers import solve_qp

from cala import *
from kernel import *
from cmetrics import *


def cmeasure(Label, labels):
    # +++++ Normalized Mutual Information +++++
    print('The Normalized Mutual Information:')
    A, nmi, avg = calNMI(Label, labels)
    str = 'The obtained NMI: %f' %nmi + '\navg: %f' %avg + '\n'
    print(str)
    
    # +++++ Accuracy +++++
    print('The Accuracy Measure:')
    accuracy, acc = Accuracy(Label, labels)
    str = 'The obtained Accuarcy: %f' %accuracy + '\n'
    print(str)
    
    # +++++ f +++++
    print('The F Measure:')
    f, p, r = calF(Label, labels)
    str = 'The obtained f: %f' %f
    print(str)
    str = 'The obtained p: %f' %p
    print(str)
    str = 'The obtained r: %f\n' %r
    print(str)
    
    
    # +++++ RandIndex +++++
    print('The RandIndex Measure:')
    ar, ri, MI, HI = RandIndex(Label, labels)
    str = 'The obtained ar: %f' %ar
    print(str)
    str = 'The obtained ri: %f' %ri
    print(str) 
    str = 'The obtained MI: %f' %MI
    print(str)
    str = 'The obtained HI: %f\n' %HI
    print(str)
    
    
    return nmi, avg, acc, f, p, r, ar, ri, MI, HI


def mkkm(Fea, c, nIter):
    nFea = len(Fea)
    tmx = Fea[0]
    nDim, nSam = np.shape(tmx)
    
    K = []
    # +++++ Calculate the kernel matrices +++++
    for i in range(nFea):
        tmx = Fea[i]
        tmk = Kernel(tmx, tmx, 'Gaussian')
        K.append(tmk)
        
    # +++++ Initialize the parameters +++++
    beta = np.ones((nFea, 1))
    beta = ( float(1) / nFea ) * beta
    beta = beta[:, 0]
    
    obj = 1e8
    H = np.ones((nSam, c))
    # +++++ Iterative Optimization +++++
    for ii in range(nIter):
        tk = np.zeros((nSam, nSam))
        # +++++ Update the kernels +++++
        for i in range(nFea):
            tmk = K[i]
            alpha = beta[i] ** 2
            tmk = alpha * tmk
            
            tk = tk + tmk
            
        # +++++ Calculate the H +++++
        old_H = H
        
        #s, V = la.eig(tk)
        V, s, _ = la.svd(tk)
        
        ss, r = getRank(s)
        assert c <= r, 'The input parameter is incorrect !'
        s = s[0:c]
        V = V[:, 0:c]
        H = V
        
        # +++++ Update the beta +++++
        tmh = np.dot(H, np.transpose(H))
        I = np.eye(nSam)
        tm = I - tmh
        
        P = []
        for i in range(nFea):
            tmk = K[i]
            tmp = np.dot(tmk, tm)
            tmr = np.trace(tmp)
            
            P.append(tmr)
            
            
        P = np.diag(P)
        
        q = np.zeros((nFea, 1)).reshape((nFea, ))
        G = np.zeros((nFea, nFea))
        for i in range(nFea):
            G[i, i] = -1
            
        h = np.zeros((nFea, 1)).reshape((nFea, ))
        A = np.ones((nFea, ))
        b = 1
        
        beta = solve_qp(P, q, G, h, A, b)
        
        # +++++ Calculate the new Obj +++++
        tmp = H - old_H
        obj = norm(tmp, 2)
        
        if obj < 1e-7:
            break
        
    # +++++ Generate the clustered group +++++
    B, Index = iMax(H, axis=1)
    clusters = Index + 1
    
    
    return clusters, H
    
    
    
    
    
    
        
        
        
        
        
    
    
        
    
