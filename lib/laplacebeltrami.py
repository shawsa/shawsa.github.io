import numpy as np
from numpy.linalg import cond, norm, inv

import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, lsqr
from scipy.spatial import cKDTree

from rbf import *

# Tangent Plane Method
'''
# Weighted distance function
def dist_outerW(nodes1, nodes2, W):
    d = len(nodes1[0]) # the dimension of each vector
    n1 = len(nodes1)
    n2 = len(nodes2)
    # create a row vector of d-dimensional vectors
    row = nodes1.reshape((1,n1,d)) 
    # create a column vector of d-dimensional vectors
    col = nodes2.reshape((n2,1,d)) 
    diff = row-col
    ret = diff @ W * diff
    #ret = (row-col)**2
    ret = np.sum(ret,2) #sum each d-dimensional vector
    return np.sqrt(ret)

def TPM_old(nodes, normals, rbf_obj=rbf_dict['multiquadric'], epsilon=None, stencil_size=15):
    n = len(nodes)
    k = stencil_size
    rbf = rbf_obj['rbf']
    zeta  = rbf_obj['zeta']
    d2rbf = rbf_obj['d2rbf']
    Lrbf = lambda r,epsilon: 1*zeta(r,epsilon) + d2rbf(r,epsilon)

    tree = cKDTree(np.array(nodes))

    weights = np.zeros((n, stencil_size))
    row_index = [r for r in range(n) for c in range(stencil_size)]
    col_index = np.zeros((n, stencil_size))

    e1, e2, e3 = np.eye(3)
    E = np.eye(3)
    E[2,2] = 0
    
    for i, (node, normal) in enumerate(zip(nodes, normals)):
        t1 = e2 - np.dot(normal, e2)*normal
        t1 /= la.norm(t1)
        t2 = e3 - np.dot(normal, e3)*normal - np.dot(t1, e3)*t1
        t2 /= la.norm(t2)
        
        R = np.zeros((3,3))
        R[:,0] = t1
        R[:,1] = t2
        R[:,2] = normal
        W = R @ E @ R.T

        stencil = tree.query(nodes[i], k)[1]
        col_index[i] = stencil
        nn = np.array([nodes[i] for i in stencil])

        if epsilon==None:
            dist_mat = dist_outerW(nn, nn, W)
            epsilon = optimize_eps(rbf, dist_mat, P=None, target_cond=10**12)
            print('epsilon = %g' %epsilon)

        A = rbf(dist_outerW(nn, nn, W), epsilon)
        if i==0:
            print('cond(A): %g' % cond(A))
        rhs = Lrbf(dist_outerW(nn,nn[0].reshape((1,3)), W), epsilon)
        weights[i] = la.solve(A, rhs.flatten())

    C = sp.csc_matrix((weights.ravel(), (row_index, col_index.ravel())),shape=(n,n))
    return C
'''

def TPM(nodes, normals, rbf_obj=rbf_dict['multiquadric'], epsilon=None, stencil_size=15, poly_deg=None):
    n = len(nodes)
    k = stencil_size
    rbf = rbf_obj['rbf']
    zeta  = rbf_obj['zeta']
    d2rbf = rbf_obj['d2rbf']
    Lrbf = lambda r,epsilon: zeta(r,epsilon) + d2rbf(r,epsilon)

    tree = cKDTree(np.array(nodes))

    weights = np.zeros((n, stencil_size))
    row_index = [r for r in range(n) for c in range(stencil_size)]
    col_index = np.zeros((n, stencil_size))

    e1, e2, e3 = np.eye(3)
    E = np.eye(3)
    E[2,2] = 0
    
    for i, (node, normal) in enumerate(zip(nodes, normals)):
        t1 = e2 - np.dot(normal, e2)*normal
        t1 /= la.norm(t1)
        t2 = e3 - np.dot(normal, e3)*normal - np.dot(t1, e3)*t1
        t2 /= la.norm(t2)
        
        R = np.zeros((3,2))
        R[:,0] = t1
        R[:,1] = t2

        stencil = tree.query(node, k)[1]
        col_index[i] = stencil
        nn = np.array([nodes[i] for i in stencil])

        nn = nn @ R
        scale = np.max(np.abs(nn))
        nn /= scale
        
        dist_mat = dist_outer(nn, nn)
        if not poly_deg is None:
            terms = (poly_deg+1)*(poly_deg+2)//2
            P = np.ones((k, terms))
            x = nn[:,0]
            y = nn[:,1]
            p1, p2 = 0, 1
            for d in range(1,poly_deg+1):
                for j in range(d):
                    P[:,p2+j] = P[:,p1+j]*x
                P[:,p2+d] = P[:,p1+j]*y
                p1, p2 = p2, p2+d+1
            if epsilon==None:
                epsilon = optimize_eps(rbf, dist_mat, P=P, target_cond=10**12)
                print('epsilon = %g' % epsilon)
            A = rbf(dist_mat, epsilon)
            A = np.block([[A, P],[P.T, np.zeros((terms, terms))]])
            rhs = np.zeros(k+terms)
            rhs[:k] = Lrbf(dist_mat[0], epsilon)
            if poly_deg >=2:
                rhs[k+3] = 2
                rhs[k+5] = 2
            rhs /= scale**2
            weights[i] = la.solve(A, rhs.flatten())[:k]
        else:
            if epsilon==None:
                epsilon = optimize_eps(rbf, dist_mat, P=None, target_cond=10**12)
                print('epsilon = %g' % epsilon)
            A = rbf(dist_mat, epsilon)
            rhs = Lrbf(dist_mat[0], epsilon) / scale**2
            weights[i] = la.solve(A, rhs.flatten())
        if i==0:
            print('cond(A): %g' % cond(A))

    C = sp.csc_matrix((weights.ravel(), (row_index, col_index.ravel())),shape=(n,n))
    return C

# Shankar-Wright Method
#def SWM(nodes, normals, rbf_obj=rbf_dict['multiquadric'], epsilon=None, stencil_size=15, poly_deg=None, poly_type):











