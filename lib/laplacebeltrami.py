import numpy as np
from numpy.linalg import cond, norm, inv

import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, lsqr
from scipy.spatial import cKDTree

from rbf import *
from poly_basis import *

from math import factorial as fac

#######################################################
#
# Tangent Plane Method
#
#######################################################
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
            if epsilon is None and rbf_obj['shape']:
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
            if epsilon is None and rbf_obj['shape']:
                epsilon = optimize_eps(rbf, dist_mat, P=None, target_cond=10**12)
                print('epsilon = %g' % epsilon)
            A = rbf(dist_mat, epsilon)
            rhs = Lrbf(dist_mat[0], epsilon) / scale**2
            weights[i] = la.solve(A, rhs.flatten())
        if i==0:
            print('cond(A): %g' % cond(A))

    C = sp.csc_matrix((weights.ravel(), (row_index, col_index.ravel())),shape=(n,n))
    return C

#######################################################
#
# Shankar-Wright Method
#
#######################################################
def schur_solve(A, P, f, g):
    lam = la.solve(P.T @ la.solve(A, P), P.T @ la.solve(A,f) - g)
    w = la.solve(A, f- P@lam)
    return w, lam

def grad_poly(nodes, projectors, deg):
    n = len(nodes)
    x = nodes[:,0]
    y = nodes[:,1]
    z = nodes[:,2]
    cols = fac(deg+3)//(fac(deg)*fac(3))
    P = np.zeros((n, cols))
    rhs_dx = np.zeros((cols, n))
    rhs_dy = np.zeros((cols, n))
    rhs_dz = np.zeros((cols, n))
    rhs_x = np.zeros((cols, n))
    rhs_y = np.zeros((cols, n))
    rhs_z = np.zeros((cols, n))
    i = 0
    for d in range(deg+1):
        #x^a y^b z^c with a+b+c = d
        for a in range(d,-1, -1):
            for b in range(d-a, -1, -1):
                c = d-a-b
                P[:,i] = x**a * y**b * z**c
                if a == 0:
                    rhs_dx[i] = 0
                else:
                    rhs_dx[i] = a * x**(a-1) * y**b * z**c
                if b == 0:
                    rhs_dy[i] = 0
                else:
                    rhs_dy[i] = b * x**a * y**(b-1) * z**c
                if c == 0:
                    rhs_dz[i] = 0
                else:
                    rhs_dz[i] = c * x**a * y**b * z**(c-1)
                i += 1
    for i, p in enumerate(projectors):
        rhs_x[:,i] = p[0,0]*rhs_dx[:,i] + p[1,0]*rhs_dy[:,i] + p[2,0]*rhs_dz[:,i]
        rhs_y[:,i] = p[0,1]*rhs_dx[:,i] + p[1,1]*rhs_dy[:,i] + p[2,1]*rhs_dz[:,i]
        rhs_z[:,i] = p[0,2]*rhs_dx[:,i] + p[1,2]*rhs_dy[:,i] + p[2,2]*rhs_dz[:,i]
    return P, rhs_x, rhs_y, rhs_z

def grad_rbf_outer(nodes, centers, zeta, epsilon):
    n_len = len(nodes)
    c_len = len(centers)
    r = dist_outer(nodes, centers)[:,:,np.newaxis]
    xs = (np.array(nodes).reshape((1,n_len,3)) - np.array(centers).reshape((c_len,1,3)))
    return zeta(r, epsilon) * xs

def SWM(nodes, normals, rbf_obj=rbf_dict['multiquadric'], epsilon=None, 
        stencil_size=15, poly_deg=None, poly_type='s'):
    n = len(nodes)
    k = stencil_size
    rbf = rbf_obj['rbf']
    zeta = rbf_obj['zeta']
    
    weights = np.zeros((n, stencil_size))
    row_index = [r for r in range(n) for c in range(stencil_size)]
    col_index = np.zeros((n, stencil_size))
    
    tree = cKDTree(np.array(nodes))
    projectors = [np.eye(3) - np.outer(node, node) for node in normals]
    
    for i, node in enumerate(nodes):
        stencil = tree.query(nodes[i], k)[1]
        col_index[i] = stencil
        nn = np.array([nodes[i] for i in stencil])
        nn_proj = np.array([projectors[i] for i in stencil])
        # center stencil
        nn -= nn[0]
        # scale stencil
        scale = np.max(np.abs(nn))
        nn /= scale        
        
        if poly_deg is None:
            P = None
        else:
            if poly_type == 'p':
                P, rhs_x, rhs_y, rhs_z = grad_poly(nn, nn_proj, poly_deg)
            elif poly_type == 's':
                P, rhs_x, rhs_y, rhs_z = gen_sphere_harm_basis(poly_deg, nn, nn_proj)
            rhs_x /= scale
            rhs_y /= scale
            rhs_z /= scale
        
        dist_mat = dist_outer(nn,nn)
        #print(cond(rbf(dist_mat, 1)))
        if i==0 and epsilon is None and rbf_obj['shape']:
            epsilon = optimize_eps(rbf, dist_mat, P=None, target_cond=10**12)
            print('epsilon set: %g' % epsilon)
            
        A = rbf(dist_mat, epsilon)
        rhsAs = np.matmul(nn_proj, grad_rbf_outer(nn, nn, zeta, epsilon).reshape(
                (stencil_size,stencil_size,3,1))).reshape((stencil_size,stencil_size,3))
        rhsAs /= scale
        
        if P is None:
            rhs = rhsAs[:,:,0] # only the x coordinates
            weights_grad = la.solve(A, rhs).T
            weights[i] = (weights_grad@weights_grad)[0]

            rhs = rhsAs[:,:,1] # only the y coordinates
            weights_grad = la.solve(A, rhs).T
            weights[i] += (weights_grad@weights_grad)[0]

            rhs = rhsAs[:,:,2] # only the z coordinates
            weights_grad = la.solve(A, rhs)[:stencil_size,:].T
            weights[i] += (weights_grad@weights_grad)[0]
        else:
            weights_grad = schur_solve(A, P, rhsAs[:,:,0], rhs_x)[0].T
            weights[i] = (weights_grad@weights_grad)[0]
            weights_grad = schur_solve(A, P, rhsAs[:,:,1], rhs_y)[0].T
            weights[i] += (weights_grad@weights_grad)[0]
            weights_grad = schur_solve(A, P, rhsAs[:,:,2], rhs_z)[0].T
            weights[i] += (weights_grad@weights_grad)[0]

    C = sp.csc_matrix((weights.ravel(), (row_index, col_index.ravel())),shape=(n,n))
    return C







