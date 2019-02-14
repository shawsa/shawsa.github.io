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
    phi1  = rbf_obj['phi1']
    d2rbf = rbf_obj['d2rbf']
    Lrbf = lambda r,epsilon: 1*phi1(r,epsilon) + d2rbf(r,epsilon)

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
    phi1  = rbf_obj['phi1']
    d2rbf = rbf_obj['d2rbf']
    Lrbf = lambda r,epsilon: phi1(r,epsilon) + d2rbf(r,epsilon)

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

def grad_rbf_outer(nodes, centers, phi1, epsilon):
    n_len = len(nodes)
    c_len = len(centers)
    r = dist_outer(nodes, centers)[:,:,np.newaxis]
    xs = (np.array(nodes).reshape((1,n_len,3)) - np.array(centers).reshape((c_len,1,3)))
    return phi1(r, epsilon) * xs

def SWM(nodes, normals, rbf_obj=rbf_dict['multiquadric'], epsilon=None, 
        stencil_size=15, poly_deg=None, poly_type='s'):

    assert poly_type in 'ps'
    n = len(nodes)
    k = stencil_size
    rbf = rbf_obj['rbf']
    phi1 = rbf_obj['phi1']
    
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
        rhsAs = np.matmul(nn_proj, grad_rbf_outer(nn, nn, phi1, epsilon).reshape(
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
            weights_grad = la.solve(A, rhs).T
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

#######################################################
#
# Symmetric Orthogonal Gradients
#
#######################################################
def SOGr(nodes, normals, rbf_obj=rbf_dict['multiquadric'], eps=None, 
        stencil_size=15, poly_deg=None, poly_type='p'):

    assert poly_type is 'p'
    
    n = len(nodes)
    k = stencil_size
    phi  = rbf_obj['rbf']
    phi1 = rbf_obj['phi1']
    phi2 = rbf_obj['phi2']
    phi3 = rbf_obj['phi3']
    phi4 = rbf_obj['phi4']
    def Gx(r, d, eps):
        return d*phi1(r,eps)
    def Hx(r, d,eps):
        return phi1(r,eps) + d**2 * phi2(r,eps)
    def GHx(r, d,eps):
        return 3*d*phi2(r,eps) + d**3 * phi3(r,eps)
    def HHx(r, d,eps):
        return 3*phi2(r,eps) + 6 * d**2 * phi3(r,eps) + d**4 * phi4(r,eps)
    def Lphi(r, eps):
        return 3*phi1(r,eps) + r**2 * phi2(r,eps)
    def LG(r, d,eps):
        return 5*d*phi2(r,eps) + r**2 * d * phi3(r,eps)
    def LH(r,d,eps):
        return 5*phi2(r,eps) + (r**2 + 7*d**2)*phi3(r,eps) + (d*r)**2 * phi4(r,eps)
    
    weights = np.zeros((n, stencil_size))
    row_index = [r for r in range(n) for c in range(stencil_size)]
    col_index = np.zeros((n, stencil_size))
    
    tree = cKDTree(np.array(nodes))
    
    for i, node in enumerate(nodes):
        stencil = tree.query(nodes[i], k)[1]
        col_index[i] = stencil
        nn = np.array([nodes[i] for i in stencil])
        # center stencil
        nn -= nn[0]
        # scale stencil
        scale = np.max(np.abs(nn))
        nn /= scale
        
        r = dist_outer(nn,nn)
        
        if i==0 and eps is None and rbf_obj['shape']:
            eps = optimize_eps(phi, r, P=None, target_cond=10**12)
            print('epsilon set: %g' % eps)

        d = nn @ normals[i]

        if poly_deg is None:
            A = np.zeros((k+2,k+2))

            A[:k, :k] = phi(r, eps)
            A[:k, -2] = -Gx(r[0], d, eps)/scale
            A[:k, -1] = Hx(r[0], d, eps)/scale**2
            A[-2, -2] = -Hx(0, 0, eps)/scale**2
            A[-2, -1] = -GHx(0, 0, eps)/scale**3
            A[-1, -1] = HHx(0, 0, eps)/scale**4
            A[-2, :k] = A[:k, -2]
            A[-1, :-1] = A[:-1, -1]

            B = np.zeros(k+2)
            B[:k] = Lphi(r[0], eps)/scale**2
            B[-2] = LG(0, 0, eps)/scale**3
            B[-1] = LH(0, 0, eps)/scale**4

            weights[i] = la.solve(A, B)[:k]
        else:
            P, *trash = grad_poly(nn, [], poly_deg)
            terms = P.shape[1]
            
            A = np.zeros((k+2+terms,k+2+terms))

            B = np.zeros(k+2+terms)
            B[:k] = Lphi(r[0], eps)/scale**2
            B[k] = LG(0, 0, eps)/scale**3
            B[k+1] = LH(0, 0, eps)/scale**4
            
            A[:k, k+2:] = P
            A[k+2:, :k] = P.T

            if poly_deg >= 1:
                # Lp
                A[k, k+3:k+6] = normals[i]/scale
                A[k+3:k+6, k] = normals[i]/scale
            if poly_deg >= 2:
                # Hp
                nHn = 2*np.outer(normals[i], normals[i])/scale**2
                A[k+1, k+6:k+9] = nHn[0]
                A[k+1, k+9:k+11] = nHn[1, 1:]
                A[k+1, k+11] = nHn[2,2]
                A[k+6:k+12, k+1] = A[k+1, k+6:k+12]
                
                B[k+6] = 2/scale**2
                B[k+9] = 2/scale**2
                B[k+11] = 2/scale**2

            A[:k,  :k] = phi(r, eps)
            A[:k,   k] = -Gx(r[0], d, eps)/scale
            A[:k, k+1] = Hx(r[0], d, eps)/scale**2

            A[ k, k] = -Hx(0, 0, eps)/scale**2
            A[k, k+1] = -GHx(0, 0, eps)/scale**3
            A[k+1, k+1] = HHx(0, 0, eps)/scale**4

            A[k, :k] = A[:k, k]
            A[k+1, :k+1] = A[:k+1, k+1]
            
            try:
                weights[i] = schur_solve(A[:k+2,:k+2], A[:k+2, k+2:], B[:k+2], B[k+2:])[0][:k]
            except:
                weights[i] = la.solve(A, B)[:k]

    C = sp.csc_matrix((weights.ravel(), (row_index, col_index.ravel())),shape=(n,n))
    return C



