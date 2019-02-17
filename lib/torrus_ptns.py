import numpy as np
from scipy.optimize import fsolve

def torus_phyllotaxis_points(N):
    R, r = 1, 1/3
    gr = (1 + np.sqrt(5))/2
    gamma = 2*np.pi*(1- 1/gr)
    pmax = 2*np.pi
    caparea = lambda x: 2*np.pi*(3*x + np.sin(x))/9
    c = N/caparea(pmax)
    t = np.arange(1,N+1)*gamma
    p = np.zeros(N)
    p[0] = fsolve(lambda x: caparea(x)-1/c, 0)
    for i in range(1,N):
        p[i] = fsolve(lambda x: caparea(x)-i/c, p[i-1])
    nodes = np.zeros((N, 3))
    nodes[:, 0] = (R + r*np.cos(p)) * np.cos(t)
    nodes[:, 1] = (R + r*np.cos(p)) * np.sin(t)
    nodes[:, 2] = r*np.sin(p)

    normals = np.zeros((N,3))
    normals[:, 0] = r*np.cos(p)*np.cos(t)*(R+r*np.cos(p))
    normals[:, 1] = r*np.cos(p)*np.sin(t)*(R+r*np.cos(p))
    normals[:, 2] = r*np.sin(p) * (R + r*np.cos(p)) * np.cos(t)**2 + \
        r*np.sin(p) * (R+r*np.cos(p)) * np.sin(t)**2
    
    return nodes, normals

def get_parameters(x, R, r):
    theta = np.arctan2(x[:,1], x[:,0])
    phi = np.arctan2(x[:,2], np.sqrt(x[:,0]**2 + x[:,1]**2) - R)
    return theta, phi

def torus_to_cart(theta, phi, R, r):
    nodes = np.zeros((len(phi), 3))
    nodes[:,0] = (R + r*np.cos(phi))*np.cos(theta)
    nodes[:,1] = (R + r*np.cos(phi))*np.sin(theta)
    nodes[:,2] = r*np.sin(phi)
    return nodes

def torus_forcing(nodes):
    # I don't know what these are... maybe related to curvature?
    # they seem to affect the way the gaussians spread along each
    # axis of rotation
    a, b = 9, 3
    s = 0
    
    # create gaussian centers
    theta_cs = np.array([0, .5,   1,    2, 4, 5, 3.141])
    phi_cs   = np.array([0,  4, 1.5, -1.5, 0, 4, 3.141/2])
    centers = torus_to_cart(theta_cs, phi_cs, 1, 1/3)
    
    thetas, phis = get_parameters(nodes, 1, 1/3)
    
    N = len(nodes)
    K = len(centers)
    
    us = np.zeros(N)
    fs = np.zeros(N)
    
    cth, sth = np.cos(thetas), np.sin(thetas)
    for k in range(K):
        slak = np.sin(phis - phi_cs[k])
        clak = np.cos(phis - phi_cs[k])
        sthk = np.sin(thetas - theta_cs[k])
        cthk = np.cos(thetas - theta_cs[k])
        
        uk = np.exp(-a**2*(1-clak) - b**2*(1-cthk))
        C = a**4*slak**2-9*b**2*cthk-a**2*clak+9*b**4*sthk**2-6*b**2*cthk*cth+ \
               3*b**2*sthk*sth-b**2*cthk*cth**2+6*b**4*sthk**2*cth+b**4*sthk**2*cth**2+ \
               b**2*sthk*cth*sth
        fs += (-s - 9*C/(cth+3)**2)*uk
        us += uk
    return us, fs
