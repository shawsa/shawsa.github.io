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
    normals[:, 2] = r*np.sin(p) * (R + r*np.cos(p)) * np.cos(t)**2 + r*np.sin(p) * (R+r*np.cos(p))*np.sin(t)**2
    
    lengths = np.sqrt(normals[:,0]**2 + normals[:,1]**2 + normals[:,2]**2)
    
    normals[:,0] /= lengths
    normals[:,1] /= lengths
    normals[:,2] /= lengths
    
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
#     a, b = np.sqrt(4), np.sqrt(20)
    a, b = 2, 2*3
    
    # create gaussian centers
    theta_cs = np.array([0, .5,   1,    2, 4, 5, 3.141])
    phi_cs   = np.array([0,  4, 1.5, -1.5, 0, 4, 3.141/2])
    shapes = [1, .5, 2, .1, .7, .9, .3]
    
    thetas, phis = get_parameters(nodes, 1, 1/3)
    
    N, K = len(nodes), len(theta_cs)
    
    us = np.zeros(N)
    lap = np.zeros(N)
    
    ct, st = np.cos(thetas), np.sin(thetas)
    cp, sp = np.cos(phis), np.sin(phis)
    for k in range(K):
        s = shapes[k]
        spk = np.sin(phis - phi_cs[k])
        cpk = np.cos(phis - phi_cs[k])
        stk = np.sin(thetas - theta_cs[k])
        ctk = np.cos(thetas - theta_cs[k])
        
        uk = np.exp(-s* (a**2*(1-cpk) + b**2*(1-ctk))  )
        C = 1*a**4*s**2*spk**2*cp*ct + 3*a**4*s**2*spk**2*cp + \
            3*a**4*s**2*spk**2*ct + 9*a**4*s**2*spk**2 - 1*a**2*s*cp*ct*cpk \
            - 3*a**2*s*cp*cpk - 3*a**2*s*ct*cpk - 9*a**2*s*cpk \
            + b**4*s**2*stk**2 - b**2*s*ctk
        C /= (1+cp/3)**2
        lap += C*uk
        us += uk
    return us, lap
