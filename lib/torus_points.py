import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import Delaunay

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

fib_nums = [1, 1]
for i in range(50):
    fib_nums += [fib_nums[-1] + fib_nums[-2]]

def gen_torus_nodes(nodeset, n_try,):
    assert nodeset in ['phyllotaxis']

    if nodeset is 'phyllotaxis':
        i = np.argmin(np.abs([n_try - Ni for Ni in fib_nums]))
        n = fib_nums[i]
        nodes, normals = torus_phyllotaxis_points(n)
        return n, nodes, normals
    

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
    
    thetas, phis = get_parameters(nodes, 1, 1/3)
    
    N, K = len(nodes), len(theta_cs)
    
    us = np.zeros(N)
    lap = np.zeros(N)
    
    ct, st = np.cos(thetas), np.sin(thetas)
    cp, sp = np.cos(phis), np.sin(phis)
    for k in range(K):
        s = 1
        spk = np.sin(phis - phi_cs[k])
        cpk = np.cos(phis - phi_cs[k])
        stk = np.sin(thetas - theta_cs[k])
        ctk = np.cos(thetas - theta_cs[k])
        
        uk = np.exp(-s* (a**2*(1-cpk) + b**2*(1-ctk))  )
        C = 1*a**4*s**2*spk**2*cp**2 + 6*a**4*s**2*spk**2*cp + 9*a**4*s**2*spk**2 \
                + 1*a**2*s*sp*spk*cp + 3*a**2*s*sp*spk - 1*a**2*s*cp**2*cpk - 6*a**2*s*cp*cpk \
                - 9*a**2*s*cpk + b**4*s**2*stk**2 - b**2*s*ctk
        C /= (1+cp/3)**2
        lap += C*uk
        us += uk
    return us, lap

def torus_time(nodes, t):
    a, b = 2, 2*3
    
    # create gaussian centers
    theta_cs = np.array([0, .5,   1,    2, 4, 5, 3.141])
    phi_cs   = np.array([0,  4, 1.5, -1.5, 0, 4, 3.141/2])
    thetas, phis = get_parameters(nodes, 1, 1/3)
    N, K = len(nodes), len(theta_cs)
    us = np.zeros(N)
    lapu = np.zeros(N)
    ct, st = np.cos(thetas), np.sin(thetas)
    cp, sp = np.cos(phis), np.sin(phis)
    for k in range(K):
        s = 1
        spk = np.sin(phis - phi_cs[k])
        cpk = np.cos(phis - phi_cs[k])
        stk = np.sin(thetas - theta_cs[k])
        ctk = np.cos(thetas - theta_cs[k])
        
        uk = np.exp(-s* (a**2*(1-cpk) + b**2*(1-ctk))  )
        C = 1*a**4*s**2*spk**2*cp**2 + 6*a**4*s**2*spk**2*cp + 9*a**4*s**2*spk**2 \
                + 1*a**2*s*sp*spk*cp + 3*a**2*s*sp*spk - 1*a**2*s*cp**2*cpk - 6*a**2*s*cp*cpk \
                - 9*a**2*s*cpk + b**4*s**2*stk**2 - b**2*s*ctk
        C /= (1+cp/3)**2
        lapu += C*uk
        us += uk
    us *= np.exp(-t)
    lapu *= np.exp(-t)
    f = -us - lapu
    return us, f

def torus_triangulate(nodes):
    extend_range = .5
    #def torus_tri(nodes):
    theta, phi = get_parameters(nodes, 1, 1/3)
    nmap = np.arange(len(nodes))
    # extend theta
    ids = theta > np.pi-extend_range
    theta = np.concatenate((theta, theta[ids]-2*np.pi))
    phi = np.concatenate((phi, phi[ids]))
    nmap = np.concatenate((nmap, nmap[ids]))
    ids = theta < -np.pi+extend_range
    theta = np.concatenate((theta, theta[ids]+2*np.pi))
    phi = np.concatenate((phi, phi[ids]))
    nmap = np.concatenate((nmap, nmap[ids]))
    # extend phi
    ids = phi > np.pi-extend_range
    theta = np.concatenate((theta, theta[ids]))
    phi = np.concatenate((phi, phi[ids]-2*np.pi))
    nmap = np.concatenate((nmap, nmap[ids]))
    ids = phi < -np.pi+extend_range
    theta = np.concatenate((theta, theta[ids]))
    phi = np.concatenate((phi, phi[ids]+2*np.pi))
    nmap = np.concatenate((nmap, nmap[ids]))

    map_max = len(nmap)

    # extend theta
    ids = theta > np.pi
    theta = np.concatenate((theta, theta[ids]-2*np.pi-2*extend_range))
    phi = np.concatenate((phi, phi[ids]))
    ids = theta < -np.pi
    theta = np.concatenate((theta, theta[ids]+2*np.pi+2*extend_range))
    phi = np.concatenate((phi, phi[ids]))
    # extend phi
    ids = phi > np.pi
    theta = np.concatenate((theta, theta[ids]))
    phi = np.concatenate((phi, phi[ids]-2*np.pi-2*extend_range))
    ids = phi < -np.pi
    theta = np.concatenate((theta, theta[ids]))
    phi = np.concatenate((phi, phi[ids]+2*np.pi+2*extend_range))

    # Triangulate
    tri = Delaunay(np.block([[theta], [phi]]).T).simplices
    # remove boundary faces
    tri = tri[np.logical_not(np.any(tri>map_max, axis=1))]
    # map to original nodes
    tri[tri>=len(nodes)] = nmap[tri[tri>=len(nodes)]]
    # remove duplicates
    tri = np.unique(tri, axis=0)
    return tri
