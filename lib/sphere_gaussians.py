import numpy as np
from numpy.linalg import cond, norm, inv
def __dist(node1, node2):
    return np.sqrt( (node1[0]-node2[0])**2 + (node1[1]-node2[1])**2 + (node1[2]-node2[2])**2 )

def L_gauss(x, sigmas, ys):
    ret = 0
    for sigma, y in zip(sigmas,ys):
        r2 = __dist(x,y)**2
        ret -= sigma*np.exp(-sigma*r2) * (4+r2*(-2+(-4+r2)*sigma))
    return ret
    
def sum_gauss(x, sigmas, ys):
    ret = 0
    for sigma, y in zip(sigmas,ys):
        ret += np.exp(-sigma*__dist(x,y)**2)
    return ret

def get_sphere_gaussians():
    sigmas = [2, .5, .3, .1, 5, 2, 1.5]
    ys = np.array([ [0,0,1],
                    [1,-1,1],
                    [2,0,1],
                    [-5,7,0],
                    [2,-13,1],
                    [2,12,-15],
                    [1,0,-1]], dtype=np.float)
    # project the centers onto the sphere
    for i, y in enumerate(ys):
        ys[i] = 1.0/norm(y)* y
    foo = lambda x: sum_gauss(x, sigmas, ys)
    exact = lambda x: L_gauss(x, sigmas, ys)

    return foo, exact

def dist2_vec(nodes, pt):
    return (nodes[:,0]-pt[0])**2 + (nodes[:,1]-pt[1])**2 + (nodes[:,2]-pt[2])**2

def sphere_forcing(nodes, sigmas, ys, t):
    u = 0
    lapu = 0
    f = 0
    for s, y in zip(sigmas,ys):
        r2 = dist2_vec(nodes, y)
        u += np.exp(-s*r2)
        lapu += -s*np.exp(-s*r2) * (4+r2*(-2+(-4+r2)*s))
    u *= np.exp(-t)
    lapu *= np.exp(-t)
    f = -u - lapu
    return u, f

def get_sphere_forcing():
    sigmas = [2, .5, .3, .1, 5, 2, 1.5]
    ys = np.array([ [0,0,1],
                    [1,-1,1],
                    [2,0,1],
                    [-5,7,0],
                    [2,-13,1],
                    [2,12,-15],
                    [1,0,-1]], dtype=np.float)
    # project the centers onto the sphere
    for i, y in enumerate(ys):
        ys[i] = 1.0/norm(y)* y
    return lambda nodes, t: sphere_forcing(nodes, sigmas, ys, t)
