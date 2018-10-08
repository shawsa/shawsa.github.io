import numpy as np
MEPS = np.finfo(float).eps
import numpy.linalg as la
from scipy.optimize import brentq

'''
Each RBF in the dictionary should have the following values defined
    rbf             the radial kernel
    label           the text label for the rbf
    tex             the tex code for the formula
    shape           True iff it requires a shape parameter
    zeta            1/r* d/dr of the radial kernel
    chi             1/r d/dr of zeta
    beta            1/r d/dr of chi
'''


rbf_dict = {}
shape_labels = []
phs_labels = []

#########################################################################
#
# Shape paramater RBFs
#
#########################################################################
shape = True


label = 'gaussian'
tex = '$e^{-(\\varepsilon r)^2}$'
def rbf(r, eps):
    return np.exp(-(eps*r)**2)
def zeta(r, eps):
	return -2*eps**2*np.exp(-eps**2*r**2)
def chi(r, eps):
	return 4*eps**4*np.exp(-eps**2*r**2)
def beta(r, eps):
	return -8*eps**6*np.exp(-eps**2*r**2)
def drbf(r, eps):
	return -2*eps**2*r*np.exp(-eps**2*r**2)
def d2rbf(r, eps):
	return eps**2*(4*eps**2*r**2 - 2)*np.exp(-eps**2*r**2)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
shape_labels += [label]

label = 'multiquadric'
tex = '$\sqrt{1+(\\varepsilon r)^2}$'
def rbf(r, eps):
    return np.sqrt(1+(eps*r)**2)
def zeta(r, eps):
	return eps**2/np.sqrt(eps**2*r**2 + 1)
def chi(r, eps):
	return -eps**4/(eps**2*r**2 + 1)**(3/2)
def beta(r, eps):
	return 3*eps**6/(eps**2*r**2 + 1)**(5/2)
def drbf(r, eps):
	return eps**2*r/np.sqrt(eps**2*r**2 + 1)
def d2rbf(r, eps):
	return -eps**4*r**2/(eps**2*r**2 + 1)**(3/2) + eps**2/np.sqrt(eps**2*r**2 + 1)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
shape_labels += [label]

label = 'inverse quadratic'
tex = '$\\frac{1}{1+(\\varepsilon r)^2}$'
def rbf(r, eps):
    return 1/(1+(eps*r)**2)
def zeta(r, eps):
	return -2*eps**2/(eps**2*r**2 + 1)**2
def chi(r, eps):
	return 8*eps**4/(eps**2*r**2 + 1)**3
def beta(r, eps):
	return -48*eps**6/(eps**2*r**2 + 1)**4
def drbf(r, eps):
	return -2*eps**2*r/(eps**2*r**2 + 1)**2
def d2rbf(r, eps):
	return 8*eps**4*r**2/(eps**2*r**2 + 1)**3 - 2*eps**2/(eps**2*r**2 + 1)**2
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
shape_labels += [label]

label = 'inverse multiquadric'
tex = '$\\frac{1}{\\sqrt{1+(\\varepsilon r)^2}}$'
def rbf(r, eps):
    return 1/np.sqrt(1+(eps*r)**2)
def zeta(r, eps):
	return -eps**2/(eps**2*r**2 + 1)**(3/2)
def chi(r, eps):
	return 3*eps**4/(eps**2*r**2 + 1)**(5/2)
def beta(r, eps):
	return -15*eps**6/(eps**2*r**2 + 1)**(7/2)
def drbf(r, eps):
	return -eps**2*r/(eps**2*r**2 + 1)**(3/2)
def d2rbf(r, eps):
	return 3*eps**4*r**2/(eps**2*r**2 + 1)**(5/2) - eps**2/(eps**2*r**2 + 1)**(3/2)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
shape_labels += [label]


#########################################################################
#
# PHS RBFs
#
#########################################################################
shape = False

label = 'thin plate spline'
tex = '$\\log(r)r^2$'
def rbf(r,eps):
    return r**2 * np.log(r+MEPS)
def zeta(r, eps):
	return 2*np.log(MEPS + r) + 1
def chi(r, eps):
	return 2/r**2
def beta(r, eps):
	return -4/r**4
def drbf(r, eps):
	return r*(2*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return 2*np.log(MEPS + r) + 3
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'fourth degree PHS'
tex = '$\\log(r)r^4$'
def rbf(r,eps):
    return r**4 * np.log(r+MEPS)
def zeta(r, eps):
	return r**2*(4*np.log(MEPS + r) + 1)
def chi(r, eps):
	return 8*np.log(MEPS + r) + 6
def beta(r, eps):
	return 8/r**2
def drbf(r, eps):
	return r**3*(4*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**2*(12*np.log(MEPS + r) + 7)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'sixth degree PHS'
tex = '$\\log(r)r^6$'
def rbf(r,eps):
    return r**6 * np.log(r+MEPS)
def zeta(r, eps):
	return r**4*(6*np.log(MEPS + r) + 1)
def chi(r, eps):
	return r**2*(24*np.log(MEPS + r) + 10)
def beta(r, eps):
	return 48*np.log(MEPS + r) + 44
def drbf(r, eps):
	return r**5*(6*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**4*(30*np.log(MEPS + r) + 11)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'eighth degree PHS'
tex = '$\\log(r)r^8$'
def rbf(r,eps):
    return r**8 * np.log(r+MEPS)
def zeta(r, eps):
	return r**6*(8*np.log(MEPS + r) + 1)
def chi(r, eps):
	return r**4*(48*np.log(MEPS + r) + 14)
def beta(r, eps):
	return r**2*(192*np.log(MEPS + r) + 104)
def drbf(r, eps):
	return r**7*(8*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**6*(56*np.log(MEPS + r) + 15)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'cubic spline'
tex = '$r^3$'
def rbf(r,eps):
    return r**3
def zeta(r, eps):
	return 3*r
def chi(r, eps):
	return 3/r
def beta(r, eps):
	return -3/r**3
def drbf(r, eps):
	return 3*r**2
def d2rbf(r, eps):
	return 6*r
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'fith degree PHS'
tex = '$r^5$'
def rbf(r,eps):
    return r**5
def zeta(r, eps):
	return 5*r**3
def chi(r, eps):
	return 15*r
def beta(r, eps):
	return 15/r
def drbf(r, eps):
	return 5*r**4
def d2rbf(r, eps):
	return 20*r**3
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'seventh degree PHS'
tex = '$r^7$'
def rbf(r,eps):
    return r**7
def zeta(r, eps):
	return 7*r**5
def chi(r, eps):
	return 35*r**3
def beta(r, eps):
	return 105*r
def drbf(r, eps):
	return 7*r**6
def d2rbf(r, eps):
	return 42*r**5
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'ninth degree PHS'
tex = '$r^9$'
def rbf(r,eps):
    return r**9
def zeta(r, eps):
	return 9*r**7
def chi(r, eps):
	return 63*r**5
def beta(r, eps):
	return 315*r**3
def drbf(r, eps):
	return 9*r**8
def d2rbf(r, eps):
	return 72*r**7
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'zeta':zeta, 'chi':chi, 'beta':beta, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

# define list of all labels
rbf_labels = shape_labels + phs_labels


#########################################################################
#
# Shape parameter optimization
#
#########################################################################

def functional(eps, rbf, dist_mat, P, target_cond):
    A = rbf(dist_mat, eps)
    if P is None:
        AP = A
    else:
        k = len(P[0])
        AP = np.block([[A, P],[P.T, np.zeros((k,k))]])
    return np.log( la.cond(AP) / target_cond)

def optimize_eps(rbf, dist_mat, P=None, target_cond=10**12):
    n = dist_mat.shape[0]
    eps_guess = 1/np.min(dist_mat+np.diag([np.max(dist_mat)]*n))
    
    #try:
    root = brentq(functional, 
            MEPS, 10,
            args=(rbf, dist_mat, P, target_cond))
    #except ValueError:
    #    root = eps_guess
    return root

#########################################################################
#
# Distance matrix
#
#########################################################################
def dist_outer(nodes1, nodes2):
    d = len(nodes1[0]) # the dimension of each vector
    n1 = len(nodes1)
    n2 = len(nodes2)
    # create a row vector of d dimensional vectors
    row = nodes1.reshape((1,n1,d)) 
    # create a column vector of d dimensional vectors
    col = nodes2.reshape((n2,1,d)) 
    ret = (row-col)**2
    ret = np.sum(ret,2) #sum each d-dimensional vector
    return np.sqrt(ret)

#########################################################################
#
# Interpolation
#
#########################################################################

def rbf_interp(xs, fs, zs, rbf, eps=1, optimize_shape=False, target_cond=10**12, return_cond=False):
    # generate distance matrix
    if xs.ndim == 1:
        dist_mat = np.abs(np.subtract.outer(xs,xs))
    else:
        dist_mat = dist_outer(xs,xs)
    # optimize shape parameter
    if optimize_shape:
        eps = optimize_eps(rbf, dist_mat, target_cond=target_cond)

    A = rbf(dist_mat, eps)

    if return_cond:
        A_cond = la.cond(A)
    else:
        A_cond = None
    
    cs = la.solve(A, fs)
    if xs.ndim == 1:
        dist_mat = np.abs(np.subtract.outer(zs,xs))
    else:
        dist_mat = dist_outer(zs,xs)
    A = rbf(dist_mat, eps)
    return A @ cs, eps, A_cond

#########################################################################
#
# Local Interpolation - 1D
#
#########################################################################
def get_closest_indices(i, n, k):
    half = (k+1)//2
    if i < half:
        return np.arange(k)
    elif i < n-half:
        return np.arange(i-half+1, i+half)
    else:
        return np.arange(n-k,n)

def rbf_interp_local_1D(xs, fs, zs, rbf, stencil_size=10, eps=1, optimize_shape=False, target_cond=10**12, return_cond=False):
    k = stencil_size    
    us = np.zeros(len(zs))
    full_dist_mat = np.abs(np.subtract.outer(zs,xs))
    closest_ids = np.argmin(full_dist_mat, axis=1)
    # in a zoop build surface around each sample point
    for i in range(len(xs)):
        c = xs[i]
        close_to_c_ids = closest_ids == i
        zs_local = zs[close_to_c_ids]
        x_ids = get_closest_indices(i, len(xs), k)
        xs_local = xs[x_ids]
        fs_local = fs[x_ids]
        dist_mat = np.abs(np.subtract.outer(xs_local,xs_local))
        if optimize_shape:   
            eps = optimize_eps(rbf, dist_mat)
        A = rbf(dist_mat, eps)
        cs = la.solve(A, fs_local)
        dist_mat = np.abs(np.subtract.outer(zs_local,xs_local))
        A = rbf(dist_mat, eps)
        us_local = A @ cs
        us[close_to_c_ids] = us_local
    return us






