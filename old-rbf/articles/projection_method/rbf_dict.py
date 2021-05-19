import numpy as np
MEPS = np.finfo(float).eps
import numpy.linalg as la
from scipy.optimize import minimize_scalar

rbf_dict = {}
rbf_labels = []

#################################################################################
#
# Shape paramater RBFs
#
#################################################################################

# Gaussian
rbf_label = 'gaussian'
rbf_type = 'shape'
tex = 'gaussian'
def phi(r, eps):
    return np.exp(-(eps*r)**2)
def zeta(r, eps):
    return -2*eps**2 * np.exp(-(eps*r)**2)

rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

# Multiquadric
rbf_label = 'multiquadric'
rbf_type = 'shape'
tex = 'multiquadric'
def phi(r, eps):
    return np.sqrt(1+(eps*r)**2)
def zeta(r, eps):
    return eps**2/np.sqrt(eps**2*r**2 + 1)

rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

# Inverse Quadratic
rbf_label = 'inverse quadratic'
rbf_type = 'shape'
tex = 'inverse quadratic'
def phi(r, eps):
    return 1/(1+(eps*r)**2)
def zeta(r, eps):
    return -2*eps**2/(eps**2*r**2 + 1)**2

rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]


# Inverse Multiquadric
rbf_label = 'inverse multiquadric'
rbf_type = 'shape'
tex = 'inverse multiquadric'
def phi(r, eps):
    return 1/np.sqrt(1+(eps*r)**2)
def zeta(r, eps):
    return -eps**2/np.sqrt(eps**2*r**2 + 1)**3

rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

#################################################################################
#
# PHS RBFs
#
#################################################################################

# Thin Plate Spline
rbf_label = 'log(r)r^2'
rbf_type = 'phs'
tex = '$r^2\\log(r)$'
def phi(r, eps):
    return r**2 * np.log(r+MEPS)
def zeta(r, eps):
    return 1 + 2*np.log(r+MEPS)
rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

# r^3
rbf_label = 'r^3'
rbf_type = 'phs'
tex = '$r^3$'
def phi(r, eps):
    return r**3
def zeta(r, eps):
    return 3*r
rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

# log(r)r^4
rbf_label = 'log(r)r^4'
rbf_type = 'phs'
tex = '$r^4\\log(r)$'
def phi(r, eps):
    return r**4 * np.log(r+MEPS)
def zeta(r, eps):
    return r**2 * (1 + 4*np.log(r+MEPS))
rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

# r^5
rbf_label = 'r^5'
rbf_type = 'phs'
tex = '$r^5$'
def phi(r, eps):
    return r**5
def zeta(r, eps):
    return 5*r**3
rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

# log(r)r^6
rbf_label = 'log(r)r^6'
rbf_type = 'phs'
tex = '$r^6\\log(r)$'
def phi(r, eps):
    return r**6 * np.log(r+MEPS)
def zeta(r, eps):
    return r**4 * (1 + 6*np.log(r+MEPS))
rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

# r^5
rbf_label = 'r^7'
rbf_type = 'phs'
tex = '$r^5$'
def phi(r, eps):
    return r**7
def zeta(r, eps):
    return 7*r**5
rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

# log(r)r^8
rbf_label = 'log(r)r^8'
rbf_type = 'phs'
tex = '$r^8\\log(r)$'
def phi(r, eps):
    return r**8 * np.log(r+MEPS)
def zeta(r, eps):
    return r**6 * (1 + 8*np.log(r+MEPS))
rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

# r^9
rbf_label = 'r^9'
rbf_type = 'phs'
tex = '$r^9$'
def phi(r, eps):
    return r**9
def zeta(r, eps):
    return 9*r**7
rbf_dict[rbf_label] = {'label': rbf_label, 'phi':phi, 'type':rbf_type, 'tex': tex, 'zeta':zeta}
rbf_labels += [rbf_label]

#################################################################################
#
# Helper phitions
#
#################################################################################

def dist_outer(nodes1, nodes2):
    n1 = len(nodes1)
    n2 = len(nodes2)
    L = nodes1.reshape((1,n1,3))
    R = nodes2.reshape((n2,1,3))
    ret = (L-R)**2
    ret = np.sum(ret,2)
    return np.sqrt(ret)

def functional(eps, dist_mat, rbf , target_cond):
    return np.log( la.cond(phi(dist_mat, eps)) / target_cond) **2

def optimize_eps(rbf, dist_mat, target_cond=10**12):
    n = dist_mat.shape[0]
    eps_guess = 1/np.min(dist_mat+np.diag([1]*n))
    optimization_result = minimize_scalar(functional, 
            bracket=[eps_guess/2, eps_guess*2],
            args=(dist_mat, rbf, target_cond))
    return optimization_result['x']


