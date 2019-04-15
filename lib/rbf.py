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
    phi1            1/r* d/dr of the radial kernel
    phi n+1         1/r d/dr of phi n
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
    return np.exp( -(eps*r)**2 )
def phi1(r, eps):
	return -2*eps**2*np.exp(-eps**2*r**2)
def phi2(r, eps):
	return 4*eps**4*np.exp(-eps**2*r**2)
def phi3(r, eps):
	return -8*eps**6*np.exp(-eps**2*r**2)
def phi4(r, eps):
	return 16*eps**8*np.exp(-eps**2*r**2)
def phi5(r, eps):
	return -32*eps**10*np.exp(-eps**2*r**2)
def drbf(r, eps):
	return -2*eps**2*r*np.exp(-eps**2*r**2)
def d2rbf(r, eps):
	return eps**2*(4*eps**2*r**2 - 2)*np.exp(-eps**2*r**2)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
shape_labels += [label]

label = 'multiquadric'
tex = '$\sqrt{1+(\\varepsilon r)^2}$'
def rbf(r, eps):
    return np.sqrt(1+(eps*r)**2)
def phi1(r, eps):
	return eps**2/np.sqrt(eps**2*r**2 + 1)
def phi2(r, eps):
	return -eps**4/(eps**2*r**2 + 1)**(3/2)
def phi3(r, eps):
	return 3*eps**6/(eps**2*r**2 + 1)**(5/2)
def phi5(r, eps):
	return 105*eps**10/(eps**2*r**2 + 1)**(9/2)
def drbf(r, eps):
	return eps**2*r/np.sqrt(eps**2*r**2 + 1)
def d2rbf(r, eps):
	return -eps**4*r**2/(eps**2*r**2 + 1)**(3/2) + eps**2/np.sqrt(eps**2*r**2 + 1)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
shape_labels += [label]

label = 'inverse quadratic'
tex = '$\\frac{1}{1+(\\varepsilon r)^2}$'
def rbf(r, eps):
    return 1/(1+(eps*r)**2)
def phi1(r, eps):
	return -2*eps**2/(eps**2*r**2 + 1)**2
def phi2(r, eps):
	return 8*eps**4/(eps**2*r**2 + 1)**3
def phi3(r, eps):
	return -48*eps**6/(eps**2*r**2 + 1)**4
def phi4(r, eps):
	return 384*eps**8/(eps**2*r**2 + 1)**5
def phi5(r, eps):
	return -3840*eps**10/(eps**2*r**2 + 1)**6
def drbf(r, eps):
	return -2*eps**2*r/(eps**2*r**2 + 1)**2
def d2rbf(r, eps):
	return eps**2*(6*eps**2*r**2 - 2)/(eps**2*r**2 + 1)**3
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
shape_labels += [label]

label = 'inverse multiquadric'
tex = '$\\frac{1}{\\sqrt{1+(\\varepsilon r)^2}}$'
def rbf(r, eps):
    return 1/(1+(r*eps)**2)**(1/2)
def phi1(r, eps):
	return -eps**2/(eps**2*r**2 + 1)**(3/2)
def phi2(r, eps):
	return 3*eps**4/(eps**2*r**2 + 1)**(5/2)
def phi3(r, eps):
	return -15*eps**6/(eps**2*r**2 + 1)**(7/2)
def phi4(r, eps):
	return 105*eps**8/(eps**2*r**2 + 1)**(9/2)
def phi5(r, eps):
	return -945*eps**10/(eps**2*r**2 + 1)**(11/2)
def drbf(r, eps):
	return -eps**2*r/(eps**2*r**2 + 1)**(3/2)
def d2rbf(r, eps):
	return eps**2*(2*eps**2*r**2 - 1)/(eps**2*r**2 + 1)**(5/2)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
shape_labels += [label]


#########################################################################
#
# PHS RBFs
#
#########################################################################
shape = False

label = 'r^2 log(r)'
tex = '$r^2 \\log(r)$'
def rbf(r,eps):
    return r**2 * np.log(r+MEPS)
def phi1(r, eps):
	return 2*np.log(MEPS + r) + 1
def phi2(r, eps):
	return 2/r**2
def phi3(r, eps):
	return -4/r**4
def phi4(r, eps):
	return 16/r**6
def phi5(r, eps):
	return -96/r**8
def drbf(r, eps):
	return r*(2*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return 2*np.log(MEPS + r) + 3
	return 2*np.log(MEPS + r) + 3
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^4 log(r)'
tex = '$r^4 \\log(r)$'
def rbf(r,eps):
    return r**4 * np.log(r+MEPS)
def phi1(r, eps):
	return r**2*(4*np.log(MEPS + r) + 1)
def phi2(r, eps):
	return 8*np.log(MEPS + r) + 6
def phi3(r, eps):
	return 8/r**2
def phi4(r, eps):
	return -16/r**4
def phi5(r, eps):
	return 64/r**6
def drbf(r, eps):
	return r**3*(4*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**2*(12*np.log(MEPS + r) + 7)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^6 log(r)'
tex = '$r^6 \\log(r)$'
def rbf(r,eps):
    return r**6 * np.log(r+MEPS)
def phi1(r, eps):
	return r**4*(6*np.log(MEPS + r) + 1)
def phi2(r, eps):
	return r**2*(24*np.log(MEPS + r) + 10)
def phi3(r, eps):
	return 48*np.log(MEPS + r) + 44
def phi4(r, eps):
	return 48/r**2
def phi5(r, eps):
	return -96/r**4
def drbf(r, eps):
	return r**5*(6*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**4*(30*np.log(MEPS + r) + 11)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^8 log(r)'
tex = '$r^8 \\log(r)$'
def rbf(r,eps):
    return r**8 * np.log(r+MEPS)
def phi1(r, eps):
	return r**6*(8*np.log(MEPS + r) + 1)
def phi2(r, eps):
	return r**4*(48*np.log(MEPS + r) + 14)
def phi3(r, eps):
	return r**2*(192*np.log(MEPS + r) + 104)
def phi4(r, eps):
	return 384*np.log(MEPS + r) + 400
def phi5(r, eps):
	return 384/r**2
def drbf(r, eps):
	return r**7*(8*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**6*(56*np.log(MEPS + r) + 15)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^10 log(r)'
tex = '$r^{10} \\log(r)$'
def rbf(r, eps):
	return r**10*np.log(MEPS + r)
def phi1(r, eps):
	return r**8*(10*np.log(MEPS + r) + 1)
def phi2(r, eps):
	return r**6*(80*np.log(MEPS + r) + 18)
def phi3(r, eps):
	return r**4*(480*np.log(MEPS + r) + 188)
def phi4(r, eps):
	return r**2*(1920*np.log(MEPS + r) + 1232)
def phi5(r, eps):
	return 3840*np.log(MEPS + r) + 4384
def drbf(r, eps):
	return r**9*(10*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**8*(90*np.log(MEPS + r) + 19)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^12 log(r)'
tex = '$r^{12} \\log(r)$'
def rbf(r, eps):
	return r**12*np.log(MEPS + r)
def phi1(r, eps):
	return r**10*(12*np.log(MEPS + r) + 1)
def phi2(r, eps):
	return r**8*(120*np.log(MEPS + r) + 22)
def phi3(r, eps):
	return r**6*(960*np.log(MEPS + r) + 296)
def phi4(r, eps):
	return r**4*(5760*np.log(MEPS + r) + 2736)
def phi5(r, eps):
	return r**2*(23040*np.log(MEPS + r) + 16704)
def drbf(r, eps):
	return r**11*(12*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**10*(132*np.log(MEPS + r) + 23)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^14 log(r)'
tex = '$r^{14} \\log(r)$'
def rbf(r, eps):
	return r**14*np.log(MEPS + r)
def phi1(r, eps):
	return r**12*(14*np.log(MEPS + r) + 1)
def phi2(r, eps):
	return r**10*(168*np.log(MEPS + r) + 26)
def phi3(r, eps):
	return r**8*(1680*np.log(MEPS + r) + 428)
def phi4(r, eps):
	return r**6*(13440*np.log(MEPS + r) + 5104)
def phi5(r, eps):
	return r**4*(80640*np.log(MEPS + r) + 44064)
def drbf(r, eps):
	return r**13*(14*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**12*(182*np.log(MEPS + r) + 27)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^16 log(r)'
tex = '$r^{16} \\log(r)$'
def rbf(r, eps):
	return r**16*np.log(MEPS + r)
def phi1(r, eps):
	return r**14*(16*np.log(MEPS + r) + 1)
def phi2(r, eps):
	return r**12*(224*np.log(MEPS + r) + 30)
def phi3(r, eps):
	return r**10*(2688*np.log(MEPS + r) + 584)
def phi4(r, eps):
	return r**8*(26880*np.log(MEPS + r) + 8528)
def phi5(r, eps):
	return r**6*(215040*np.log(MEPS + r) + 95104)
def drbf(r, eps):
	return r**15*(16*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**14*(240*np.log(MEPS + r) + 31)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^18 log(r)'
tex = '$r^{18} \\log(r)$'
def rbf(r, eps):
	return r**18*np.log(MEPS + r)
def phi1(r, eps):
	return r**16*(18*np.log(MEPS + r) + 1)
def phi2(r, eps):
	return r**14*(288*np.log(MEPS + r) + 34)
def phi3(r, eps):
	return r**12*(4032*np.log(MEPS + r) + 764)
def phi4(r, eps):
	return r**10*(48384*np.log(MEPS + r) + 13200)
def phi5(r, eps):
	return r**8*(483840*np.log(MEPS + r) + 180384)
def drbf(r, eps):
	return r**17*(18*np.log(MEPS + r) + 1)
def d2rbf(r, eps):
	return r**16*(306*np.log(MEPS + r) + 35)
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]




label = 'r^3'
tex = '$r^3$'
def rbf(r, eps):
	return r**3
def phi1(r, eps):
	return 3*r
def phi2(r, eps):
	return 3/r
def phi3(r, eps):
	return -3/r**3
def phi4(r, eps):
	return 9/r**5
def phi5(r, eps):
	return -45/r**7
def drbf(r, eps):
	return 3*r**2
def d2rbf(r, eps):
	return 6*r
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^5'
tex = '$r^5$'
def rbf(r, eps):
	return r**5
def phi1(r, eps):
	return 5*r**3
def phi2(r, eps):
	return 15*r
def phi3(r, eps):
	return 15/r
def phi4(r, eps):
	return -15/r**3
def phi5(r, eps):
	return 45/r**5
def drbf(r, eps):
	return 5*r**4
def d2rbf(r, eps):
	return 20*r**3
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^7'
tex = '$r^7$'
def rbf(r, eps):
	return r**7
def phi1(r, eps):
	return 7*r**5
def phi2(r, eps):
	return 35*r**3
def phi3(r, eps):
	return 105*r
def phi4(r, eps):
	return 105/r
def phi5(r, eps):
	return -105/r**3
def drbf(r, eps):
	return 7*r**6
def d2rbf(r, eps):
	return 42*r**5
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^9'
tex = '$r^9$'
def rbf(r,eps):
    return r**9
def phi1(r, eps):
	return 9*r**7
def phi2(r, eps):
	return 63*r**5
def phi3(r, eps):
	return 315*r**3
def phi4(r, eps):
	return 945*r
def phi5(r, eps):
	return 945/r
def drbf(r, eps):
	return 9*r**8
def d2rbf(r, eps):
	return 72*r**7
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^11'
tex = '$r^{11}$'
def rbf(r, eps):
	return r**11
def phi1(r, eps):
	return 11*r**9
def phi2(r, eps):
	return 99*r**7
def phi3(r, eps):
	return 693*r**5
def phi4(r, eps):
	return 3465*r**3
def phi5(r, eps):
	return 10395*r
def drbf(r, eps):
	return 11*r**10
def d2rbf(r, eps):
	return 110*r**9
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^13'
tex = '$r^{13}$'
def rbf(r, eps):
	return r**13
def phi1(r, eps):
	return 13*r**11
def phi2(r, eps):
	return 143*r**9
def phi3(r, eps):
	return 1287*r**7
def phi4(r, eps):
	return 9009*r**5
def phi5(r, eps):
	return 45045*r**3
def drbf(r, eps):
	return 13*r**12
def d2rbf(r, eps):
	return 156*r**11
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]

label = 'r^15'
tex = '$r^{15}$'
def rbf(r, eps):
	return r**15
def phi1(r, eps):
	return 15*r**13
def phi2(r, eps):
	return 195*r**11
def phi3(r, eps):
	return 2145*r**9
def phi4(r, eps):
	return 19305*r**7
def phi5(r, eps):
	return 135135*r**5
def drbf(r, eps):
	return 15*r**14
def d2rbf(r, eps):
	return 210*r**13
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
rbf_dict[label] = rbf_obj
phs_labels += [label]


label = 'r^17'
tex = '$r^{17}$'
def rbf(r, eps):
	return r**17
def phi1(r, eps):
	return 17*r**15
def phi2(r, eps):
	return 255*r**13
def phi3(r, eps):
	return 3315*r**11
def phi4(r, eps):
	return 36465*r**9
def phi5(r, eps):
	return 328185*r**7
def drbf(r, eps):
	return 17*r**16
def d2rbf(r, eps):
	return 272*r**15
rbf_obj = {'label':label, 'tex':tex, 'shape':shape, 'rbf':rbf,
            'phi1':phi1, 'phi2':phi2, 'phi3':phi3, 'phi4':phi4, 
            'phi5': phi5, 'drbf':drbf, 'd2rbf':d2rbf}
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
    #print(np.log( la.cond(AP) / target_cond))
    return np.log( la.cond(AP) / target_cond)

def optimize_eps(rbf, dist_mat, P=None, target_cond=10**12, interval=[MEPS, 10]):
    n = dist_mat.shape[0]
    eps_guess = 1/np.min(dist_mat+np.diag([np.max(dist_mat)]*n))
    
    #try:
    root = brentq(functional, 
            interval[0], interval[1],
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

def rbf_interp(xs, fs, zs, rbf, eps=1, optimize_shape=False, target_cond=10**12, return_cond=False, eps_interval=[MEPS, 10]):
    # generate distance matrix
    if xs.ndim == 1:
        dist_mat = np.abs(np.subtract.outer(xs,xs))
    else:
        dist_mat = dist_outer(xs,xs)
    # optimize shape parameter
    if optimize_shape:
        eps = optimize_eps(rbf, dist_mat, target_cond=target_cond, interval=eps_interval)

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

def rbf_interp_local(xs, fs, zs, rbf_obj, k=25, eps=None):
    rbf = rbf_obj['rbf']
    us = np.zeros(len(zs))
    full_dist_mat = dist_outer(xs, zs)
    closest_ids = np.argmin(full_dist_mat, axis=1)
    # in a zoop build surface around each sample point
    for i in range(len(xs)):
        c = xs[i]
        close_to_c_ids = closest_ids == i
        zs_local = zs[close_to_c_ids]
        x_ids = get_closest_indices(i, len(xs), k)
        xs_local = xs[x_ids]
        fs_local = fs[x_ids]
        dist_mat = dist_outer(xs_local,xs_local)
        if rbf_obj['shape'] and eps is None:   
            eps = optimize_eps(rbf, dist_mat)
        A = rbf(dist_mat, eps)
        cs = la.solve(A, fs_local)
        dist_mat = dist_outer(xs_local, zs_local)
        A = rbf(dist_mat, eps)
        us_local = A @ cs
        us[close_to_c_ids] = us_local
    return us






