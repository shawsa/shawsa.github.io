import numpy as np

from rbf_dict import *

def surf_laplacian_weights(nodes, rbf, eps, stencil_size, basis_deg, basis_type, 
