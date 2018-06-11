import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

##################################################################################
#
# Function to interpolate and RBF
#
##################################################################################
def foo(x,y,z):
    return 1 + np.sin(11*(x-.1))*np.cos(9*(y+.1))*np.cos(3*(z+.2))
    #return x**3

def dist(x1, y1, z1, x2, y2, z2):
    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )

def rbf(r):
    return r**3

##################################################################################
#
# Sample Points
#
##################################################################################
n = 600
indices = np.arange(0, n, dtype=float) + 0.5
phi = np.arccos(1 - 2*indices/n)
theta = np.pi * (1 + 5**0.5) * indices
xs, ys, zs = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);

##################################################################################
#
# Calculate Weights
#
##################################################################################
fs = foo(xs, ys, zs)
A = rbf( dist(xs.reshape((n,1)), ys.reshape((n,1)), zs.reshape((n,1)), xs, ys, zs) )
P = np.block([[np.ones(n)], [xs], [ys], [zs]]).T
AP = np.block([[A, P],[P.T, np.zeros((4,4))]])
weights = la.solve(AP, np.concatenate([fs, np.zeros(4)]))

##################################################################################
#
# Construct Interpolant
#
##################################################################################
n_grid_dir = 50

u = np.linspace(0, 2 * np.pi, n_grid_dir)
v = np.linspace(0, np.pi, n_grid_dir)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones(np.size(u)), np.cos(v))

n_grid = len(X.ravel())

X = X.reshape((n_grid,1))
Y = Y.reshape(X.shape)
Z = Z.reshape(X.shape)

us = rbf( dist( X, Y, Z, xs, ys, zs) )
us = np.block([us, np.ones((n_grid, 1)), X, Y, Z])
us = us @ weights

X = X.reshape((n_grid_dir,n_grid_dir))
Y = Y.reshape(X.shape)
Z = Z.reshape(X.shape)
us = us.reshape(X.shape)
print(np.max(np.abs(us - foo(X,Y,Z))))

##################################################################################
#
# Plot Interpolant
#
##################################################################################
color_dimension = us # change to desired fourth dimension
minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

fig = plt.figure()
ax = fig.gca(projection='3d')
line = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=True)

plt.axis('off')
ax.grid(False)

fig.canvas.draw()
plt.show()









