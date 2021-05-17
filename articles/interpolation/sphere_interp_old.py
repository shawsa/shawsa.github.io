import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.animation as animation
def init():
    ax.view_init(azim=0)
    return line, 

def rotate(angle):
    ax.view_init(azim=angle)
    return line,

##################################################################################
#
# Points to plot
#
##################################################################################
# steps in each parameter
n_grid_dir = 10

u = np.linspace(0, 2 * np.pi, n_grid_dir)
v = np.linspace(0, np.pi, n_grid_dir)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones(np.size(u)), np.cos(v))

n_grid = len(X.ravel())

##################################################################################
#
# Function to Interpolate
#
##################################################################################
def foo(x,y,z):
    return 1 + np.sin(11*(x-.1))*np.cos(9*(y+.1))*np.cos(3*(z+.2))

def foo(x,y,z):
    return z


##################################################################################
#
# Generate Nodes and rbf
#
##################################################################################

def dist(x1, y1, z1, x2, y2, z2):
    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )

def rbf(r):
    return r**3

n = 100
indices = np.arange(0, n, dtype=float) + 0.5

phi = np.arccos(1 - 2*indices/n)
theta = np.pi * (1 + 5**0.5) * indices

xs, ys, zs = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);

##################################################################################
#
# Interpolate
#
##################################################################################

fs = foo(xs, ys, zs) 
xs = xs.reshape((1,n))
ys = ys.reshape((1,n))
zs = zs.reshape((1,n))
x2 = ys.reshape((n,1))
y2 = ys.reshape((n,1))
z2 = zs.reshape((n,1))
A = rbf( dist(x2, y2, z2, xs, ys, zs) )

P = np.block([ [np.ones(n)], [xs], [ys], [zs] ]).T
AP = np.block([[A, P],[P.T, np.zeros((4,4))]])
weights = la.solve(AP, np.concatenate([fs, np.zeros(4)]) )

ws = weights[:n]
cs = weights[n:]


X = X.reshape((n_grid, 1))
Y = Y.reshape((n_grid, 1))
Z = Z.reshape((n_grid, 1))
#.reshape((n_grid, 1))

val = rbf( dist(X, Y, Z, xs, ys, zs) )
val = np.block( [val, np.ones((n_grid, 1)), X, Y, Z])
print(val.shape)

#val = val @ ws
val = val @ weights
print(val.shape)
print(la.norm(val))

#val = val + cs[0] + cs[1]*X.ravel() + cs[2]*Y.ravel() + cs[3]*Z.ravel()

print(val.shape)

X = X.reshape((n_grid_dir, n_grid_dir))
Y = Y.reshape((n_grid_dir, n_grid_dir))
Z = Z.reshape((n_grid_dir, n_grid_dir))
val = val.reshape((n_grid_dir, n_grid_dir))


print(np.max(np.abs( val - foo(X,Y,Z) )) )




##################################################################################
#
# Plot the Function
#
##################################################################################

#val = foo(X,Y,Z)


# fourth dimention - colormap
# create colormap according to x-value (can use any 50x50 array)
color_dimension = val # change to desired fourth dimension
minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

# plot

fig = plt.figure()
ax = fig.gca(projection='3d')
line = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=True)

plt.axis('off')
ax.grid(False)

fig.canvas.draw()
plt.show()

#ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 1), init_func=init,
#                                  interval=200, blit=True)

#ani.save('test.mp4', writer='imagemagick', fps=30)

