import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def to_polar(vec):
    x,y,z = vec
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))
    return r, theta, phi

def f(pos):
    r, theta, phi = to_polar(pos)
    return np.cos(5*phi) * np.sin(2*theta)

# steps in each parameter
n = 200

u = np.linspace(0, 2 * np.pi, n)
v = np.linspace(0, np.pi, n)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones(np.size(u)), np.cos(v))

val = np.array([ f((x,y,z)) for x,y,z in zip(X,Y,Z)])

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
line = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)

plt.axis('off')
ax.grid(False)

#fig.canvas.draw()
#plt.show()

import matplotlib.animation as animation

def init():
    ax.view_init(azim=0)
    return line, 

def rotate(angle):
    ax.view_init(azim=angle)
    return line,

ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), init_func=init,
                                  interval=200, blit=True)

ani.save('test.mp4', writer='imagemagick', fps=30)

