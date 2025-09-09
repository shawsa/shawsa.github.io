# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

from numpy import pi, cos, sin, arccos, arange
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from spherepts import *

import matplotlib.animation as animation
def init():
    ax.view_init(azim=0)
    return line, 

def rotate(angle):
    ax.view_init(azim=angle)
    return line,

n = 400
n, nodes = gen_min_energy_nodes(n)

xs, ys, zs = nodes[:,0], nodes[:,1], nodes[:,2]


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')

line = ax.scatter(xs, ys, zs, 'b.')
plt.axis('off')
plt.title(str(n) + ' Minimum Energy Nodes')

#plt.show()

ani = animation.FuncAnimation(fig, rotate, frames=range(0, 360, 1), init_func=init,
                                  interval=200, blit=True)

ani.save('min_energy_n' + str(n) + '.mp4', writer='imagemagick', fps=30)
