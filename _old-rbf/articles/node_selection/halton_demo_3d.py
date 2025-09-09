import matplotlib.pyplot as plt
from random import random
from halton import halton_sequence
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation
def init():
    ax.view_init(azim=0)
    return line, 

def rotate(angle):
    ax.view_init(azim=angle)
    return line,

# Halton Points
n = 400

xs, ys, zs = halton_sequence(1,n,3)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')

line = ax.scatter(xs, ys, zs, 'b.')
plt.axis('off')
plt.title(str(n) + ' Halton Points')

import matplotlib.animation as animation

ani = animation.FuncAnimation(fig, rotate, frames=range(0, 360, 2), init_func=init,
                                  interval=200, blit=True)

ani.save('halton_3d.mp4', writer='imagemagick', fps=30)


# Random Points

xs = [random() for i in range(n)]
ys = [random() for i in range(n)]
zs = [random() for i in range(n)]

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')

line = ax.scatter(xs, ys, zs, 'b.')
plt.axis('off')
plt.title(str(n) + ' Random Points')

ani = animation.FuncAnimation(fig, rotate, frames=range(0, 360, 2), init_func=init,
                                  interval=200, blit=True)

ani.save('random_3d.mp4', writer='imagemagick', fps=30)
