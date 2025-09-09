"""Diagram of spatially organized network.
"""


import matplotlib.pyplot as plt
import numpy as np


nodes = np.arange(101)
ts = np.linspace(0, np.pi, 201)

plt.figure(figsize=(10, 4))
x = len(nodes)//2
plt.plot(nodes, 0*nodes, 'bo', markersize=20)
plt.plot(x, 0, 'ro', markersize=20)
for y in nodes:
    c = (x + y)/2
    r = abs(x - y)/2
    plt.plot(c + r*np.cos(ts), r/10*np.sin(ts), 'k-',
             linewidth=10*np.exp(-abs(r)/2))

plt.xlim(x-5, x+5)
plt.ylim(-.1, 1)
plt.axis('off')
plt.tight_layout()
plt.savefig('network_diagram.png')
plt.close()
