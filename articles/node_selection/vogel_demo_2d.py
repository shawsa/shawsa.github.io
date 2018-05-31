import matplotlib.pyplot as plt
from math import sqrt, cos, sin, pi
import numpy as np

n = 100

theta = pi*(3-sqrt(5))
theta = pi*(sqrt(5)-1)
xs = [sqrt(i/n)*cos(i*theta) for i in range(n)]
ys = [sqrt(i/n)*sin(i*theta) for i in range(n)]

plt.figure(figsize=(5,5))
plt.plot(xs, ys, 'bo')
plt.title(str(n) + ' Vogel Nodes')
plt.show()

N = n*100
ts = np.linspace(0, n, N)
txs = [sqrt(i/n)*cos(i*theta) for i in ts]
tys = [sqrt(i/n)*sin(i*theta) for i in ts]

plt.figure(figsize=(5,5))
plt.plot(xs, ys, 'bo')
plt.plot(txs, tys, 'r-')
plt.title('Spiral Generatation')
plt.show()
