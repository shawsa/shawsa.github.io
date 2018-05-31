import matplotlib.pyplot as plt
from halton import halton_sequence

# Halton Points
n = 100

xs, ys = halton_sequence(1,n,2)

plt.figure(figsize=(5,5))
plt.plot(xs, ys, 'bo')
plt.title(str(n) + ' Halton Points')
plt.show()

# Random Points
from random import random
n = 100

xs = [random() for i in range(n)]
ys = [random() for i in range(n)]

plt.figure(figsize=(5,5))
plt.plot(xs, ys, 'bo')
plt.title(str(n) + ' Random Points')
plt.show()
