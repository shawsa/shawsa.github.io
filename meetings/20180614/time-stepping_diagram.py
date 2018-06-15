import matplotlib.pyplot as plt

plt.figure(figsize=(4,1))
plt.axis('off')

plt.vlines([0, 1, 2], -.1, .1)
plt.plot([0, 2], [0,0], 'k-')
plt.xlim((-.5, 2.5))
plt.ylim((-.5, .5))
plt.text(0, -.11, '$t_{n-2}$', horizontalalignment='center', verticalalignment='top')
plt.text(1, -.11, '$t_{n-1}$', horizontalalignment='center', verticalalignment='top')
plt.text(2, -.11, '$t_{n}$', horizontalalignment='center', verticalalignment='top')
plt.show()
