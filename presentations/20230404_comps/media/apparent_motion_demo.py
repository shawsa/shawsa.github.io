"""An animation demonstrating the phenomenon of apparent
motion.
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os

FILE_NAME = 'apparent_motion_demo.gif'

dist_fast = 1/31
dist_slow = 1/5
def effective_time(t, dist):
    return np.floor(t/dist)*dist

ts = np.linspace(0, 1, 151)

fig = plt.figure(figsize=(7, 3))
plt.tight_layout()
smooth, = plt.plot(0, 2, 'ro', markersize=30)
fast, = plt.plot(0, 1, 'ro', markersize=30)
slow, = plt.plot(0, 0, 'ro', markersize=30)
plt.text(-.9, 1.85, 'Motion', fontsize=20)
plt.text(-.9, 0.85, 'Apparent Motion', fontsize=20)
plt.text(-.9, -0.15, 'Discrete', fontsize=20)
plt.xlim(-1.1, 1.2)
plt.ylim(-.3, 2.3)
plt.axis('off')
plt.tight_layout()
with imageio.get_writer(FILE_NAME, mode='I', duration=.02) as writer:
    for t in ts:
        t_fast = effective_time(t, dist_fast)
        t_slow = effective_time(t, dist_slow)
        smooth.set_xdata(t)
        fast.set_xdata(effective_time(t, dist_fast))
        slow.set_xdata(effective_time(t, dist_slow))
        plt.savefig(FILE_NAME + '.png')
        image = imageio.imread(FILE_NAME + '.png')
        writer.append_data(image)
        plt.pause(0.01)

plt.close()
os.remove(FILE_NAME + '.png')
