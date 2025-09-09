"""An animation demonstrating the phenomenon of apparent
motion.
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm

# Apparent motion comparison
FILE_NAME = "apparent_motion_demo.gif"

dist_fast = 1 / 31
dist_slow = 1 / 5


def effective_time(t, dist):
    return np.floor(t / dist) * dist


def is_on(t, dist):
    return (t - np.floor(t / dist) * dist) / dist < 0.5


ts = np.linspace(0, 1, 151)

fig = plt.figure(figsize=(7, 3))
plt.tight_layout()
(smooth,) = plt.plot(0, 2, "ro", markersize=30)
fast_y = 1
slow_y = 0
(fast,) = plt.plot(0, fast_y, "ro", markersize=30)
(slow,) = plt.plot(0, slow_y, "ro", markersize=30)
plt.text(-0.9, 1.85, "Motion", fontsize=20)
plt.text(-0.9, 0.85, "Apparent Motion", fontsize=20)
plt.text(-0.9, -0.15, "Discrete", fontsize=20)
plt.xlim(-1.1, 1.2)
plt.ylim(-0.3, 2.3)
plt.axis("off")
plt.tight_layout()
with imageio.get_writer(FILE_NAME, mode="I", duration=0.02) as writer:
    for t in ts:
        t_fast = effective_time(t, dist_fast)
        t_slow = effective_time(t, dist_slow)
        smooth.set_xdata(t)
        fast.set_xdata(effective_time(t, dist_fast))
        slow.set_xdata(effective_time(t, dist_slow))

        if is_on(t, dist_fast):
            fast.set_ydata(fast_y)
        else:
            fast.set_ydata(-100)

        if is_on(t, dist_slow):
            slow.set_ydata(slow_y)
        else:
            slow.set_ydata(-100)

        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(0.01)

plt.close()
os.remove(FILE_NAME + ".png")

# Constant moving
FILE_NAME = "smooth_dot.gif"
dist_fast = 1 / 31
dist_slow = 1 / 5
ts = np.linspace(0, 1, 151)

fig = plt.figure(figsize=(7, 3))
plt.tight_layout()
(smooth,) = plt.plot(0, 2, "ro", markersize=30)
plt.xlim(-0.1, 1.2)
# plt.ylim(-1.1, 2.3)
plt.axis("off")
plt.tight_layout()
with imageio.get_writer(FILE_NAME, mode="I", duration=0.02) as writer:
    for t in tqdm(ts):
        smooth.set_xdata(t)
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(0.01)

plt.close()
os.remove(FILE_NAME + ".png")

# Flashing
FILE_NAME = "flashing_dot.gif"
dist_fast = 1 / 31
dist_slow = 1 / 5
ts = np.linspace(0, 1, 151)

freq = 20


def ydata(t):
    return np.sign(np.sin(t * 2 * np.pi * freq))


fig = plt.figure(figsize=(7, 3))
plt.tight_layout()
(smooth,) = plt.plot(0, 0, "ro", markersize=30)
plt.xlim(-0.1, 1.2)
plt.ylim(0.9, 1.1)
plt.axis("off")
plt.tight_layout()
with imageio.get_writer(FILE_NAME, mode="I", duration=0.02) as writer:
    for t in tqdm(ts):
        smooth.set_xdata(t)
        smooth.set_ydata(ydata(t))
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(0.01)

plt.close()
os.remove(FILE_NAME + ".png")

# Apparent
FILE_NAME = "apparent_dot.gif"
ts = np.linspace(0, 1, 151)
freq = 20


def effective_time(t):
    return np.floor(t * freq) / freq


fig = plt.figure(figsize=(7, 3))
plt.tight_layout()
(smooth,) = plt.plot(0, 1, "ro", markersize=30)
plt.xlim(-0.1, 1.2)
plt.ylim(0.9, 1.1)
plt.axis("off")
plt.tight_layout()
with imageio.get_writer(FILE_NAME, mode="I", duration=0.02) as writer:
    for t in tqdm(ts):
        smooth.set_xdata(effective_time(t))
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(0.01)

plt.close()
os.remove(FILE_NAME + ".png")
