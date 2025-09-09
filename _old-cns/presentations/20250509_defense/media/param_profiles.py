import numpy as np
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 24,
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)

theta = 0.3

figsize = (8, 3)
fig = plt.figure("function_params", figsize=figsize)

grid = gs.GridSpec(1, 2)

f_ax = fig.add_subplot(grid[0, 0])
f_ax.plot([0, theta], [0, 0], "r-")
f_ax.plot([theta, 1], [1, 1], "r-")
f_ax.plot([theta, theta], [-1, 2], "k:")
f_ax.set_ylim(-0.1, 1.1)
f_ax.set_xlim(0, 1)
f_ax.set_xticks([0, theta, 1], ["0", "$\\theta$", "1"])
f_ax.set_xlabel("$u$")
f_ax.set_title("$f[u]$")
f_ax.set_yticks([0, 1])

xs = np.linspace(-5, 5, 201)

w_ax = fig.add_subplot(grid[0, 1], sharey=f_ax)
w_ax.plot(xs, 0.5*np.exp(-np.abs(xs)), "g-")
w_ax.set_xticks([0])
w_ax.set_xlabel("$x - y$")
w_ax.set_xlim(xs[0], xs[-1])
w_ax.set_yticks([])
w_ax.set_title("$w(x, y)$")

plt.tight_layout()
plt.savefig("function_params.png")
