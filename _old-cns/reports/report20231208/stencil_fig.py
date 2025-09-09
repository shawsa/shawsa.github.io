
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial


xs = np.arange(7)

interval = (2, 3)
stencil = np.array([1, 2, 3, 4])


def foo(x):
    return np.sin(x*2) + 5


zs = np.linspace(xs[0], xs[-1], 2001)
xis = np.linspace(*interval, 2001)


plt.figure(figsize=(10, 4))

plt.plot(zs, foo(zs), "b-", label="$f$")
plt.plot(stencil, foo(stencil), "bo")
plt.plot(stencil, 0*stencil, "ko", label="Stencil")
plt.plot(interval, [0, 0], "k-", label="$[x_i, x_{i+1}$]")

poly = Polynomial.fit(stencil, foo(stencil), deg=len(stencil)-1)

plt.plot(zs, poly(zs), "g:")
plt.plot(xis, poly(xis), "g-", label="interpolant")
plt.xlim(0, 5)
plt.ylim(-1, 7)
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.savefig("stencil.png")
