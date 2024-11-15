
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 16,
        "text.usetex": True,
    }
)


rs = np.linspace(0, 4, 201)

rs_abs = np.abs(rs)

plt.figure(figsize=(4, 4))
plt.plot(rs, 0*rs, "k-")
plt.plot([0, 0], [0, 1], "k-")
plt.plot(rs, np.exp(-rs_abs), label="excitatory")
plt.plot(rs, np.exp(-(rs_abs**2)), label="smooth")
plt.plot(rs, np.exp(-rs_abs) * (2-rs_abs)/2, label="lateral inhibition")
plt.axis("off")
plt.text(np.average(rs), -.1, "$r$")
plt.title("$w(x, y) = w(||x - y||) = w(r)$")
plt.legend()
plt.tight_layout()
plt.savefig("kernels.eps")
