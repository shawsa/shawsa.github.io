

import matplotlib.pyplot as plt
import numpy as np

Delta = 10
y_star = 5
front = 3

plt.plot([-100, 100], [0, 0], 'k-')

plt.plot(front, 0, 'ko')
plt.text(front, .1, r'$\Delta_c t - \varepsilon \nu$')

plt.plot(0, 0, 'ko')
plt.text(0, -.1, '0')

plt.plot(front-y_star, 0, 'ko')
plt.text(front-y_star, .1, r'$\Delta_c t - \varepsilon \nu - y^{\ast}$')

plt.plot(-Delta, 0, 'ko')
plt.text(-Delta, -.1, r'$-\Delta$')


plt.xlim(-Delta*1.2, 7)
plt.ylim(-1.2, 1.2)

plt.plot([-Delta, 0], [-.5, -.5], 'bo-')
plt.text(-2*Delta/3, -.6, 'Active Region')

plt.plot([front - y_star, front], [.5, .5], 'go-')
plt.text(0, .6, 'Stimulus')

plt.axis('off')

plt.savefig('stim_diagram.png')
