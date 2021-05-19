import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 16})
##################################################################
#
# Forward Euler
#
##################################################################
plt.figure(figsize=(7,7))
plt.xlabel('Real')
plt.ylabel('Imaginary')
# axis
plt.plot((-10, 10), (0,0), 'k-')
plt.plot((0,0), (-10,10), 'k-')


t = np.linspace(0,np.pi, 20)
x1 = np.cos(t) - 1
y1 = np.sin(t)
t2 = np.linspace(np.pi,2*np.pi)
x2 = np.cos(t + np.pi) - 1
y2 = np.sin(t + np.pi)

#plt.plot(x1,y1, ':b')
#plt.plot(x2,y2, '--b')
plt.fill_between(x1, y1, y2, hatch='/', facecolor='none', edgecolor='b', linestyle='--')

plt.xlim((-3,3))
plt.ylim((-3,3))

plt.title('Stability Domain for Forward Euler')

plt.show()


##################################################################
#
# Backward Euler
#
##################################################################

plt.figure(figsize=(7,7))
plt.xlabel('Real')
plt.ylabel('Imaginary')
# axis
plt.plot((-10, 10), (0,0), 'k-')
plt.plot((0,0), (-10,10), 'k-')

t = np.linspace(0,np.pi)
x1 = np.cos(t) + 1
y1 = np.sin(t)
t2 = np.linspace(np.pi,2*np.pi)
x2 = np.cos(t + np.pi) + 1
y2 = np.sin(t + np.pi)

#plt.plot(x1,y1, 'b--')
#plt.plot(x2,y2, 'b--')

x1_shade = np.append([10], np.append(x1,[-10]))
y1_shade = np.append([0], np.append(y1,[0]))
y2_shade = np.append([0], np.append(y2,[0]))

y_top = [10]*len(x1_shade)
y_bot = [-10]*len(x1_shade)

plt.fill_between(x1_shade, y1_shade, y_top, hatch='/', facecolor='none', edgecolor='b', linestyle='--')
plt.fill_between(x1_shade, y2_shade, y_bot, hatch='/', facecolor='none', edgecolor='b', linestyle='--')

plt.xlim((-3,3))
plt.ylim((-3,3))

plt.title('Stability Domain for Backward Euler')

plt.show()
