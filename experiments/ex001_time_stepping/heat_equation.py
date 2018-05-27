import numpy as np
from scipy.integrate import quadrature as quad

def fourier_sin(init, L, n):
    Bn = [2/L * quad(lambda x: np.sin(i*np.pi/L * x)*init(x), 0, L)[0] for i in range(1,n+1)]
    return Bn
    

def dirichlet_from_series(x, t, c, L, Bn):
    n = len(Bn)
    terms = [np.sin(i*np.pi/L * x)*np.exp(-c * (i * np.pi / L)**2 * t) for i in range(1,n+1)]
    terms = np.array(terms).T
    return np.dot(terms, Bn)

def dirichlet_solution(c, init, L=1, n=20):
    Bn = fourier_sin(init, L, n)
    return lambda x,t, c=c, L=L, Bn=Bn: dirichlet_from_series(x,t,c,L,Bn)


if __name__=='__main__':
    def f(x):
        return x**2 * (1-x)
    u = dirichlet_solution(1,f)
    x = np.linspace(0,1)

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    plt.xlabel('Position')
    plt.ylabel('Temperature')
    plt.title('Temperature Distribution')

    line, = ax.plot(x, f(x))
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


    def animate(i):
        line.set_ydata(u(x,i/1000))  # update the data
        time_text.set_text('time (s) = %.3f' % (i/1000))
        return line, time_text


    # Init only required for blitting to give a clean slate.
    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        time_text.set_text('')
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                                  interval=25, blit=True)

    ani.save('test.mp4', writer='imagemagick', fps=30)
                                  
    plt.show()
