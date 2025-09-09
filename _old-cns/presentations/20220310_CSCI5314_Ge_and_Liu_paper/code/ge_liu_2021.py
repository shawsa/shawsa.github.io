'''
An implementation of the model from *Foraging behaviours lead to 
spatiotemporal self-similar dynamics in grazing ecosystems* by 
Zhenpeng Ge and Quan-Xing Liu
https://doi.org/10.1111/ele.13928
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.sparse


def RK4update(t, u, F, Δt):
    k1 = F(t,u)
    k2 = F(t+Δt/2, u + Δt/2*k1)
    k3 = F(t+Δt/2, u + Δt/2*k2)
    k4 = F(t+Δt, u+Δt*k3)
    return u + Δt/6*(k1 + 2*k2 + 2*k3 + k4)

def euler_update(t, u, F, Δt):
    return u + Δt*F(t, u)

class Model:
    def __init__(self, α, β, κ, λ, D0, 
                 x_left, Δx, spatial_points,
                 t0, Δt,
                 stocking_rate):
        self.α = α
        self.β = β
        self.κ = κ
        self.λ = λ
        self.D0 = D0
        self.x_left = x_left
        self.Δx = Δx
        self.spatial_points = spatial_points
        self.t0 = t0
        self.Δt = Δt
        self.stocking_rate = stocking_rate
        
        self.re_initialize()
        
    def re_initialize(self):
        self.create_domain()
        self.create_fd_matrices()
        self.create_forcing()
        
    def create_forcing(self):
        def δx(Z):
            return (self.D@Z.T).T # same as Z@D.T but faster
        def δy(Z):
            return self.D@Z

        def grad(Z):
            g = np.zeros((2, *Z.shape))
            g[0] = δx(Z)
            g[1] = δy(Z)
            return g
        def divergence(Z):
            return δx(Z[0]) + δy(Z[1])
        def Δ(Z):
            return self.D2@Z + (self.D2@Z.T).T

        def norm(Z):
            return la.norm(Z, ord='fro')/spatial_points**2

        α = self.α
        β = self.β
        λ = self.λ
        κ = self.κ
        D0 = self.D0
        def F(t, u):
            p, h = u
            v = α*p**2 + β*p + 1
            vp = 2*α*p + β
            df = np.zeros_like(u)
            df[0] = λ*p*(1-p) - p*h + Δ(p)
            df[1] = D0*divergence( v**2*grad(h) + h*v*vp*grad(p)) - κ*Δ(Δ(h))
            return df
        
        self.F = F
        
    def create_domain(self):
        xs = np.linspace(self.x_left, 
                         self.x_left + self.Δx*self.spatial_points, 
                         self.spatial_points, 
                         endpoint=False)
        self.X, self.Y = np.meshgrid(xs, xs)
        
    def create_fd_matrices(self):
        shape = (self.spatial_points, self.spatial_points)
        fd_mat_data = ([-1]*self.spatial_points,
                       [-1]*self.spatial_points,
                       [ 1]*self.spatial_points,
                       [ 1]*self.spatial_points)
        fd_mat_offsets = (-1, self.spatial_points-1, 1, -self.spatial_points+1)
        self.D = scipy.sparse.csr_matrix(
                    scipy.sparse.dia_matrix(
                        (fd_mat_data, fd_mat_offsets),
                        shape = shape)/(2*self.Δx))

        fd_mat_data = ([ 1]*self.spatial_points,
                       [-2]*self.spatial_points,
                       [ 1]*self.spatial_points,
                       [ 1]*self.spatial_points,
                       [ 1]*self.spatial_points)
        fd_mat_offsets = (-1, 0, 1, self.spatial_points-1, -self.spatial_points+1)
        self.D2 = scipy.sparse.csr_matrix(
                    scipy.sparse.dia_matrix(
                        (fd_mat_data, fd_mat_offsets),
                        shape = shape)/self.Δx**2)
        
    def get_u0(self):
        shape = (2, self.spatial_points, self.spatial_points)
        u0 = 0.1*(2*np.random.rand(*shape) - 1)
        u0[0] += 1 - self.stocking_rate/self.λ
        u0[1] += self.stocking_rate
        return u0
    
    def simulate(self, stepper=euler_update):
        t = self.t0
        u = self.get_u0()
        yield t, u
        while True:
            u = stepper(t, u, self.F, self.Δt)
            t = t+self.Δt
            yield t, u


def plot_state(u, figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    p_map = axes[0].pcolormesh(u[0], shading='gouraud', cmap='Greens')
    h_map = axes[1].pcolormesh(u[1], shading='gouraud', cmap='copper')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(p_map, ax=axes[0], label='Plant Density')
    fig.colorbar(h_map, ax=axes[1], label='Herbivore Density')
    plt.show()

def animate(us, density_scale='unit', figsize=(12,5)):
    if density_scale == 'unit':
        p_max, p_min = 0, 1
        h_max, h_min = 0, 1
    elif density_scale == 'tight':
        p_min = min(np.min(u[0]) for u in us)
        p_max = max(np.max(u[0]) for u in us)
        h_min = min(np.min(u[1]) for u in us)
        h_max = max(np.max(u[1]) for u in us)
    else:
        p_min, p_max, h_min, h_max = density_scale

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_params = {'shading': 'gouraud'}
    p_map = axes[0].pcolormesh(us[0][0], cmap='Greens', **plot_params, vmin=p_min, vmax=p_max)
    h_map = axes[1].pcolormesh(us[0][1], cmap='copper', **plot_params, vmin=h_min, vmax=h_max)

    fig.colorbar(p_map, ax=axes[0], label='Plant Density')
    fig.colorbar(h_map, ax=axes[1], label='Herbivore Density')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    def animate(i):
        print(f'{i}/{len(us)}' + ' '*100, end='\r')
        p_map.set_array(us[i][0].flatten())
        h_map.set_array(us[i][1].flatten())
        return h_map,


    # Init only required for blitting to give a clean slate.
    def init():
        p_map.set_array(us[0][0].flatten())
        h_map.set_array(us[0][1].flatten())
        return h_map,

    plt.close()
    return animation.FuncAnimation(fig, animate, range(len(us)), init_func=init,
                                  interval=1/24*1000, blit=True)

