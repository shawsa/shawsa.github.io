function [u,lapu] = sphereForcing(x,y,z)
% sphereForcing  Laplacian of 23 randomly placed Gaussians over the sphere.
% u = Gaussians sampled at x, y, and z.
% lapU = Surface Laplacian of u

% Number of Gaussians.
K = 23;

% Define the centers of the Gaussian.
rng(11231974);
lam = (2*rand(K,1)-1)*pi;
th = (2*rand(K,1)-1)*pi/2;
[xc,yc,zc] = sph2cart(lam,th,1+0*lam);

% Scaling for the Gaussians.
sig = 10;

lapu = 0*x;
u = lapu;

for k = 1:K
    % Distance (squared) from [x,y,z] to center of the Gaussian
    r2 = 2*abs(1 - x*xc(k) - y*yc(k) - z*zc(k));
    temp = exp(-sig*r2);
    u = u + temp;
    lapu = lapu - temp.*sig.*(4+r2.*(-2 + (-4 + r2)*sig));
end
end