%% Forward Euler
ntests = 6;
N = 2.^(0:ntests-1)*10+1;
err = zeros(ntests,1);
H = 1./(N-1);
uexact = @(t,x) exp(-pi^2*t)*sin(pi*x);

for k = 1:length(N)
    n = N(k);
    x = linspace(0,1,n)';
    h = 1/(n-1);

    D = 1/h^2*spdiags(ones(n,1)*[1 -2 1],-1:1,n,n);
    D(1,1) = 1; D(1,2) = 0;
    D(n,n) = 1; D(n,n-1) = 0;
    
    tfinal = 0.1;
    dt = 0.5*h^2;  % Need for stability
    ntsteps = round(tfinal/dt);

    u = uexact(0,x);

    I = speye(n);

    for j = 1:ntsteps
        u = u + dt*D*u;
    end
    err(k) = norm(u-uexact(tfinal,x),2)/norm(uexact(tfinal,x),2);
end
loglog(H,err,'x-',H,H.^2,'r-'), legend('Rel. error','Second order')
xlabel('h'), ylabel('Error'), title('Forward Euler, \Delta t = 0.5h^2')
ratio = err(1:end-1)./err(2:end)

%% Backward Euler
ntests = 8;
N = 2.^(0:ntests-1)*10+1;
err = zeros(ntests,1);
H = 1./(N-1);
uexact = @(t,x) exp(-pi^2*t)*sin(pi*x);

for k = 1:length(N)
    n = N(k);
    x = linspace(0,1,n)';
    h = 1/(n-1);

    D = 1/h^2*spdiags(ones(n,1)*[1 -2 1],-1:1,n,n);
    % Remove boundar conditions
    D = D(2:n-1,2:n-1);
    
    tfinal = 0.1;
    dt = 0.25*h;  % No stability restrictions
    ntsteps = round(tfinal/dt);

    u = uexact(0,x(2:end-1));

    I = speye(n-2);

    for j = 1:ntsteps
        u = (I-dt*D)\u;
    end
    % Add boundary conditions back in
    u = [0;u;0];
    err(k) = norm(u-uexact(tfinal,x),2)/norm(uexact(tfinal,x),2);
end
loglog(H,err,'x-',H,H,'r-'), legend('Rel. error','First order')
xlabel('h'), ylabel('Error'), title('Backward Euler, \Delta t = h/4')
ratio = err(1:end-1)./err(2:end)

%% BDF 2

ntests = 8;
N = 2.^(0:ntests-1)*10+1;
err = zeros(ntests,1);
H = 1./(N-1);
uexact = @(t,x) exp(-pi^2*t)*sin(pi*x);

for k = 1:length(N)
    n = N(k);
    x = linspace(0,1,n)';
    h = 1/(n-1);

    D = 1/h^2*spdiags(ones(n,1)*[1 -2 1],-1:1,n,n);
    % Remove boundar conditions
    D = D(2:n-1,2:n-1);
    
    tfinal = 0.1;
    dt = 0.125*h;  % No stability restrictions
    ntsteps = round(tfinal/dt);

    u0 = uexact(0,x(2:end-1));

    I = speye(n-2);
    
    % One step of backward euler
    u = (I-dt*D)\u0;

    for j = 2:ntsteps
        temp = u;
        u = (I-dt*2/3*D)\(4/3*u - 1/3*u0);
        u0 = temp;
    end
    % Add boundary conditions back in
    u = [0;u;0];
    err(k) = norm(u-uexact(tfinal,x),2)/norm(uexact(tfinal,x),2);
end
loglog(H,err,'x-',H,H.^2,'r-'), legend('Rel. error','Second order')
xlabel('h'), ylabel('Error'), title('BDF2, \Delta t = h/4')
ratio = err(1:end-1)./err(2:end)
