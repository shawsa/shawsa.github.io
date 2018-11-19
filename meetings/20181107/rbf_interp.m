runs = 10;

a = 0;
b = 2*pi;

n = 10^3;
m = 10^5;
eps = 45;

x = linspace(a,b,n)';
z = linspace(a,b,m)';

y = sin(x);

times = zeros(runs, 1);

for i=1:runs

    tic;
    A = exp(-eps*abs(x - x').^2);
    c = A\y;
    A = exp(-eps*abs(z - x').^2);
    u =  A * c;

    times(i) = toc;
end

min(times)

%hold on
%plot(z, sin(z), 'b-')
%plot(z, u, 'r--')
%hold off