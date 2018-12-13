for k= 0:8
    [x] = getIcosNodes(k,0);
    n = size(x,1)
    %save(sprintf('eami%05d.mat',n), 'x')
end