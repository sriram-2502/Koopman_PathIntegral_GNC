function [Psi, DPsi] = monomial_basis_sin(deg, dim)

k = linspace(2, deg, deg-1);
d = dim;
x=sym('x',[d,1]);
assume(x,'real')

Psi = [x.'];
for i=1:size(k,2)
    m = nchoosek(k(i)+d-1,d-1);
    dividers = [zeros(m,1),nchoosek((1:(k(i)+d-1))',d-1),ones(m,1)*(k(i)+d)];
    a = diff(dividers,1,2)-1;
    for i = 1:size(a,1)
        Psi = [Psi prod(x.' .^ a(i,:))];
    end
end
DPsi = jacobian(Psi,x);
Psi = Psi'; % monomial only
% Psi = [Psi; sin(x(1))-x(1); x(2).*sin(x(1)); x(2).*cos(x(1))-x(2)]; % correct and good approximation
Psi = [Psi; sin(x(2));cos(x(1));sin(x(4));cos(x(3))]; % correct and good approximation
% Psi = [Psi; sin(x).*cos(x)]; % correct and good approximation
% Psi = [Psi; x(2)^2;x(4)^2 ;cos(x(1));cos(x(3));cos(x(1)-x(3));sin(x(2));sin(x(4))];


% x(2)^2+x(4)^2 -2*cos(x(1))-cos(x(3))- cos(x(1)-x(3))-0.1*x(3)
end
