%% ========================================================================
% Flexible Koopman Path-Integral Analysis for Planar CR3BP
% - Works for any mass parameter mu in (0,1)
% - Example: mu = 0.0121505856   (Earth–Moon)
%            mu = 3.003e-6       (Sun–Earth)
% ========================================================================
clc; clear; close all;

set(groot,'defaultAxesFontSize',16)
set(groot,'defaultTextFontSize',16)
addpath('sampling')

%% ---------------------------- Mass parameter ----------------------------
% Choose mu here:
% mu = 0.0121505856;   % Earth–Moon
mu = 3.003e-6;     % Sun–Earth

% Primary locations in rotating frame
x_primary1 = -mu;       % large primary
x_primary2 = 1 - mu;    % small primary

% Label primaries depending on mu
if mu > 1e-4
    label1 = 'Earth'; label2 = 'Moon';
else
    label1 = 'Sun';   label2 = 'Earth';
end

%% ---------------- Equilibria (L1..L5) -----------------------------------
gradOmega2D = @(x,y) [ ...
    x - (1-mu)*(x+mu)./((x+mu).^2+y.^2).^(3/2) - mu*(x-1+mu)./((x-1+mu).^2+y.^2).^(3/2); ...
    y - (1-mu)*y./((x+mu).^2+y.^2).^(3/2)      - mu*y./((x-1+mu).^2+y.^2).^(3/2) ];

gx_only = @(x) x ...
    - (1-mu)*(x+mu)./(abs(x+mu).^3) ...
    - mu*(x-1+mu)./(abs(x-1+mu).^3);

opts_fzero = optimset('TolX',1e-14,'Display','off');
x_m1 = -mu; x_m2 = 1-mu; epsx = 1e-6;
xL1 = bracket_then_fzero(gx_only, [x_m1+10*epsx, x_m2-10*epsx], opts_fzero);
xL2 = bracket_then_fzero(gx_only, [x_m2+10*epsx, x_m2+5],       opts_fzero);
xL3 = bracket_then_fzero(gx_only, [x_m1-5, x_m1-10*epsx],       opts_fzero);

L1 = [xL1;0;0;0]; L2 = [xL2;0;0;0]; L3 = [xL3;0;0;0];

opts_fsolve = optimoptions('fsolve','Display','off', ...
    'FunctionTolerance',1e-14,'StepTolerance',1e-14,'OptimalityTolerance',1e-14);
xg = 0.5-mu; yg = sqrt(3)/2;
xyL4 = fsolve(@(xy) gradOmega2D(xy(1),xy(2)), [xg; yg],  opts_fsolve);
xyL5 = fsolve(@(xy) gradOmega2D(xy(1),xy(2)), [xg;-yg], opts_fsolve);
L4 = [xyL4(1);0;xyL4(2);0]; L5 = [xyL5(1);0;xyL5(2);0];

x0_u = [L1.'; L2.'; L3.'; L4.'; L5.'];   % pack 5x4
labels = {'L1','L2','L3','L4','L5'};

%% ---------------- Adjust plotting window --------------------------------
x_all = [x0_u(:,1); x_primary1; x_primary2];
y_all = [x0_u(:,3); 0; 0];
margin = 0.2 * (max(x_all)-min(x_all));   % 20% padding
Dom = [min(x_all)-margin, max(x_all)+margin];
DomY = [min(y_all)-margin, max(y_all)+margin];

%% ---------------- Vector field and Jacobian -----------------------------
n=4; syms x1 x2 x3 x4 real; xs=[x1;x2;x3;x4];
r1=sqrt((x1+mu)^2+x3^2); r2=sqrt((x1-1+mu)^2+x3^2);
dOdx = x1 - (1-mu)*(x1+mu)/r1^3 - mu*(x1-1+mu)/r2^3;
dOdy = x3 - (1-mu)*x3/r1^3     - mu*x3/r2^3;
f_sym=[x2; 2*x4+dOdx; x4; -2*x2+dOdy];
f_scalar=matlabFunction(f_sym,'Vars',{x1,x2,x3,x4});
Jfx_scalar=matlabFunction(jacobian(f_sym,xs),'Vars',{x1,x2,x3,x4});
f_vec=@(X) f_scalar(X(1),X(2),X(3),X(4));
Jfx_at=@(X) Jfx_scalar(X(1),X(2),X(3),X(4));

%% ---------------- Koopman implicit curves -------------------------------
figure(1); clf; hold on;
optsEv=odeset('RelTol',1e-6,'AbsTol',1e-9,'Events',@(t,y) offFrame(t,y,max(abs(Dom))));

colors = lines(5);

for j=1:5
    x_star=x0_u(j,:).';
    A=Jfx_at(x_star);
    [V,D,W]=eig(A); lam=diag(D);
    [~,idx]=max(real(lam));
    lambda_u=lam(idx); w_u=W(:,idx); v_u=V(:,idx);
    w_u=w_u/(w_u.'*v_u);

    r_fun=@(X) f_vec(X)-A*(X-x_star);
    g_fun=@(X) real(w_u.'*r_fun(X));

    [x_0n,t_0n]=backward_sample_box(x_star.',@(xrow) f_vec(xrow.'), ...
        'NumPoints',200,'HalfSize',0.05,'Dom',Dom, ...
        'Dt',0.01,'Tmax',10,'RelTol',1e-6,'AbsTol',1e-9);

    Phi_vals=zeros(1,size(x_0n,1));
    for k=1:size(x_0n,1)
        if t_0n(k)==0, tspan=[0 0.1]; else, tspan=[0 t_0n(k)/64]; end
        xk0=x_0n(k,:).';
        [t,X]=ode45(@(t,xx) f_vec(xx),tspan,xk0,optsEv);
        t=t(:); decay=exp(-real(lambda_u)*t);
        gx=arrayfun(@(i) g_fun(X(i,:).'),1:numel(t));
        Phi_vals(k)=real(w_u.'*(xk0-x_star))+trapz(t,decay.*gx(:));
    end

    deg=1; syms x [n 1] real
    [Psi_sym,~]=monomial_basis_sin(deg,n);
    Psi_fun=matlabFunction(Psi_sym,'Vars',{x});
    Psi_grid=zeros(length(Psi_sym),size(x_0n,1));
    for k=1:size(x_0n,1), Psi_grid(:,k)=Psi_fun(x_0n(k,:).'); end
    Q=(Phi_vals)*pinv(Psi_grid);
    Phi_approx=Q*Psi_sym;

    Phi_fun4=matlabFunction(Phi_approx,'Vars',{x1,x2,x3,x4});
    slice_fun=@(X1,X3) Phi_fun4(X1,0,X3,0);

    % Plot implicit curve in its unique color
    fimplicit(slice_fun,[Dom DomY],'Color',colors(j,:),'LineWidth',2);

    % Plot the corresponding L-point in same color
    plot(x0_u(j,1),x0_u(j,3),'o','Color',colors(j,:), ...
        'MarkerFaceColor',colors(j,:),'MarkerSize',8);
    text(x0_u(j,1)+0.02,x0_u(j,3)+0.02,labels{j}, ...
        'FontSize',12,'FontWeight','bold','Color',colors(j,:));

        % Plot implicit curve in its unique color
    fimplicit(slice_fun,[Dom DomY],'Color',colors(j,:),'LineWidth',2);

    % Plot the corresponding L-point in same color
    plot(x0_u(j,1),x0_u(j,3),'o','Color',colors(j,:), ...
        'MarkerFaceColor',colors(j,:),'MarkerSize',8);
    text(x0_u(j,1)+0.02,x0_u(j,3)+0.02,labels{j}, ...
        'FontSize',12,'FontWeight','bold','Color',colors(j,:));

    % -------- One sample trajectory near this L-point -----------------
%     perturb = 1e-3*randn(4,1);          % small random perturbation
%     [t,X] = ode45(@(t,xx) f_vec(xx),[0 50],x_star+perturb);
%     plot(X(:,1),X(:,3),'--','Color',colors(j,:),'LineWidth',1.2);

end

% Primaries
plot(x_primary1,0,'ko','MarkerFaceColor','b','MarkerSize',10); text(x_primary1,0.05,label1);
plot(x_primary2,0,'ko','MarkerFaceColor','r','MarkerSize',8);  text(x_primary2,0.1,label2);

xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
title('Koopman implicit curves (slice $v_x=v_y=0$)','Interpreter','latex');
axis equal; box on; grid minor;

%% ---------------- Trajectories near L-points ----------------------------
figure(2); clf; hold on;
for j=1:5
    x_star=x0_u(j,:).';
    for s=1:3
        perturb=1e-3*randn(4,1);
        [t,X]=ode45(@(t,xx) f_vec(xx),[0 50],x_star+perturb);
        plot(X(:,1),X(:,3),'Color',colors(j,:));
    end
    % Mark L-points in matching color
    plot(x0_u(j,1),x0_u(j,3),'o','Color',colors(j,:), ...
        'MarkerFaceColor',colors(j,:),'MarkerSize',8,'LineWidth',1.25);
    text(x0_u(j,1)+0.02,x0_u(j,3)+0.02,labels{j}, ...
        'FontSize',12,'FontWeight','bold','Color',colors(j,:));
end

% Primaries
plot(x_primary1,0,'ko','MarkerFaceColor','b','MarkerSize',10); text(x_primary1,0.05,label1);
plot(x_primary2,0,'ko','MarkerFaceColor','r','MarkerSize',8);  text(x_primary2,0.1,label2);
xlim([-1.5 1.5])
ylim([-1.5 1.5])

xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
title('Trajectories near Lagrange points','Interpreter','latex');
axis equal; box on; grid minor;

%% ============================== Helpers =================================
function [value,isterminal,direction]=offFrame(~,Y,DomMax)
value=double(max(abs(Y))>4*DomMax || norm(Y,1)<1e-3);
isterminal=1; direction=0;
end

function xr=bracket_then_fzero(g,interval,opts)
a=interval(1); b=interval(2); N=800;
xs=linspace(a,b,N+1); gs=arrayfun(g,xs);
valid=isfinite(gs); xs=xs(valid); gs=gs(valid);
idx=find(sign(gs(1:end-1)).*sign(gs(2:end))<0,1,'first');
if ~isempty(idx), xr=fzero(g,[xs(idx),xs(idx+1)],opts); return; end
[~,k]=min(abs(gs)); x0=xs(k); w=1e-3*(b-a);
for j=1:12
    aa=max(a,x0-w); bb=min(b,x0+w);
    ga=g(aa); gb=g(bb);
    if isfinite(ga)&&isfinite(gb)&&sign(ga)~=sign(gb)
        xr=fzero(g,[aa,bb],opts); return;
    end
    w=2*w;
end
error('No sign change in [%g,%g]',a,b);
end
