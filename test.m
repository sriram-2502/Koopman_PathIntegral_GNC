clc; clear; close all

% --- default plot options ---
set(0,'DefaultLineLineWidth',2)
set(0,'DefaultAxesLineWidth',2)
set(0,'DefaultAxesFontSize',20)
set(0,'defaultfigurecolor',[1 1 1])

% --- add paths ---
addpath('dynamics')
addpath('compute_eigfuns')

% --- params ---
show_diagnositcs = false;
sys_params.use_stable   = false; % locally stable
sys_params.use_unstable = false; % locally unstable

%% ================== System setup ==================
sys_info  = duffing_info(sys_params);
dynamics  = @dynamics_duffing;
n_states  = sys_info.state_dim;
n_ctrl    = sys_info.ctrl_dim;
x         = sym('x',[n_states;1],'real');
u         = sym('u',[n_ctrl;1],'real');   
dx_dxt    = dynamics(x, u, sys_info);    
A = sys_info.A;
B = sys_info.B;
W = sys_info.eig_vectors;
D = sys_info.eig_vals;

if all(round(diag(D)) > 0)
    disp('---- Eqb point is unstable -----')
elseif all(round(diag(D)) < 0)
    disp('---- Eqb point is stable -----')
else
    disp('---- Eqb point is saddle ----')
end

%% ================== (Grid sweep: follow your setup exactly ==================
dim = 1; % dimension for integraton (1 for scalar)
Dom = [2,2];
bounds = Dom(2);
grid = -bounds:0.05:bounds; %define grid where eigenfunction is well defined
[q1,q2] = meshgrid(grid);

x_0 = [q1(:),q2(:)]; 
phi1=[];phi2=[];

% stable eigenfunction for saddle point
w_bar = waitbar(0,'1','Name','Calcualting path integral...',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');

options = odeset('RelTol',1e-9,'AbsTol',1e-300,'events',@(t, x)offFrame(t, x, Dom(2)));
%options = odeset('events',@(t, x)offFrame(t, x, Dom(2)));
%options = odeset('RelTol',1e-9,'AbsTol',1e-300);

for i = 1:length(x_0)
    waitbar(i/length(x_0),w_bar,sprintf(string(i)+'/'+string(length(x_0))))
    
    % grid IC at index i
    x    = x_0(i,:);                        % 1x2
    phi = compute_path_integrals(x', dynamics, sys_info);
    phi_x  = phi.phi_x_op(:);               % column vector [φ1; φ2; ...]

    % log
    phi1 = [phi1, phi_x(1)];
    phi2 = [phi2, phi_x(2)];

end
F = findall(0,'type','figure','tag','TMWWaitbar');
delete(F);

%% reshape
% phi for saddle at (0,0)
phi1 = reshape((phi1),size(q2)); %phi1_saddle(phi1_saddle>10)=10;
phi2 = reshape((phi2),size(q2)); %phi2_saddle(phi2_saddle>10)=10;

%% plots
% ---------- Vector field for streamslice overlay (u = 0) ----------
X = q1; Y = q2;
U = zeros(size(X));
V = zeros(size(Y));
for r = 1:size(X,1)
    for c = 1:size(X,2)
        dx = dynamics([X(r,c); Y(r,c)], 0, sys_info);
        U(r,c) = double(dx(1));
        V(r,c) = double(dx(2));
    end
end

figure(2)
subplot(1,2,1)
p1 = pcolor(q1,q2,phi1); hold on;
set(p1,'Edgecolor','none')
colormap jet
l = streamslice(X,Y,U,V); hold on;
set(l,'LineWidth',1)
set(l,'Color','k');
axes1 = gca;
axis square
axis([-bounds bounds -bounds bounds])
set(axes1,'FontSize',15);
xlabel('$x_1$','FontSize',20, 'Interpreter','latex')
ylabel('$x_2$','FontSize',20, 'Interpreter','latex')
title('Stable eigenfunction $\phi_1(x)$','Interpreter','latex');
colorbar
%clim([-5e10, 5e10])
box on
axes.LineWidth=2;

subplot(1,2,2)
% remove max and min value from the eqb points (-1,0) and (1,0)
% [min_val, min_idx] = min(phi2,[],'all');
% [r,c] = ind2sub(size(q1), min_idx);
% phi2(r,c) = nan;
% [max_val, max_idx] = max(phi2,[],'all');
% [r,c] = ind2sub(size(q1), max_idx);
% phi2(r,c) = nan;

p2 = pcolor(q1,q2,phi2); hold on;
set(p2,'Edgecolor','none')
colormap jet
l = streamslice(X,Y,U,V); hold on;
set(l,'LineWidth',1)
set(l,'Color','k');
axes2 = gca;
axis square
axis([-bounds bounds -bounds bounds])
set(axes2,'FontSize',15);
xlabel('$x_1$','FontSize',20, 'Interpreter','latex')
ylabel('$x_2$','FontSize',20, 'Interpreter','latex')
colorbar
title('Unstable eigenfunction $\phi_2(x)$','Interpreter','latex');
% clim([-5e-4,5e-4])
box on
axes.LineWidth=2;

%% zero level set
phi2_level_set = phi2;
phi2_level_set(abs(phi2_level_set)<10-5)=0;
phi2_level_set(abs(phi2_level_set)>10e-4)=-100;

subplot(2,2,4)
p2 = pcolor(q1,q2,phi2_level_set); hold on;
set(p2,'Edgecolor','none')
colormap jet
l = streamslice(X,Y,u,v); hold on;
set(l,'LineWidth',1)
set(l,'Color','k');
f = @(t,x)[x(2); -delta*x(2) - x(1)^3 + x(1)]; 
xl = -bounds; xh = bounds;
yl = -bounds; yh = bounds;
for x0 = linspace(-bounds, bounds, ic_pts)
    for y0 = linspace(-bounds, bounds, ic_pts)
        [ts,xs] = ode45(@(t,x)f(t,x),tspan,[x0 y0]);
        plot(xs(:,1),xs(:,2),'k','LineWidth',1); hold on;
    end
end

axes2 = gca;
axis square
axis([-bounds bounds -bounds bounds])
set(axes2,'FontSize',15);
xlabel('$x_1$','FontSize',20, 'Interpreter','latex')
ylabel('$x_2$','FontSize',20, 'Interpreter','latex')
%title ('Unstable eigenfunction $\psi_2(x)$ at (0,0)','FontSize',20, 'Interpreter','latex')
colorbar
clim([-5e-4, 5e-4])
box on
axes.LineWidth=2;