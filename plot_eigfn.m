%% ========================================================================
% Standalone: Uniform 4D Grid ([-2,2]^4) + Koopman Eigenfunction Maps
% Planar CR3BP (state: [x; vx; y; vy])
%
% - Builds a UNIFORM 4D grid in each coordinate over [-2, 2]
% - Chooses a 2D SLICE from that 4D grid (default vx=vy=0) for evaluation
% - For each equilibrium (L1..L5):
%     * Linearize to get eigenpairs (lambda_j, v_j, w_j)
%     * For each (x,y) on the chosen slice, integrate the true nonlinear
%       CR3BP briefly and evaluate a variation-of-constants eigenfunction
%     * Plot surf(x,y) of Phi_j(x,y) on that slice

% ========================================================================
clear; clc; close all;

%% -------------------------- User configuration ---------------------------
% Mass parameter:
% mu = 0.0121505856;           % Earth–Moon
mu = 3.003e-6;                 % Sun–Earth (default)
% Primaries in rotating frame
x_primary1 = -mu;        % larger primary
x_primary2 = 1 - mu;     % smaller primary

% Labels based on mu
if mu > 1e-4
    label1 = 'Earth'; label2 = 'Moon';
else
    label1 = 'Sun';   label2 = 'Earth';
end


% 4D uniform grid limits and resolution (same for all 4 coordinates)
box_min = -2.0;  box_max = 2.0;
Dom = [box_min, box_max];
N1D     = 101;                      % points per coordinate (odd keeps 0 on-grid)
x_grid  = linspace(box_min, box_max, N1D);
vx_grid = linspace(box_min, box_max, N1D);
y_grid  = linspace(box_min, box_max, N1D);
vy_grid = linspace(box_min, box_max, N1D);

% Choose the 2D slice FROM the 4D grid (pick target values in vx, vy)
vx_slice_target = 0.0;
vy_slice_target = 0.0;

% Short forward integration for variation-of-constants
t_int  = 0.15;                    % horizon
RelTol = 1e-7; AbsTol = 1e-9;     % ODE tolerances

% Classification threshold for "neutral" eigenvalues
tol_neutral = 1e-6;

% What to show from complex Phi: 'real' | 'imag' | 'abs'
plot_part = 'abs';

% Colormap
cmap = parula;

set(groot,'defaultAxesFontSize',12);
set(groot,'defaultTextInterpreter','latex');

% ODE options
optsEv=odeset('RelTol',1e-6,'AbsTol',1e-9,'Events',@(t,y) offFrame(t,y,max(abs(Dom))));

%% ------------------- Helper: nearest grid index for slice ----------------
[~, vx_idx] = min(abs(vx_grid - vx_slice_target));
[~, vy_idx] = min(abs(vy_grid - vy_slice_target));
vx_slice = vx_grid(vx_idx);    % actual slice value (exactly on uniform grid)
vy_slice = vy_grid(vy_idx);

fprintf('Using slice: vx = %.4f (grid idx %d), vy = %.4f (grid idx %d)\n', ...
    vx_slice, vx_idx, vy_slice, vy_idx);

%% ------------------------ CR3BP f(x) and Jacobian ------------------------
% State ordering: X = [x; vx; y; vy]
f_vec = @(X) cr3bp_f(X, mu);
J_fx  = @(X) cr3bp_J(X, mu);

%% ------------------------- Find L1..L5 (equilibria) ----------------------
[Ls, labelsL] = lagrange_points(mu);  % 5x4 (x,vx,y,vy) with vx=vy=0

% Primaries in rotating frame (for markers)
xP1 = -mu;  xP2 = 1 - mu;
if mu > 1e-4, label1='Earth'; label2='Moon'; else, label1='Sun'; label2='Earth'; end

%% -------- Build the 2D (x,y) mesh from the UNIFORM 4D grid vectors -------
% This slice uses the x_grid and y_grid directly (uniform); vx,vy fixed by indices
[xg, yg] = meshgrid(x_grid, y_grid);
Nx = numel(x_grid); Ny = numel(y_grid);

%% ---------------------- Per-equilibrium evaluation -----------------------
for iEq = 1:5
    x_star = Ls(iEq,:).';                      % 4x1 equilibrium (vx=vy=0)
    A = J_fx(x_star);                          % 4x4 Jacobian at L*

    % Eigen decomposition (right eigenvectors) and left eigenvectors via inv
    [V, D] = eig(A);
    lam = diag(D);
    W = inv(V).';                              % left eigenvectors as columns

    % Normalize so w_j^T v_j = 1 (better conditioned)
    for j = 1:4
        s = W(:,j).' * V(:,j);
        if s ~= 0, W(:,j) = W(:,j) / s; end
    end

    % Classify eigenvalues
    cats = strings(4,1);
    for j = 1:4
        if real(lam(j)) >  tol_neutral, cats(j) = "unstable";
        elseif real(lam(j)) < -tol_neutral, cats(j) = "stable";
        else, cats(j) = "neutral";
        end
    end

    % Pre-allocate Phi maps on the (x,y) slice
    Z = zeros(Ny, Nx, 4);   % Z(:,:,j) corresponds to eigenpair j

    % ---- Evaluate on the uniform (x,y) slice at fixed (vx_slice, vy_slice)
    for ix = 1:Nx
        for iy = 1:Ny
            X0 = [xg(iy,ix); vx_slice; yg(iy,ix); vy_slice];

            % Short forward integration of true nonlinear CR3BP
            [t, Xtraj] = ode45(@(t,xx) f_vec(xx), [0 t_int], X0, optsEv);

            % Nonlinear residual r(x) = f(x) - A(x - x*)
            % Project r(x(t)) onto ALL left eigenvectors at once (reuse)
            gx_all = zeros(numel(t),4);
            for m = 1:numel(t)
                Xm = Xtraj(m,:).';
                r  = f_vec(Xm) - A*(Xm - x_star);
                gx_all(m,:) = (W.' * r).';
            end

            % Compute Phi_j for each eigenpair using the *same* trajectory
            for j = 1:4
                decay = exp(-real(lam(j)) * t);
                base  = W(:,j).' * (X0 - x_star);
                Phi_j = base + trapz(t, decay .* gx_all(:,j));
                switch lower(plot_part)
                    case 'real', Z(iy,ix,j) = real(Phi_j);
                    case 'imag', Z(iy,ix,j) = imag(Phi_j);
                    case 'abs',  Z(iy,ix,j) = abs(Phi_j);
                    otherwise,   Z(iy,ix,j) = real(Phi_j);
                end
            end
        end
    end

    % ------------------------------ Plotting ------------------------------
    figure('Name',sprintf('Uniform-Grid Slice: %s (\\mu=%.4g, vx=%.2f, vy=%.2f)', ...
                          labelsL{iEq}, mu, vx_slice, vy_slice), ...
           'Color','w','Position',[80 80 1200 850]);
    tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    for j = 1:4
        nexttile;

        % Surf (top-down heat) of Phi_j on the uniform (x,y) grid
        surf(xg, yg, Z(:,:,j), 'EdgeAlpha', 0.08);
        view(2); shading interp; axis tight equal;
        colormap(cmap); colorbar;

        % Mark the equilibrium and primaries
        hold on;
        ztag = max(Z(:,:,j),[],'all'); ztag = ztag + 0.05*abs(ztag + (ztag==0));
        plot3(x_star(1), x_star(3), ztag, 'ko', 'MarkerFaceColor','k','MarkerSize',6);
        plot3(xP1, 0, ztag, 'ko', 'MarkerFaceColor','r','MarkerSize',12);
        plot3(xP2, 0, ztag, 'ko', 'MarkerFaceColor','b','MarkerSize',7);

         % Add labels to primaries 
        text(xP1, 0.1, ztag, label1, ...
            'VerticalAlignment','bottom','HorizontalAlignment','center', ...
            'BackgroundColor','w','Margin',1);
        text(xP2, 0.1, ztag, label2, ...
            'VerticalAlignment','bottom','HorizontalAlignment','center', ...
            'BackgroundColor','w','Margin',1);
        if(iEq==1 || iEq==2)
            text(x_star(1), x_star(3)+0.5, ztag, labelsL{iEq}, ...
            'VerticalAlignment','bottom','HorizontalAlignment','center', ...
            'FontWeight','bold','BackgroundColor','w','Margin',1);
        else
            text(x_star(1), x_star(3)+0.1, ztag, labelsL{iEq}, ...
            'VerticalAlignment','bottom','HorizontalAlignment','center', ...
            'FontWeight','bold','BackgroundColor','w','Margin',1);
        end

        title(sprintf('%s, eigen %d: \\lambda=%.4g%+.4gi (%s)', ...
              labelsL{iEq}, j, real(lam(j)), imag(lam(j)), cats(j)), 'Interpreter','tex');
        xlabel('$x$'); ylabel('$y$');
        grid on; box on;
    end

    title(tl, sprintf('Koopman eigenfunction maps on uniform 4D-grid slice at %s', labelsL{iEq}));
end

%% ============================== FUNCTIONS ================================
function f = cr3bp_f(X, mu)
% Planar CR3BP vector field in rotating frame
% X = [x; vx; y; vy]
x  = X(1); vx = X(2); y  = X(3); vy = X(4);
r1 = sqrt((x+mu)^2 + y^2);
r2 = sqrt((x-1+mu)^2 + y^2);
dOdx = x - (1-mu)*(x+mu)/r1^3 - mu*(x-1+mu)/r2^3;
dOdy = y - (1-mu)*y/r1^3     - mu*y/r2^3;
f = [vx; 2*vy + dOdx; vy; -2*vx + dOdy];
end

function J = cr3bp_J(X, mu)
% Jacobian of the planar CR3BP vector field w.r.t X = [x; vx; y; vy]
x  = X(1); vx = X(2); y  = X(3); vy = X(4); %#ok<NASGU>
r1 = sqrt((x+mu)^2 + y^2);
r2 = sqrt((x-1+mu)^2 + y^2);

Ux_x = 1 - (1-mu)*(1/r1^3 - 3*(x+mu)^2/r1^5) - mu*(1/r2^3 - 3*(x-1+mu)^2/r2^5);
Ux_y =      -(1-mu)*(-3*(x+mu)*y/r1^5)       - mu*(-3*(x-1+mu)*y/r2^5);
Uy_x = Ux_y;
Uy_y = 1 - (1-mu)*(1/r1^3 - 3*y^2/r1^5)     - mu*(1/r2^3 - 3*y^2/r2^5);

J = [ 0,   1,   0,   0;
      Ux_x, 0, Ux_y, 2;
      0,   0,   0,   1;
      Uy_x, -2, Uy_y, 0 ];
end

function [Ls, labels] = lagrange_points(mu)
% Compute L1..L5 equilibria for planar CR3BP (vx=vy=0)
gx = @(x) x ...
    - (1-mu)*(x+mu)./abs(x+mu).^3 ...
    - mu*(x-1+mu)./abs(x-1+mu).^3;

optsZ = optimset('TolX',1e-14,'Display','off');
xm1 = -mu; xm2 = 1 - mu; epsx = 1e-6;
xL1 = bracket_then_fzero(gx, [xm1+10*epsx, xm2-10*epsx], optsZ);
xL2 = bracket_then_fzero(gx, [xm2+10*epsx, xm2+5],       optsZ);
xL3 = bracket_then_fzero(gx, [xm1-5,       xm1-10*epsx], optsZ);

gradOmega2D = @(x,y) [ ...
    x - (1-mu)*(x+mu)/((x+mu)^2+y^2)^(3/2) - mu*(x-1+mu)/((x-1+mu)^2+y^2)^(3/2); ...
    y - (1-mu)*y/((x+mu)^2+y^2)^(3/2)      - mu*y/((x-1+mu)^2+y^2)^(3/2) ];
optsF = optimoptions('fsolve','Display','off', ...
    'FunctionTolerance',1e-14,'StepTolerance',1e-14,'OptimalityTolerance',1e-14);
xg = 0.5 - mu; yg = sqrt(3)/2;
xyL4 = fsolve(@(xy) gradOmega2D(xy(1),xy(2)), [xg; yg],  optsF);
xyL5 = fsolve(@(xy) gradOmega2D(xy(1),xy(2)), [xg;-yg], optsF);

L1 = [xL1; 0; 0; 0];
L2 = [xL2; 0; 0; 0];
L3 = [xL3; 0; 0; 0];
L4 = [xyL4(1); 0; xyL4(2); 0];
L5 = [xyL5(1); 0; xyL5(2); 0];

Ls = [L1.'; L2.'; L3.'; L4.'; L5.'];
labels = {'L1','L2','L3','L4','L5'};
end

function xr = bracket_then_fzero(g, interval, opts)
% Robust root finder: seek a sign change sub-interval, then call fzero
a = interval(1); b = interval(2); N = 1000;
xs = linspace(a,b,N+1); gs = arrayfun(g,xs);
valid = isfinite(gs); xs = xs(valid); gs = gs(valid);
idx = find(sign(gs(1:end-1)).*sign(gs(2:end))<0,1,'first');
if ~isempty(idx)
    xr = fzero(g, [xs(idx), xs(idx+1)], opts); return;
end
[~,k] = min(abs(gs)); x0 = xs(k); w = 1e-3*(b-a);
for j = 1:12
    aa = max(a, x0-w); bb = min(b, x0+w);
    ga = g(aa); gb = g(bb);
    if isfinite(ga) && isfinite(gb) && sign(ga) ~= sign(gb)
        xr = fzero(g, [aa, bb], opts); return;
    end
    w = 2*w;
end
error('No sign change detected in [%.3g, %.3g]', a, b);
end

function [value,isterminal,direction]=offFrame(~,Y,DomMax)
value=double(max(abs(Y))>4*DomMax || norm(Y,1)<1e-3);
isterminal=1; direction=0;
end