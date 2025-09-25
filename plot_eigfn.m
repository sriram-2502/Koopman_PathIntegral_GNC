%% ========================================================================
% Standalone: Uniform 4D Grid ([-2,2]^4) + Koopman Eigenfunction Maps
% Planar CR3BP (state: [x; vx; y; vy])
%
% This version:
%   * Plots ONLY eigenvalues with non-zero real part (|Re λ| > tol_neutral)
%   * For complex λ, plots both |Phi| (magnitude) and ∠Phi (phase)
%   * For real λ, plots Re{Phi} (single map)
% ========================================================================

clear; clc; close all;

%% -------------------------- User configuration ---------------------------
% Mass parameter:
mu = 0.0121505856;           % Earth–Moon
% mu = 3.003e-6;            % Sun–Earth

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

% Short integration for variation-of-constants
t_int  = 0.15;                    % horizon
RelTol = 1e-7; AbsTol = 1e-9;     % ODE tolerances

% Classification threshold for "neutral" eigenvalues
tol_neutral = 1e-6;

% Colormap
cmap = parula;

set(groot,'defaultAxesFontSize',12);
set(groot,'defaultTextInterpreter','latex');

% ODE options (+ simple off-frame event)
optsEv=odeset('RelTol',RelTol,'AbsTol',AbsTol,'Events',@(t,y) offFrame(t,y,max(abs(Dom))));

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

%% -------- Build the 2D (x,y) mesh from the UNIFORM 4D grid vectors -------
[xg, yg] = meshgrid(x_grid, y_grid);
Nx = numel(x_grid); Ny = numel(y_grid);

%% ---------------------- Per-equilibrium evaluation -----------------------
eq_list = [1 2 3];               
E = numel(eq_list);

% Storage for the final 12-subplot overview
Zmag_all = cell(E, 2);   % {e, m}  m=1..2 modes
Zphs_all = cell(E, 2);
lam_all  = complex(zeros(E,2));
cats_all = strings(E,2);
xstar_all = zeros(E,4);  % to annotate primaries & eqb in the big overview
labEq_all = cell(E,1);

e_row = 0;

for iEq = 1:5
    x_star = Ls(iEq,:).';                    % 4x1 equilibrium (vx=vy=0)
    A = J_fx(x_star);                        % 4x4 Jacobian at L*

    % Eigen decomposition (right eigenvectors) and left eigenvectors via inv
    [V, D] = eig(A);
    lam = diag(D);
    W = inv(V).';                            % left eigenvectors as columns

    % Normalize so w_j^T v_j = 1
    for j = 1:4
        s = W(:,j).' * V(:,j);
        if s ~= 0, W(:,j) = W(:,j) / s; end
    end

    % Classify (for titles)
    cats = strings(4,1);
    for j = 1:4
        if real(lam(j)) >  tol_neutral, cats(j) = "unstable";
        elseif real(lam(j)) < -tol_neutral, cats(j) = "stable";
        else, cats(j) = "neutral";
        end
    end

    % Keep only eigenvalues with nonzero real part; take top-2 by |Re|
    keep_idx = find(abs(real(lam)) > tol_neutral);
    if isempty(keep_idx)
        fprintf('Skipping %s: no modes with |Re λ| > %.1e\n', labelsL{iEq}, tol_neutral);
        continue;
    end
    [~, order] = sort(abs(real(lam(keep_idx))), 'descend');
    keep_idx = keep_idx(order);                   % sorted by |Re| desc
    keep_idx = keep_idx(1:min(2, numel(keep_idx)));  % top 1 or 2

    % Pre-allocate per-kept-mode grids (magnitude & phase)
    K = numel(keep_idx);                          % 1 or 2
    Zmag  = cell(1,K);
    Zphs  = cell(1,K);
    for k = 1:K
        Zmag{k} = zeros(Ny, Nx);
        Zphs{k} = zeros(Ny, Nx);
    end

    % ---- Evaluate on the uniform (x,y) slice at fixed (vx_slice, vy_slice)
    for ix = 1:Nx
        for iy = 1:Ny
            X0 = [xg(iy,ix); vx_slice; yg(iy,ix); vy_slice];

            % Forward integration (for unstable eigenvalues)
            [tf, Xf] = ode45(@(t,xx) f_vec(xx), [0, t_int], X0, optsEv);
            % Backward integration (for stable eigenvalues): integrate backward flow
            [tb, Xb] = ode45(@(t,xx) f_vec(xx), [0, -t_int], X0, optsEv);

            % Project residuals along all left eigenvectors at each step
            gF = zeros(numel(tf), 4);
            for m = 1:numel(tf)
                Xm = Xf(m,:).';
                r  = f_vec(Xm) - A*(Xm - x_star);
                gF(m,:) = (W.' * r).';
            end
            gB = zeros(numel(tb), 4);
            for m = 1:numel(tb)
                Xm = Xb(m,:).';
                r  = f_vec(Xm) - A*(Xm - x_star);
                gB(m,:) = (W.' * r).';
            end

            % Compute Phi_j for each kept eigenpair and store |.|, angle
            for k = 1:K
                j    = keep_idx(k);
                lamj = lam(j);

                if real(lamj) > 0
                    decay = exp(-real(lamj) * tf);
                    base  = W(:,j).' * (X0 - x_star);
                    Phi_j = base + trapz(tf, decay .* gF(:,j));
                else % real(lamj) < 0
                    tpos  = abs(tb); % positive times for decay
                    decay = exp( real(lamj) * tpos);
                    base  = W(:,j).' * (X0 - x_star);
                    Phi_j = base + trapz(tpos, decay .* gB(:,j));
                end

                Zmag{k}(iy,ix) = abs(Phi_j);
                Zphs{k}(iy,ix) = angle(Phi_j);
            end
        end
    end

    %% ------------------------------ Per-eqb figure ------------------------------
    % 2×2 layout: [|Phi_1|, angle Phi_1 ; |Phi_2|, angle Phi_2] (if K=1, top row only)
    fig = figure('Name',sprintf('Slice @ %s (\\mu=%.4g, vx=%.2f, vy=%.2f)', ...
                          labelsL{iEq}, mu, vx_slice, vy_slice), ...
           'Color','w','Position',[80 80 1200 900]);
    tl = tiledlayout(2, 2, 'TileSpacing','compact','Padding','compact');

    for k = 1:K
        lamj = lam(keep_idx(k));
        % Magnitude
        nexttile;
        surf(xg, yg, Zmag{k}, 'EdgeAlpha', 0.06);
        view(2); shading interp; axis equal tight;
        colormap(cmap); colorbar; hold on;
        ztag = max(Zmag{k},[],'all'); ztag = ztag + 0.05*max(1,abs(ztag));
        mark_scene(ztag, x_star, xP1, xP2, label1, label2, labelsL{iEq}, iEq);
        title(sprintf('%s — eigen %d: |\\Phi|, \\lambda=%.4g%+.4gi (%s)', ...
              labelsL{iEq}, keep_idx(k), real(lamj), imag(lamj), cats(keep_idx(k))), 'Interpreter','tex');
        xlabel('$x$'); ylabel('$y$'); grid on; box on;

        % Phase
        nexttile;
        surf(xg, yg, Zphs{k}, 'EdgeAlpha', 0.06);
        view(2); shading interp; axis equal tight;
        colormap(cmap); colorbar; hold on;
        ztag = max(Zphs{k},[],'all'); ztag = ztag + 0.05*max(1,abs(ztag));
        mark_scene(ztag, x_star, xP1, xP2, label1, label2, labelsL{iEq}, iEq);
        title(sprintf('%s — eigen %d: \\angle\\Phi, \\lambda=%.4g%+.4gi (%s)', ...
              labelsL{iEq}, keep_idx(k), real(lamj), imag(lamj), cats(keep_idx(k))), 'Interpreter','tex');
        xlabel('$x$'); ylabel('$y$'); grid on; box on;
    end
    title(tl, sprintf('Koopman eigenfunction maps (|Re\\,\\lambda|>%.1e) — %s', ...
        tol_neutral, labelsL{iEq}));

    %% --------- Save for overview if this eqb is in eq_list ----------
    if ismember(iEq, eq_list)
        e_row = e_row + 1;
        xstar_all(e_row,:) = x_star(:).';
        labEq_all{e_row}   = labelsL{iEq};
        % Copy up to 2 modes into the overview storage
        for k = 1:min(2,K)
            Zmag_all{e_row,k} = Zmag{k};
            Zphs_all{e_row,k} = Zphs{k};
            lam_all(e_row,k)  = lam(keep_idx(k));
            cats_all(e_row,k) = cats(keep_idx(k));
        end
    end
end

%% -------------------------- BIG overview figure --------------------------
if e_row == 0
    warning('No equilibria had nonzero-real modes to show in overview.');
else
    % We’ll make a 3x4 (or e_row x 4) figure: row = eqb, columns = [|Phi1|, angle1, |Phi2|, angle2]
    nrows = e_row; ncols = 4;
    figAll = figure('Name', sprintf('Overview: %d eqb × 4 tiles (\\mu=%.4g, vx=%.2f, vy=%.2f)', ...
                            nrows, mu, vx_slice, vy_slice), ...
                    'Color','w','Position',[50 50 1500 1000]);
    tlAll = tiledlayout(nrows, ncols, 'TileSpacing','compact','Padding','compact');

    for e = 1:nrows
        x_star = xstar_all(e,:).';
        for m = 1:2
            % Column mapping: (m=1) -> cols 1-2 ; (m=2) -> cols 3-4
            col_mag  = (m-1)*2 + 1;
            col_phs  = (m-1)*2 + 2;

            % Skip if this mode missing
            if isempty(Zmag_all{e,m}), continue; end

            lamj = lam_all(e,m);

            % Magnitude tile
            nexttile(sub2ind([ncols nrows], col_mag, e));
            surf(xg, yg, Zmag_all{e,m}, 'EdgeAlpha', 0.06);
            view(2); shading interp; axis equal tight;
            colormap(cmap); colorbar; hold on;
            ztag = max(Zmag_all{e,m},[],'all'); ztag = ztag + 0.05*max(1,abs(ztag));
            mark_scene(ztag, x_star, xP1, xP2, label1, label2, labEq_all{e}, iEq);
            title(sprintf('%s — |\\Phi_%d|, \\lambda=%.4g%+.4gi (%s)', ...
                  labEq_all{e}, m, real(lamj), imag(lamj), cats_all(e,m)), 'Interpreter','tex');
            xlabel('$x$'); ylabel('$y$'); grid on; box on;

            % Phase tile
            nexttile(sub2ind([ncols nrows], col_phs, e));
            surf(xg, yg, Zphs_all{e,m}, 'EdgeAlpha', 0.06);
            view(2); shading interp; axis equal tight;
            colormap(cmap); colorbar; hold on;
            ztag = max(Zphs_all{e,m},[],'all'); ztag = ztag + 0.05*max(1,abs(ztag));
            mark_scene(ztag, x_star, xP1, xP2, label1, label2, labEq_all{e}, iEq);
            title(sprintf('%s — \\angle\\Phi_%d, \\lambda=%.4g%+.4gi (%s)', ...
                  labEq_all{e}, m, real(lamj), imag(lamj), cats_all(e,m)), 'Interpreter','tex');
            xlabel('$x$'); ylabel('$y$'); grid on; box on;
        end
    end

    title(tlAll, sprintf('All equilibria overview (|Re\\,\\lambda|>%.1e): rows=L1/L2/L3, cols=[|\\Phi_1|, \\angle\\Phi_1, |\\Phi_2|, \\angle\\Phi_2]', ...
        tol_neutral));
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
x  = X(1); y  = X(3);
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

function mark_scene(ztag, x_star, xP1, xP2, label1, label2, labEq, iEq)
% Helper to annotate primaries and the equilibrium consistently
plot3(x_star(1), x_star(3), ztag, 'ko', 'MarkerFaceColor','k','MarkerSize',6);
plot3(xP1, 0, ztag, 'ko', 'MarkerFaceColor','b','MarkerSize',12);
plot3(xP2, 0, ztag, 'ko', 'MarkerFaceColor','white','MarkerSize',7);
text(xP1, 0.1, ztag, label1, 'VerticalAlignment','bottom','HorizontalAlignment','center', ...
    'BackgroundColor','w','Margin',1);
text(xP2, 0.1, ztag, label2, 'VerticalAlignment','bottom','HorizontalAlignment','center', ...
    'BackgroundColor','w','Margin',1);
% dy = iEq<=2; dy = dy*0.5 + (~dy)*0.1;
% text(x_star(1), x_star(3)+dy, ztag, labEq, 'VerticalAlignment','bottom', ...
%     'HorizontalAlignment','center','FontWeight','bold','BackgroundColor','w','Margin',1);
end
