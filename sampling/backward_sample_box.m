function [x_sample, t_sample] = backward_sample_box(x_star, f, varargin)
% BACKWARD_SAMPLE_BOX
% Generate a random 4D box of samples around the equilibrium x_star,
% integrate each sample backward in time (dx/dt = -f(x)), and return:
%   x_0n : selected_data_points that remain inside the domain (N x 4)
%   t_0n : corresponding times from backward runs (N x 1)
%
% Usage:
%   [x_0n, t_0n] = backward_sample_box(x_star, f)
%   [x_0n, t_0n] = backward_sample_box(x_star, f, 'NumPoints', 800, 'HalfSize', 0.1)
%
% Inputs
%   x_star : 1x4 (or 4x1) equilibrium [x1* 0 x3* 0]
%   f      : function handle @(x) -> 4x1 vector field (no time arg)
%
% Name-Value (optional)
%   'NumPoints' : number of random samples (default 500)
%   'HalfSize'  : half-size of the 4D box per coord (scalar or 1x4) (default 0.1)
%   'Dom'       : domain vector [-B B] for selection (default [-5 5])
%   'Dt'        : time step for tspan (default 0.01)
%   'Tmax'      : backward horizon (default 5)
%   'RelTol'    : ode45 RelTol (default 1e-9)
%   'AbsTol'    : ode45 AbsTol (default 1e-300)

% ---------- parse inputs ----------
p = inputParser;
p.addParameter('NumPoints', 500,    @(z) isnumeric(z) && isscalar(z) && z>0);
p.addParameter('HalfSize',  0.1,    @(z) isnumeric(z) && (isscalar(z) || numel(z)==4));
p.addParameter('Dom',       [-5 5], @(z) isnumeric(z) && numel(z)==2);
p.addParameter('Dt',        0.01,   @(z) isnumeric(z) && isscalar(z) && z>0);
p.addParameter('Tmax',      5,      @(z) isnumeric(z) && isscalar(z) && z>0);
p.addParameter('RelTol',    1e-6,   @(z) isnumeric(z) && isscalar(z));
p.addParameter('AbsTol',    1e-9, @(z) isnumeric(z) && isscalar(z));
p.parse(varargin{:});

NumPoints = p.Results.NumPoints;
HalfSize  = p.Results.HalfSize;
Dom       = p.Results.Dom;
Dt        = p.Results.Dt;
Tmax      = p.Results.Tmax;
RelTol    = p.Results.RelTol;
AbsTol    = p.Results.AbsTol;

% ---------- normalize inputs ----------
x_star = x_star(:).';
if isscalar(HalfSize), HalfSize = repmat(HalfSize,1,4); end
B = Dom(2);  % use upper bound as in your snippet

% ================== Random 4D box around equilibrium ====================
% Centered at [x1* 0 x3* 0] with +/- HalfSize per coordinate
x_min = [x_star(1)-HalfSize(1), 0 -HalfSize(2), x_star(3)-HalfSize(3), 0 -HalfSize(4)];
x_max = [x_star(1)+HalfSize(1), 0 +HalfSize(2), x_star(3)+HalfSize(3), 0 +HalfSize(4)];

% Draw uniform random samples in the 4D rectangle
x_coords = x_min(1) + (x_max(1) - x_min(1)) * rand(NumPoints, 1);
y_coords = x_min(2) + (x_max(2) - x_min(2)) * rand(NumPoints, 1);
z_coords = x_min(3) + (x_max(3) - x_min(3)) * rand(NumPoints, 1);
w_coords = x_min(4) + (x_max(4) - x_min(4)) * rand(NumPoints, 1);

% Combine into matrix: rows = points, cols = [x1 x2 x3 x4]
data_points = [x_coords, y_coords, z_coords, w_coords];

% ================== Backward-time simulation settings ===================
% Stricter tolerances and event stop (same spirit as your offFrame)
eventFcn = @(t, X) offFrame_local(t, X, B);
ode_opts = odeset('RelTol', RelTol, 'AbsTol', AbsTol, 'Events', eventFcn);
tspan    = 0:Dt:Tmax;

% ================== Integrate backward from all samples =================
x_all = [];
t_all = [];
for i = 1:NumPoints
    % Integrate dx/dt = -f(x) backward over tspan
    [t_i, x_i] = ode45(@(t,X)(-f(X)), tspan, data_points(i,:), ode_opts);
    % Accumulate states and times across all trajectories
    x_all = [x_all; x_i];
    t_all = [t_all; t_i];
end

% =============== Keep only states within hyper-rectangle |xi|<B =========
selected_indices = abs(x_all(:,1)) < B & abs(x_all(:,2)) < B & ...
                   abs(x_all(:,3)) < B & abs(x_all(:,4)) < B;

% Outputs with the SAME names/roles as your snippet
x_sample = x_all(selected_indices, :);   % seeds for forward path integrals
t_sample = t_all(selected_indices, 1);   % corresponding times from backward run

end % function

% ----------------------- local event function ---------------------------
function [value, isterminal, direction] = offFrame_local(~, Y, B)
% Trigger event when leaving a large box or getting very close to origin.
value      = (max(abs(Y)) > 4*B) | (sum(abs(Y)) < 1e-3);
isterminal = 1;   % stop the integration
direction  = 0;   % detect all zero-crossings
end
