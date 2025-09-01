function eigen_fn = compute_eigen_fn(x_local, x_eqb, dynamics, D, W, sys_info)
% parse inputs
n_dim = length(x_eqb);

% get modified linear system
A = sys_info.A;


% parse params for path integral setup
if(strcmp(sys_info.id,'duffing'))
    path_integral_params = duffing_PI_params;
else
    disp('!!! Missing param file for path integral setup')
end

% scale eigen vectors
eig_vector_scale = path_integral_params.eigen_vector_scale;
W = W./eig_vector_scale;

% check for reverse time
use_reverse = path_integral_params.unstable_reverse;
if(use_reverse)
    eig_vals = -diag(D);
else
    eig_vals = diag(D);
end

%% open loop simualtion
t_start = 0;
dt_sim  = path_integral_params.dt_sim;
t_end   = path_integral_params.t_end;
Xout    = x_local';
x_op    = x_local;
Tout    = 0;

for t_sim = t_start:dt_sim:t_end
    % forward simulate using rk4 with no control
    x_next_full = euler(dynamics,dt_sim,x_op,0,use_reverse,sys_info);

    % get nonlinear part only
    x_next = (x_next_full-x_op) / dt_sim - A*x_op;

    % update
    x_op = x_next_full;

    % logs
    Tout  = [Tout;t_sim];
    Xout  = [Xout;x_next'];

end

% shift eqb point
Xout   = Xout - x_eqb';

%% compute nonlinear part of eigfun
integrand_convergence = cell(n_dim);
solution_convergence  = cell(n_dim);
for i = 1:n_dim
    % get eigval and eigvec
    lambda  = eig_vals(i);
    w       = W(:,i);

    % compute path integral
    integrand = exp(-Tout*lambda).*(w'*Xout')';
    phi_nonlinear{i} = trapz(Tout,integrand,1);
    phi_linear{i} = w'*x_local;
    phi{i} = phi_linear{i}  + phi_nonlinear{i};

    % check for convergence
    sol_conv = (w'*Xout')';
    integrand_convergence{i} = integrand(end);
    solution_convergence{i}  = sol_conv(end);
end

% Loop through each element in phi and assign it to phi_forward.phi
for i = 1:n_dim
    eigen_fn.phi(i)           = phi{i};
    eigen_fn.phi_linear(i)    = phi_linear{i};
    eigen_fn.phi_nonlinear(i) = phi_nonlinear{i};
    eigen_fn.integrand(i)     = integrand_convergence{i};
    eigen_fn.sol_conv(i)      = solution_convergence{i};
end