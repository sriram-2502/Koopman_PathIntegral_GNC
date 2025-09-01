function sys_info = duffing_info()

% system Setup
% Define parameters
n = 2; m = 1;
x = sym('x',[n,1],'real');

% Define the system dynamics symbolically
% lambda1 = 1;
% lambda2 = -0.1;
% delta = (lambda1+lambda2);
% alpha = -lambda1*lambda2;
alpha = 1;
delta = 0.5;
f_x = [x(2);
       alpha*x(1)-delta*x(2)-x(1)^3];
g_x = [1; 0];

% setup matlab functions
f = matlabFunction(f_x,'vars',{x}); 
g = matlabFunction(g_x,'vars',x); 

% Compute the Jacobian matrix
J = jacobian(f_x, x);

% Evaluate the Jacobian at the equilibrium point
x_eqb = [0; 0];
A = double(subs(J, x, x_eqb));
B = double(subs(g_x, x, x_eqb));

% Compute eigenvalues and eigenvectors
[~,D, W] = eig(A); % left Eigenvectors (W) and eigenvalues (D)

%% setup system parameters for xdot = f(x) + B*u
sys_info.A              = A;
sys_info.B              = B;
sys_info.A_koopman      = D;
sys_info.dynamics_f     = f;
sys_info.dynamics_g     = g;
sys_info.state_dim      = n;
sys_info.ctrl_dim       = m;
sys_info.eig_vectors    = W;
sys_info.eig_vals       = D;
sys_info.x_eqb          = x_eqb;
sys_info.A_unstable     = A;
sys_info.A_stable       = A;
sys_info.id             = "duffing";
sys_info.eigen_fun      = [];

%% phase portrait
plot_phase_portrait = true;
if(plot_phase_portrait)
    Dom = [-2 2];
    [X, Y] = meshgrid(Dom(1):0.1:Dom(2), Dom(1):0.1:Dom(2));
    
    % Initialize components of the vector field
    u = zeros(size(X));
    v = zeros(size(Y));
    
    % Evaluate f at each grid point
    for i = 1:numel(X)
        xy = [X(i); Y(i)];   % 2x1 input
        result = f(xy);      % 2x1 output
        u(i) = result(1);    % x-component
        v(i) = result(2);    % y-component
    end
    
    figure(1)
    l = streamslice(X,Y,u,v,2); hold on;
    set(l,'LineWidth',1)
    set(l,'Color','k');

    xlim(Dom)
    ylim(Dom)
    axes = gca;
    axis square
    set(axes,'FontSize',15);
    xlabel('$x_1$','FontSize',20, 'Interpreter','latex')
    ylabel('$x_2$','FontSize',20, 'Interpreter','latex')
    box on
    axes.LineWidth=2;
end
end