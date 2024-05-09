%% Lorenz system example

rng(1);

% Time discretization
Nt = 10^6;
dt = 0.01;

% Create grid of time steps
tspan = linspace(0,Nt*dt,Nt);

% Define lorenz equations
rho = 28;
sigma = 10;
beta = 8/3;
dtheta = @(t,x) [sigma*(x(2)-x(1));x(1)*(rho-x(3))-x(2);x(1)*x(2)-beta*x(3)];

% initial condition:
theta_ic = [1,1,1];

% Solve equation
[~, theta] = ode45(dtheta, tspan, theta_ic);

% Remove first time steps
N_burn = 10^4;
X = theta(N_burn:end-1,:);
Y = theta(N_burn+1:end,:);

%% Construct Koopman matrix
% statistics
N = 5000;
X_kmeans = X(1:10:end,:);
[B,W,C,Sumd] = construct_basis(X, Y, N, X_kmeans);
% Compute Gram matrix
G = diag(sum(W,2));

% Compute eigenvalue
sprintf("Solving eigenvalue problem")
[E,D] = eig(full(B));

% Normalize E with respect to the Gram matrix
nn = real(dot(E,G*E));
E = (1./sqrt(nn)).*E;

D = diag(D);
theta_eig = atan2(imag(D),real(D));
[~, I_theta] = sort(theta_eig);
% Reorder D and E according to I_theta
D = D(I_theta);
E = E(:, I_theta);
theta_eig = theta_eig(I_theta);

%% Plot trajectories

figure

% Time discretization
Nmax = 5000;

% Solve equation
tspan = linspace(0,100,100/0.01);
[~, theta] = ode45(dtheta, tspan, [1,1,1]);

subplot(1,2,1)
plot(theta(:,1),theta(:,3))
hold on
plot(C(:,1),C(:,3),'r.','MarkerSize',3)
hold off
xlabel("$x$",'Interpreter','latex')
ylabel("$z$",'Interpreter','latex')
axis square
title("Trajectories and centroids")

subplot(1,2,2)
D1 = D(abs(D)>0);
plot(real(D1),imag(D1),'.')
axis square
xlim([-1.2,1.2])
ylim([-1.2,1.2])
title("Eigenvalues")

%% Plot eigenmodes

i_mode = find(abs(D)>0);
E_sum = sum(abs(real(E(:,i_mode)))>0,1);
% Take mode with 50 largest number of basis functions
i_vec = find(sum(abs(real(E(:,i_mode)))>0,1)>50); % 50

% Sort them from smallest theta to largest
E2 = E(:,i_mode(i_vec));
theta_eig2 = theta_eig(i_mode(i_vec));
% Select only positive theta
idx = theta_eig2 >= 0;
theta_eig2 = theta_eig2(idx);
E2 = E2(:,idx);

% Color trajectories according to the x points
rng(1);

% Time discretization
Nt = 2*10^4;
dt = 0.01;

% Create grid of time steps
tspan = linspace(0,Nt*dt,Nt);

% Define lorenz equations
rho = 28;
sigma = 10;
beta = 8/3;
dtheta = @(t,x) [sigma*(x(2)-x(1));x(1)*(rho-x(3))-x(2);x(1)*x(2)-beta*x(3)];

% initial condition:
theta_ic = [1,1,1];

% Solve equation
[t, theta] = ode45(dtheta, tspan, theta_ic);

N_burn = 10^3;
X = theta(N_burn:end,:);
xi = knnsearch(C,X);

% Plot eigenvector
figure
plot_i = 1;
for eig_i = [1,3,7,8,9,10,11,12,13]
    sprintf("i = %d /  9", plot_i)
    subplot(3,3,plot_i)
    ei = E2(:,eig_i);
    vi = ei(xi);
    scatter3(X(:,1),X(:,2),X(:,3), 10, real(vi), 'filled')
    clim([-max(abs(ei)),max(abs(ei))])
    ylim([-25,25])
    xlim([-25,25])
    zlim([0,50])
    xlabel("$x$",'Interpreter','latex')
    ylabel("$y$",'Interpreter','latex')
    zlabel("$z$",'Interpreter','latex','Rotation',0)
    colormap jet
    view(43.8750, 18)
    axis square
    colorArray = real(ei);
    title(sprintf("$\\lambda=\\exp(%.2f i)$",theta_eig2(eig_i)),Interpreter="latex")
    plot_i = plot_i+1;
end