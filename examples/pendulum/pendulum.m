%% Nonlinear pendulum example

% Set random seed
rng(1)

% Number of initial points in x and y directions
M1 = 20;

% Time discretization
Nt = 101;

% Create grid of initial conditions
x1 = linspace(-0.6,0.6,M1);
x2 = linspace(-0.6,0.6,M1);
[X1,Y1] = ndgrid(x1,x2);
X1 = X1(:);
Y1 = Y1(:);

% Create grid of time steps
tspan = linspace(0,10,Nt);

% Damping parameter
lmbda = 0.0;

% Define pendulum equation
w = 3;
dtheta = @(t,theta) [theta(2);-sin(w*theta(1)) - lmbda*theta(2)];

X = [];
Y = [];
for i = 1:length(X1)
    sprintf("i = %d / %d", i, length(X1))
    theta_ic = [X1(i), Y1(i)];
    [t, theta] = ode45(dtheta, tspan, theta_ic);
    X = [X; theta(1:end-1,:)];
    Y = [Y; theta(2:end,:)];
end

%% Construct Koopman matrix
N = 1000;
[B,W,C,Sumd] = construct_basis(X, Y, N);
[E,D] = eig(full(B));
D = diag(D);
theta_eig = atan2(imag(D),real(D));
[~, I_theta] = sort(theta_eig);

% Reorder D and E according to I_theta
D = D(I_theta);
E = E(:, I_theta);
theta_eig = theta_eig(I_theta);

% Compute Gram matrix
G = diag(sum(W,2));

% Compute eigenvalue
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

% Take mode with 50 largest number of basis functions
i_mode = find(abs(D)>0);
E_sum = sum(abs(real(E(:,i_mode)))>0,1);
i_vec = find(sum(abs(real(E(:,i_mode)))>0,1)>50);

% Sort them from smallest theta to largest
E = E(:,i_mode(i_vec));
theta_eig = theta_eig(i_mode(i_vec));
D = D(i_mode(i_vec));

%% Plot trajectories and eigenvalues

figure
subplot(1,2,1)
x1 = linspace(0,0.6,11);
X1 = x1';
Y1 = X1;
tspan = linspace(0,6,Nt);
X = [];
Y = [];
for i = 1:length(X1)
    theta_ic = [X1(i), Y1(i)];
    [t, theta] = ode45(dtheta, tspan, theta_ic);
    X = [X; theta(1:end-1,:)];
    Y = [Y; theta(2:end,:)];
end
for i = 1:length(X1)
    plot(X(1+(Nt-1)*(i-1):i*(Nt-1),1), X(1+(Nt-1)*(i-1):i*(Nt-1),2),'b');
    hold on
end
hold off
xlim([-1,1])
ylim([-1.5,1.5])
title("Trajectories")
axis square

subplot(1,2,2)
plot(D,'.')
title("MultDMD")
xlim([-1.2,1.2])
ylim([-1.2,1.2])
axis square

%%

figure
i_plot = 1;
for eig_i = [106,109,110,117,120,125]
    sprintf("i = %d / 6", i_plot)
    subplot(2,3,i_plot)
    ei = E(:,eig_i);
    colorArray = real(ei);
    bs_ext = [-1 1 1 -1;-1.5 -1.5 1.5 1.5]';
    [v,c,XY] = VoronoiLimit(C(:,1),C(:,2),'bs_ext',bs_ext,'figure','off');
    % Find permutation from C to XY
    [B_XY, I_sort_XY] = sortrows(C);
    colorArray = colorArray(I_sort_XY);
    
    [s,ds] = cellfun(@size,c);
    vertex_x = {};
    vertex_y = {};
    color_cell = {};
    vertex_x{max(s)} = [];
    vertex_y{max(s)} = [];
    color_cell{max(s)} = [];
    for i = 1:length(c)
        vertex_x{length(c{i})} = [vertex_x{length(c{i})}, v(c{i},1)];
        vertex_y{length(c{i})} = [vertex_y{length(c{i})}, v(c{i},2)];
        color_cell{length(c{i})} = [color_cell{length(c{i})}; colorArray(i)];
    end
    
    % Plot voronoi diagram
    for i = 1:max(s)
        if ~isempty(vertex_x{i})
            patch(vertex_x{i}, vertex_y{i},color_cell{i}');
        end
    end
    colormap turbo
    xlim([-1,1])
    ylim([-1.5,1.5])
    title(sprintf("$\\lambda=\\exp(%.2f i)$",theta_eig(eig_i)),Interpreter="latex")
    axis square
    i_plot = i_plot + 1;
end

%% Compute residual
figure
A_matrix = W;
G_matrix = diag(sum(W,2));
L_matrix = diag(sum(W,1));
V = E;
D1 = diag(D);

RES1 = sqrt(real(dot(V,L_matrix*V) - dot(V,A_matrix'*V*D1+A_matrix*V*D1') + dot(V, G_matrix*V*abs(D1).^2)) ./ real(dot(V,G_matrix*V)));

[~,I3] = sort(RES1,'descend');
D1 = D;
D1 = D1(I3);
RES1 = RES1(I3);

scatter(real(full(D1)),imag(full(D1)),15,RES1,'filled')
xlim([-1.2,1.2])
ylim([-1.2,1.2])
axis square
colormap jet
colorbar
clim([0,1])