clear
close all
load('Vorticity_data.mat') % data from https://www.dropbox.com/sh/xj59e5in7dfsobi/AAAfkxqa1x9WFSTgrvqoqqRqa?dl=0
rng(1)

%% Set the parameters
nn = 0.4;   % noise level
r = 3;      % number of POD modes for multDMD
M = 80;     % number of snapshots used
N = M;      % number of basis functions for multDMD

t0 = 0.991314387772095 + 0.131513438841779i; % normalisation of eigenvalues - do not change

%% Add noise
NN =randn(size(VORT));  NN = std(VORT(:)).*NN;
VORT2 = VORT+nn*NN;     % noisy data

%% Plot data
figure
subplot(1,2,1)
u0 = zeros(800*200,1)+NaN;
u0(II) = VORT(:,1);
C = (reshape(u0,[800,200]));
vv=0.02:0.05:1.1;
a=mean(real(C(II)))+std(real(C(II)));
b=mean(real(C(II)))-std(real(C(II)));
C(II) = max(min(C(II),a),b);
c=(a+b)/2;
surf(Xgrid,Ygrid,real(C),'FaceColor', 'interp','EdgeColor', 'interp');
colormap jet
axis equal
view(0,90)
xlim([-2,14])
ylim([-2.3,2.3])
xlabel("$x$",Interpreter="latex")
ylabel("$y$",Interpreter="latex")
title("Noise-free data")
%
subplot(1,2,2)
u0 = zeros(800*200,1)+NaN;
u0(II) = VORT2(:,1);
C = (reshape(u0,[800,200]));
vv=0.02:0.05:1.1;
a=mean(real(C(II)))+std(real(C(II)));
b=mean(real(C(II)))-std(real(C(II)));
C(II) = max(min(C(II),a),b);
c=(a+b)/2;
surf(Xgrid,Ygrid,real(C),'FaceColor', 'interp','EdgeColor', 'interp');
axis equal
view(0,90)
colormap jet
xlim([-2,14])
ylim([-2.3,2.3])
xlabel("$x$",Interpreter="latex")
ylabel("$y$",Interpreter="latex")
title("Noisy data")


%% Perform multDMD after projection onto POD modes
ind = (1:M);
X = VORT2(:,ind);
Y = VORT2(:,ind+1);
[~,S,V_svd]=svd(X','econ');
PX=transpose(X)*V_svd(:,1:r)*diag(1./diag(S(1:r,1:r)));
PY=transpose(Y)*V_svd(:,1:r)*diag(1./diag(S(1:r,1:r)));

[K_MDMD,~,C,~] = construct_basis(PX, PY, N);
[V_MDMD,E_MDMD]=eig(full(K_MDMD),'vector');
I3 = find(abs(E_MDMD)>0.1);
V_MDMD = V_MDMD(:,I3); E_MDMD = E_MDMD(I3); % remove trivial eigenpairs
[~,I]=sort(abs(1-E_MDMD),'ascend');         % reorder modes
V_MDMD = V_MDMD(:,I); E_MDMD = E_MDMD(I);

xi = knnsearch(C,PX);
xj = knnsearch(C, PY);
Ix = sub2ind([M,N],1:M,xi');
Iy = sub2ind([M,N],1:M,xj');

PSI_X = zeros(M,N);     PSI_X(Ix) = 1;
PSI_Y = zeros(M,N);     PSI_Y(Iy) = 1;
multMODE=(PSI_X*V_MDMD)\transpose(X);

%% Perform exactDMD with noise free and noisy data
[~,LAM,Phi] = exactDMD(X,Y,40);
[~,I]=sort(abs(1-LAM),'ascend'); % reorder modes
Phi = Phi(:,I); LAM = LAM(I);

[~,LAM_nf,Phi_nf] = exactDMD(VORT(:,ind),VORT(:,ind+1),40);
[~,I]=sort(abs(1-LAM_nf),'ascend'); % reorder modes
Phi_nf = Phi_nf(:,I); LAM_nf = LAM_nf(I);

%% Perform piDMD with noisy data
[Ux,~,~] = svd(X,0); Ux = Ux(:,1:40);
Yproj = Ux'*Y; Xproj = Ux'*X; % Project X and Y onto principal components
[Uyx, ~, Vyx] = svd(Yproj*Xproj',0);
LAM_pi = sort(eig(Uyx*Vyx'));

%% Plot the eigenvalues
w1 = log(E_MDMD)/abs(log(t0));
[~,I] = sort(imag(w1)); w1 = w1(I);
w2 = log(LAM)/abs(log(t0));
[~,I] = sort(imag(w2)); w2 = w2(I);
w3 = log(LAM_pi)/abs(log(t0));
[~,I] = sort(imag(w3)); w3 = w3(I);
w0 = -20:20;


figure
subplot(3,1,1)
plot(imag(w1),real(w1),'k.','markersize',18)
xlim([-20,20])
ylim([-15,5])
title('MultDMD','interpreter','latex','fontsize',10)
grid minor
subplot(3,1,2)
plot(imag(w2),real(w2),'rx','markersize',7,'linewidth',1)
xlim([-20,20])
ylim([-15,5])
title('DMD','interpreter','latex','fontsize',10)
ylabel('$\log(|\lambda)/|\mathrm{arg}(\lambda_0)|$','interpreter','latex','fontsize',10)
grid minor
subplot(3,1,3)
plot(imag(w3),real(w3),'bx','markersize',7,'linewidth',1)
xlim([-20,20])
ylim([-15,5])
title('piDMD','interpreter','latex','fontsize',10)
xlabel('$\mathrm{arg}(\lambda)/|\mathrm{arg}(\lambda_0)|$','interpreter','latex','fontsize',10)
grid minor

%% Plot the MultDMD modes

figure
u0 = zeros(800*200,1)+NaN; u1 = u0;
plot_i = 1;
for kk = 1:6
    subplot(6,2,plot_i)
    [~,I]=sort(abs(1-LAM_nf),'ascend');
    j = I(2*kk-1);
    u00 = Phi_nf(:,j);
    [~,I]=sort(abs(1-E_MDMD),'ascend');
    j = I(2*kk-1);
    u10 = transpose(multMODE(j,:));
    aa = mean(angle(u00(abs(u00-mean(u00))>std(u00))./u10(abs(u00-mean(u00))>std(u00))));
    u0(II) = u00;
    u1(II) = exp(1i*aa)*u10;
    
    
    u = real(reshape(u0,[800,200]));
    vv=0.02:0.05:1.1;
    a=mean(real(u(II)))+3*std(real(u(II)));
    b=mean(real(u(II)))-3*std(real(u(II)));
    u(II) = max(min(u(II),a),b);
    c=(a+b)/2;
    % Rotate mode
    if kk == 1
        surf(Xgrid,Ygrid,-real(u),'FaceColor', 'interp','EdgeColor', 'interp');
        title("Noise-free (ground truth)")
    else
        surf(Xgrid,Ygrid,real(u),'FaceColor', 'interp','EdgeColor', 'interp');
    end
    axis equal
    view(0,90)
    colormap jet
    xlim([-2,14])
    ylim([-2.3,2.3])
    xlabel("$x$",Interpreter="latex")
    ylabel("$y$",Interpreter="latex",Rotation=0)

    plot_i = plot_i+1;
    %
    subplot(6,2,plot_i)
    u = real(reshape(u1,[800,200]));
    vv=0.02:0.05:1.1;
    a=mean(real(u(II)))+3*std(real(u(II)));
    b=mean(real(u(II)))-3*std(real(u(II)));
    u(II) = max(min(u(II),a),b);
    c=(a+b)/2;
    surf(Xgrid,Ygrid,real(u),'FaceColor', 'interp','EdgeColor', 'interp');
    axis equal
    view(0,90)
    colormap jet
    xlim([-2,14])
    ylim([-2.3,2.3])
    xlabel("$x$",Interpreter="latex")
    ylabel("$y$",Interpreter="latex",Rotation=0)
    if kk==1
        title("MultDMD on noisy data")
    end
    plot_i = plot_i+1;
end

