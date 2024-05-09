% Lid-driven cavity example
% Data and external codes (all codes in this folder except this script) from https://ucsb.app.box.com/s/tbtdij1r4z5qt7o4cbgmolx28do98xci
clear
close all
load('Cavity16k.mat') 
[Grid,Operators] = CavityGridOperators(VelocityField.N);
VORT = - Operators.del2*real(VelocityField.Psi);
Time = VelocityField.Time - VelocityField.Time(1);
rng(1)

%% Set the parameters
nn = 0.4;   % noise level
r = 5;      % number of POD modes for multDMD
M = 1000;   % number of snapshots used
N = M;      % number of basis functions for multDMD
N2 = min(300,M);

t0 = exp(1i*2*pi*(Time(2)-Time(1))*0.096911); % base eigenvalue
t1 = exp(1i*2*pi*(Time(2)-Time(1))*0.155375); % other base eigenvalue

TH1 = 2*pi*(Time(2)-Time(1))*0.096911*(-5:5); TH1 = exp(1i*TH1);
TH2 = 2*pi*(Time(2)-Time(1))*0.155375*(-5:5); TH2 = exp(1i*TH2);
TH = kron(TH1,TH2); TH = log(TH)/abs(log(t0)); clear TH1 TH2

%% Add noise
NN =randn(size(VORT));      NN = std(VORT(:)).*NN;
VORT2 = VORT+nn*NN;     % noisy data

%% Plot data without and with noise
figure
[xx2,yy2] = meshgrid(Grid.x,Grid.x);

subplot(1,2,1)
u = VORT(:,1)-mean(VORT(:,1));
u = reshape(real(u),VelocityField.N+1,VelocityField.N+1);
contourf(xx2,yy2,u,150,'edgecolor','none')     % vorticty
axis square equal; 
box on;
colormap jet
clim([-std(u(:))+mean(u(:)),std(u(:))+mean(u(:))])
xlabel("$x$",Interpreter="latex")
ylabel("$y$",Interpreter="latex")
title("Noise-free data")

%
u = VORT2(:,1)-mean(VORT(:,1));
u = reshape(real(u),VelocityField.N+1,VelocityField.N+1);
subplot(1,2,2)
contourf(xx2,yy2,u,150,'edgecolor','none')     % vorticty
axis square equal; 
box on;
colormap jet
clim([-std(u(:))+mean(u(:)),std(u(:))+mean(u(:))])
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
[~,LAM,Phi] = exactDMD(X,Y,N2);
[~,I]=sort(abs(1-LAM),'ascend'); % reorder modes
Phi = Phi(:,I); LAM = LAM(I);

[~,LAM_nf,Phi_nf] = exactDMD(VORT(:,ind),VORT(:,ind+1),N2);
[~,I]=sort(abs(1-LAM_nf),'ascend'); % reorder modes
Phi_nf = Phi_nf(:,I); LAM_nf = LAM_nf(I);

%% Perform piDMD with noisy data
[Ux,~,~] = svd(X,0); Ux = Ux(:,1:N2);
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

figure
subplot(3,1,1)
plot(imag(w1),real(w1),'k.','markersize',18)
hold on
plot(imag(TH),real(TH)*0,'g.','markersize',8)
xlim([-1,1]*6)
ylim([-60,5])
title('MultDMD','interpreter','latex','fontsize',10)
grid minor

subplot(3,1,2)
plot(imag(w2),real(w2),'rx','markersize',7,'linewidth',1)
hold on
plot(imag(TH),real(TH)*0,'g.','markersize',8)
xlim([-1,1]*6)
ylim([-60,5])
title('DMD','interpreter','latex','fontsize',10)
ylabel('$\log(|\lambda)/|\mathrm{arg}(\lambda_1)|$','interpreter','latex','fontsize',10)
grid minor


subplot(3,1,3)
plot(imag(w3),real(w3),'bx','markersize',7,'linewidth',1)
hold on
plot(imag(TH),real(TH)*0,'g.','markersize',8)
xlim([-1,1]*6)
ylim([-60,5])
title('piDMD','interpreter','latex','fontsize',10)
xlabel('$\mathrm{arg}(\lambda)/|\mathrm{arg}(\lambda_1)|$','interpreter','latex','fontsize',10)
grid minor

%% Plot the MultDMD modes

figure
[~,I]=sort(abs(1-LAM_nf),'ascend');
j = I(1);
u0 = Phi_nf(:,j);
[~,I]=sort(abs(1-E_MDMD),'ascend');
j = I(1);

u1 = multMODE(j,:); u1 = u1(:);
aa = mean(angle(u0(abs(u0)>std(u0))./u1(abs(u0)>std(u0))));
u1 = exp(1i*aa)*u1;

u = reshape(real(u0),VelocityField.N+1,VelocityField.N+1);
subplot(3,2,1)
contourf(xx2,yy2,u,150,'edgecolor','none')     % vorticty
axis square equal; 
box on;
colormap jet
clim([-3*std(u(:))+mean(u(:)),3*std(u(:))+mean(u(:))])
xlabel("$x$",Interpreter="latex")
ylabel("$y$",Interpreter="latex")
title("Noise-free (ground truth)")

subplot(3,2,2)
u = reshape(real(u1),VelocityField.N+1,VelocityField.N+1);
contourf(xx2,yy2,-u,150,'edgecolor','none')     % vorticty
axis square equal; 
box on;
colormap jet
clim([-3*std(u(:))+mean(u(:)),3*std(u(:))+mean(u(:))])
xlabel("$x$",Interpreter="latex")
ylabel("$y$",Interpreter="latex")
title("MultDMD on noisy data")

[~,I]=sort(abs(t0-LAM_nf),'ascend');
j = I(1);
u0 = Phi_nf(:,j);
[~,I]=sort(abs(t0-E_MDMD),'ascend');
j = I(1);

u1 = multMODE(j,:); u1 = u1(:);
aa = mean(angle(u0(abs(u0)>std(u0))./u1(abs(u0)>std(u0))));
u1 = exp(1i*aa)*u1;

subplot(3,2,3)
u = reshape(real(u0),VelocityField.N+1,VelocityField.N+1);
contourf(xx2,yy2,u,150,'edgecolor','none')     % vorticty
axis square equal; 
box on;
colormap jet
clim([-3*std(u(:))+mean(u(:)),3*std(u(:))+mean(u(:))])
xlabel("$x$",Interpreter="latex")
ylabel("$y$",Interpreter="latex")

subplot(3,2,4)
u = reshape(real(u1),VelocityField.N+1,VelocityField.N+1);
contourf(xx2,yy2,u,150,'edgecolor','none')     % vorticty
axis square equal; 
box on;
colormap jet
clim([-3*std(u(:))+mean(u(:)),3*std(u(:))+mean(u(:))])
xlabel("$x$",Interpreter="latex")
ylabel("$y$",Interpreter="latex")

[~,I]=sort(abs(t1-LAM_nf),'ascend');
j = I(1);
u0 = Phi_nf(:,j);
[~,I]=sort(abs(t1-E_MDMD),'ascend');
j = I(1);

u1 = multMODE(j,:); u1 = u1(:);
aa = mean(angle(u0(abs(u0)>std(u0))./u1(abs(u0)>std(u0))));
u1 = exp(1i*aa)*u1;

subplot(3,2,5)
u = reshape(real(u0),VelocityField.N+1,VelocityField.N+1);
contourf(xx2,yy2,u,150,'edgecolor','none')     % vorticty
axis square equal; 
box on;
colormap jet
clim([-3*std(u(:))+mean(u(:)),3*std(u(:))+mean(u(:))])
xlabel("$x$",Interpreter="latex")
ylabel("$y$",Interpreter="latex")

subplot(3,2,6)
u = reshape(real(u1),VelocityField.N+1,VelocityField.N+1);
contourf(xx2,yy2,u,150,'edgecolor','none')     % vorticty
axis square equal; 
box on;
colormap jet
clim([-3*std(u(:))+mean(u(:)),3*std(u(:))+mean(u(:))])
xlabel("$x$",Interpreter="latex")
ylabel("$y$",Interpreter="latex")