function [ Spectrum,Modes ] = KMD_FFT( Data , T )
% KMD_DFT  computes the Koopman modes of observations made on a
% measure-preserving dynamical system via FFT

% INPUTS:
% DATA: data matrix - each column represents the measurements/observations
% on the system at a single time instant - with uniform sampling in time

% T: observation interval - the time span between the first and last
% snapshot

% For example, if I measure m quantities on a system starting at time t0, 
% ending at time t1, with sampling time interval of dt, then Data would be 
% a m-by-n matrix where n=(t1-t0)/dt + 1


% OUTPUTS:
% Modes: mode matrix - m-by-n/2 matrix with each column presenting one
% Koopman mode. modes are sorted based on energy, so the first column is
% the Koopman mode with highest energy. The modes associated with complex
% eigenvalues are complex.

% Spectrum: first column is the Koopman frequency in rad/sec, second column
% is the Koopman frequency divided by the basic Fourier frequency, third
% column is the 2-norm of associated Koopman mode
% The Koopman frequency in i-th row of Spectrum is associated with the
% Koopman mode in i-th column of Modes



% Note: I assume Data is a real matrix, therefore the Fourier spectrum is
% symmetric and I cut in half to save memory



[~,n] = size (Data) ;        % n : # of snapshots

T= T*n/(n-1);               % adjusting time for FFT


DataMean = mean(Data,2);     % mean of the data is a Koopman mode

Data = bsxfun(@minus,Data,DataMean);      % taking out the average


wf = 2*pi/T  ;                      % fundamental frequency (rad/sec)
wn =(0:n/2)*wf;                     % Fourier frequencies (rad/sec)

F  = fft(Data')';                   % rowwise FFT

Fp = F(:,1:n/2+1)*T/n ;             % Complex Fourier amplitudes

A = 2*Fp/T;                         % Koopman Modes, unsorted
A(:,1)=  DataMean;                   % assigning the average to zero frequency

Cnorm = sqrt(sum(abs(A).^2,1));         % complex norm of Koopman Modes
[Norm, ind]=sort(Cnorm,'descend');      % sorting based on the norm
Modes = A(:,ind);                       % Koopman Modes sorted based on magnitude


Spectrum(:,2)= wn(ind)/wf;              % FFT(-Koopman) Harmonics  (integers)
Spectrum(:,1)= wn(ind);                 % Koopman Frequencies  (rad/sec)
Spectrum(:,3)= Norm;                    % Norm of Koopman modes 

end


%=========================================================================%
% Hassan Arbabi - 08-17-2015
% Mezic research group
% UC Santa Barbara
% arbabiha@gmail.com
%=========================================================================%
