function [ KoopmanFrequencies,ModeNorms,KoopmanModes,varargout ] = KMD_Laskar( Data, T,tau,NormFunction)
% KMD_Laskar computes the Koopman modes and frequencies 
% Orthogonal Matching Pursuit (OMP) on a grid of frequencies
% This is adapted from 
% Jacques Laskar, “The chaotic motion of the solar system: a numerical 
% estimate of the size of the chaotic zones” Icarus 88, 266–291 (1990).
% Also see Jacques Laskar, Claude Froeschl´e, and Alessandra Cel-
% letti, Physica D: Nonlinear Phenomena 56, 253–269 (1992).

% we assume data is real-valued


if ~exist('NormFunction','var')
    % if no norm function is specified we use the Euclidean norm of modes
    NormFunction = @(Modes) sqrt( sum ( abs(Modes.^2),1 ));
end

if ~exist('tau','var')
    % if no norm function is specified we use the Euclidean norm of modes
    tau = 1e-4;
end

disp('computing Koopman frequencies using Laskar algorithm')

% the mean flow is Koopman mode at frequency 0
DiscreteFreqs = 0;
Residual = bsxfun(@minus,Data,mean(Data,2));

% time
t = linspace(0,T,size(Data,2));




for Trial=1:100
    % 1. approximate the Koopman mode and frequencies using FFT
    disp('-------------------------')
    disp(['trial #',num2str(Trial)])
    disp('approximating KMD via FFT')
    [ w_star, Peak ] = DetectNew(Residual,T,NormFunction,tau,DiscreteFreqs) ;

    
    
    if ~strcmp(Peak,'found')
        disp('no peak was found ... terminating the search')
        break
    end
          
    disp(['a peak is found at ',num2str(w_star)]) 


    % 2. search in the vicinity of the w_star to find the Koopman frequency w_k
    disp('search for Koopman frequency')
    DeltaW = 2*pi/T;
    nW = 200;       % over-sampling factor
    wtest = linspace(w_star-DeltaW,w_star+DeltaW,nW);   % form a grid of frequencies

    % filtered harmonic average
    HAvg = HarmonicAverage( Data,wtest,t,'Hann');
    HAvgNorms = NormFunction(HAvg.');
    [~,ind_k]=max(HAvgNorms);
    w_k=wtest(ind_k);       
    disp(['the Koopman frequency is ',num2str(w_k)])
    
    % 3. add the frequency to the dictionary
    % since the data is real we also add the negated w_k
    DiscreteFreqs=[DiscreteFreqs,w_k,-w_k];
    ExpDictionary = exp(1i*t'*DiscreteFreqs);

    % 4. The least-square problem
    Modes = ((ExpDictionary'*ExpDictionary)\(ExpDictionary'*Data') );

    % 5. Compute the new Residual
    Residual = Data' - ExpDictionary*Modes;
    Residual = Residual.' ;
    % continue till no candidate frequencies are found
end

% one more least-square problem
  ExpDictionary = exp(1i*t'*DiscreteFreqs);
  Modes = ((ExpDictionary'*ExpDictionary)\(ExpDictionary'*Data') );


    KoopmanModes = (Modes(DiscreteFreqs>=0,:)) .';
    KoopmanFrequencies= DiscreteFreqs(DiscreteFreqs>=0);
    KoopmanModes(:,2:end)=2*KoopmanModes(:,2:end);


    KoopmanFrequencies=KoopmanFrequencies';
    ModeNorms = NormFunction(KoopmanModes)';
% optional output - number of trials
if nargout==4
    varargout{1}=Trial; % number of peaks extracted
end
end




function [ w_star, Peak ] = DetectNew( Residual,T,NormFunction,tau,PreviousFreqs)
% finds a peak of spectrum that satisfies the conditions below using MATLAB
disp('searching for peak ...')
% initialize
Peak='not_found';
w_star=[];


[ frequencies,modes] = KMD_FFT( Residual, T );
[frequencies,indf] = sort(frequencies(:,1));
modes = modes(:,indf);

dw = frequencies(2)-frequencies(1);
  
Norms = [NormFunction(modes),0,0,0];           % computing the norms of associated norms

[Peaks,P_ind] = findpeaks(Norms);      % finding the peaks 
[~,Ind]=sort(Peaks,'descend');      % extra sorting
P_ind=  P_ind(Ind);
P_ind =  P_ind(P_ind>3);        % - skipping frequencies too close to zero

    
for ip = P_ind            
            
        w_candidate = frequencies(ip);
        %%two conditions must be saisfied for w_s to be a distinguished peaks:
        % C1. it must be (%5) away from the previously-found Koopman frequencies
        Dist2Peaks  = abs(PreviousFreqs-w_candidate)> 5*dw; %0.05*PrevFreq ;
        % C2. its peak must be at least two times higher than the average of its
        % 6 neighbors
        IsPeak = Norms(ip) > 2* mean( Norms([ip-3:ip-1,ip+1:ip+3]) ) ;
        
        % C3 satisfy the norm threshold
        SatisfyThresh = Norms(ip)> tau;
        
        
        if  (Dist2Peaks & IsPeak & SatisfyThresh)  
            w_star = w_candidate;
            Peak = 'found';
            break
        end

end
end





function [ Average ] = HarmonicAverage( Data,w,t,Filter)
% computes the harmonic average using the Filter

HarmonicWeight = 2*exp(-1i*w'*t);            

% using the specified filter
if exist('Filter','var')
   switch Filter
       case 'Hamming'
           HarmonicWeight=bsxfun(@times,HarmonicWeight,hamming(length(t))');
       case 'Hann'
           HarmonicWeight=bsxfun(@times,HarmonicWeight,hann(length(t))');
       case 'exponential'       % the exponential weighting defined by Das & Yorke 2015
           HarmonicWeight=bsxfun(@times,HarmonicWeight,ExpWeight(length(t))*length(t));
       otherwise
           disp('filter not defined ... no filter used')
   end 
end

Average = HarmonicWeight*Data'./length(t);     

end


function w = ExpWeight(N)
% see "Super convergence of ergodic averages for quasiperiodic orbits" by
% Das & Yorke, 2015 in arXiv

n = (1:N)/N;
w = exp(1./( n.*(n-1) ));
w(1)=0;
w(end)=0;

w=w/sum(w);
end
