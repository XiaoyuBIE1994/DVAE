% select a pair of signals from Xiaoyu's examples

cd '/Users/girinl/Downloads/2019-2020_RVAE/test_results_20200227'
name1 = 'man_440_';
k = 2;
cd WSJ0_simon_FFNN_VAE_latent_dim=16

name = [name1 'origin-' num2str(k) '.wav'];
[x, fs] = audioread(name);

name = [name1 'recon-' num2str(k) '.wav'];
[y, fs] = audioread(name);

Ts = 1/fs; % sampling period

% y is longer than x, and we have to crop the signals to L samples.
% Because later I will apply a STFT with N-points analysis window
% with H = N/2, I set L = M*H, with M an entire number of frames. 
N = 1024; H = N/2;
L = length(y);
M = fix(L/H);   % number of STFT frames
L = M*H;        % new signal length
x = x(1:L);
y = y(1:L);

% plot the signals
figure(1); clf; plot(x,'g'); hold on; plot(y,'b');
% On this plot we can see that: 
% 1) the two signals are in phase,
% 2) the two signals have a different scaling (they are not 'aligned' on the
% y-axis)

% Calculation of time-domain (TD) RMSE between x and y
RMSE_raw = sqrt(sumsqr(x - y)/L);

%we obtain: RMSE_raw = 0.03706170

% Now, let us align the 2 signals on the y-axis = rescale one of them, say
% y, on the other one, say x. This alignment is done in the MSE sense, i.e.
% we look for the scalar coefficient a so that ||x-a*y||^2 is minimized.
a = y\x;    % this is '1D' linear regression
y_scaled = a*y;
RMSE_scaled = sqrt(sumsqr(x - y_scaled)/L);

%we obtain: RMSE_scaled = 0.03609790

% The relative difference is:
(RMSE_raw - RMSE_scaled)/RMSE_raw; 

% On this example, we get 0.0260030 = 2.6%. This is not that much, but for some signals it is larger.

% And now for the STFT domain. 
% (for one frame let us denote FD = frequency-domain).
% We work with rescaled signals.
% For simplicity of notations, we set y = y_scaled.
y = y_scaled;

w = hamming(N);                     % analysis window
w = ones(N,1);
x =[zeros(H,1); x; zeros(H,1)];     % zero-padding
X = zeros(N,M);                     % initialisation of STFT matrix of x
x_mat = zeros(N,M);                 % To store the successive frames of x
y =[zeros(H,1); y; zeros(H,1)];     % zero-padding
Y = zeros(N,M);                     % initialisation of STFT matrix of y
y_mat = zeros(N,M);                 % To store the successive frames of y

for m = 1 : M
    frame = x(1 + (m-1)*H: N + (m-1)*H).*w; % framing
    x_mat(:,m) = frame;
    X(:,m) = fft(frame);
    
    frame = y(1 + (m-1)*H: N + (m-1)*H).*w; % framing
    y_mat(:,m) = frame;
    Y(:,m) = fft(frame);
end

% Let us compare TD-RMSE and FD-RMSE at frame level.
e_mat = x_mat - y_mat;  % error signal in the TD
E = X - Y;              % error signal in the FD

% TD-RMSE
RMSE_TD_frame = sqrt(diag(e_mat'*e_mat)/N);

% FD-RMSE
RMSE_FD_frame = sqrt(diag(E'*E)/N^2);

% Here, we can check that we have exactly RMSE_TD_frame = RMSE_FD_frame.
% This is Persyval theorem (digital signal version).
% Note that we have N at the denominator in TD whereas we
% have N^2 at the denominator in FD. This is perfectly OK.

% Now if we calculate TD-RMSE on the complete signal x, we get a different
% result because of the framing + STFT analysis window effect. 
RMSE_TD = sqrt((x-y)'*(x-y)/L); % (same as RMSE_scaled)
