function [model] = positionEstimatorTraining(train)
% PCA Kalman
bin = 300;

progress = waitbar(100, "Computing spike rates...");

T = 0; % total training time
for i = 1:numel(train)
  train(i).spikes  = train(i).spikes(:, 301-bin:end);
  train(i).handPos = train(i).handPos(1:2,  299:end);
  train(i).states = handPosToHandState(train(i).handPos);
  train(i).spikes  = spikeTrainToSpikeRates(train(i).spikes, bin);
  assert(size(train(i).spikes, 2) == size(train(i).states, 2))
  T = T + size(train(i).spikes,2);
  waitbar(i/numel(train), progress)
end
close(progress)

N = size(train(1).spikes, 1);  % number of neurons 
S = size(train(1).states, 1); % size of state vector
K = 90;

X = zeros(N, T);
Y = zeros(S, T);
t = 1;
for i = 1:numel(train)
  Ti = size(train(i).spikes, 2);
  X(:,t:t+Ti-1) = train(i).spikes;
  Y(:,t:t+Ti-1) = train(i).states;
  t = t + Ti;
end

C = X * X' / T;
[U, ~] = eig(C);
U = U(:,end-K+1:end);

Y_Y    = zeros(S);
Yp_Ym  = zeros(S);
Ym_Ym  = zeros(S);
Z_Y    = zeros(K, S);

for i = 1:numel(train)
  y = train(i).states;
  z = U' * train(i).spikes;

  Yp_Ym = Yp_Ym + y(:,2:end)   * y(:,1:end-1)';
  Ym_Ym = Yp_Ym + y(:,1:end-1) * y(:,1:end-1)';
  Z_Y = Z_Y + z * y';
  Y_Y = Y_Y + y * y';
end

A = Yp_Ym / Ym_Ym;
H = Z_Y   / Y_Y;

Q = zeros(S);
R = zeros(K);
for i = 1:numel(train)
  y = train(i).states;
  z = U' * train(i).spikes;
  e = y(:,2:end) - A * y(:,1:end-1);
  n = z - H * y;
  
  Q = Q + e * e' / size(y,2);
  R = R + n * n' / size(y,2);
end
Q = Q / numel(train);
R = R / numel(train);


model.bin = bin;
model.U = U;
model.A = A;
model.Q = Q;
model.H = H;
model.R = R;
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end

function [state] = handPosToHandState(pos)
  vel = pos(:,2:end) - pos(:,1:end-1);
  state = [pos(:,2:end); vel];
end
