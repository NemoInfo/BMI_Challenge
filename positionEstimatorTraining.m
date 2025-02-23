function [modelParameters] = positionEstimatorTraining(train)
% Kalman modeling
% we need to compute A, H, Q and R
bin = 300;

progress = waitbar(100, "Computing spike rates...");
for i = 1:numel(train)
  train(i).spikes  = train(i).spikes(:, 301-bin:end);
  train(i).handPos = train(i).handPos(1:2,  298:end);
  train(i).states = handPosToHandState(train(i).handPos);
  train(i).spikes  = spikeTrainToSpikeRates(train(i).spikes, bin);
  assert(size(train(i).spikes, 2) == size(train(i).states, 2))
  waitbar(i/numel(train), progress)
end
close(progress)

N = size(train(1).spikes, 1);  % number of neurons
P = size(train(1).states, 1);  % size of state vector

X_X    = zeros(P, P);
Xp_Xm  = zeros(P, P);
Xm_Xm  = zeros(P, P);
Y_X    = zeros(N, P);

for i = 1:numel(train)
  x = train(i).states;
  y = train(i).spikes;

  Xp_Xm = Xp_Xm + x(:,2:end)   * x(:,1:end-1)';
  Xm_Xm = Xm_Xm + x(:,1:end-1) * x(:,1:end-1)';
  Y_X = Y_X + y * x';
  X_X = X_X + x * x';
end

A = Xp_Xm / Xm_Xm;
H = Y_X / X_X;

Q = zeros(P, P);
R = zeros(N, N);
for i = 1:numel(train)
  x = train(i).states;
  y = train(i).spikes;
  e = x(:,2:end) - A * x(:,1:end-1);
  n = y - H * x;

  Q = Q + e * e' / size(x,2);
  R = R + n * n' / size(x,2);
end
Q = Q / numel(train);
R = R / numel(train);

modelParameters.A = A;
modelParameters.bin = bin;
modelParameters.Q = Q;
modelParameters.H = H;
modelParameters.R = R;
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end

function [state] = handPosToHandState(pos)
  vel = pos(:,2:end) - pos(:,1:end-1);
  acc = vel(:,2:end) - vel(:,1:end-1);
  state = [pos(:,3:end); vel(:,2:end); acc];
end
