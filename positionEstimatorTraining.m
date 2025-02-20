function [modelParameters] = positionEstimatorTraining(train)
% Kalman modeling
% we need to compute A, H, Q and R
bin = 300;

progress = waitbar(100, "Computing spike rates...");
for i = 1:numel(train)
  train(i).spikes  = train(i).spikes(:, 300-bin+1:end-100);
  train(i).handPos = train(i).handPos(1:2, 300-1 :end-100);
  train(i).handPos = handPosToHandPosVel(train(i).handPos);
  train(i).spikes  = spikeTrainToSpikeRates(train(i).spikes, bin);
  assert(size(train(i).spikes, 2) == size(train(i).handPos, 2))
  waitbar(i/numel(train), progress)
end
close(progress)

X_Xm  = zeros(size(train(1),1), size(train(1),1));
Xm_Xm = zeros(size(X_Xm));

for i = 1:numel(train)
  x  = train(i).handPos(:,2:end);
  xm = train(i).handPos(:,1:end-1);
  X_Xm  = X_Xm  + x  * xm';
  Xm_Xm = Xm_Xm + xm * xm';
end

A = X_Xm / Xm_Xm;

Q = zeros(size(A));
for i = 1:numel(train)
  x  = train(i).handPos(:,2:end);
  xm = train(i).handPos(:,1:end-1);
  eps= x - A * xm;
  Q = Q + eps * eps' / size(x,2);
end

Q = Q / numel(train);

Y_X = zeros(size(train(i).spikes,1), size(A,1));
X_X = zeros(size(A));

for i = 1:numel(train)
  x = train(i).handPos;
  y = train(i).spikes;
  Y_X = Y_X + y * x';
  X_X = X_X + x * x';
end

H = Y_X / X_X;

R = zeros(size(train(i).spikes,1), size(train(i).spikes,1));

for i = 1:numel(train)
  x = train(i).handPos;
  y = train(i).spikes;
  n = y - H * x;
  R = R + n * n' / size(x,2);
end

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

function [state] = handPosToHandPosVel(pos)
  vel = pos(:,2:end) - pos(:,1:end-1);
  state = [pos(:,2:end); vel];
end
