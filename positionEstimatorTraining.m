function [model] = positionEstimatorTraining(train)
% PCA
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
K = 10;

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
Uk = U(:,end-K+1:end);
Z = Uk' * X;

Za = [Z; ones(1,T)];
Wa = Y * Za' / (Za * Za');

W = Wa(:,1:end-1);
b = Wa(:,end);

model.U = Uk;
model.W = W;
model.b = b;
model.bin = bin;
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end

function [state] = handPosToHandState(pos)
  vel = pos(:,2:end) - pos(:,1:end-1);
  state = [pos(:,2:end); vel];
end
