function [px, py, newModel] = positionEstimator(test_data, model)
if numel(test_data.decodedHandPos) == 0
  model.y  = [test_data.startHandPos(1:2); 0; 0];
  model.P  = diag([0, 0, 1e6, 1e6]);
  model.T0 = 1;
end
U = model.U;
A = model.A;
H = model.H;
Q = model.Q;
R = model.R;
y = model.y;
P = model.P;

T = size(test_data.spikes, 2);

X = spikeTrainToSpikeRates(test_data.spikes(:, model.T0:T), model.bin);
for t = 1:size(X,2)
  z  = U' * X(:,t);
  yh = A  * y;
  Ph = A  * P * A' + Q;

  v = z - H  * yh;
  S = H * Ph * H' + R + 1e-8* eye(size(R));
  K = Ph * H' / S;

  y = yh + K * v;
  P = (eye(size(A)) - K * H) *Ph;
end

newModel = model;
model.y = y;
model.P = P;
model.T0 = T + 2 - model.bin;

px = y(1);
py = y(2);
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end