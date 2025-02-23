function [px, py, newModel] = positionEstimator(test_data, model)
if numel(test_data.decodedHandPos) == 0
  model.x = [test_data.startHandPos(1:2); 0; 0; 0; 0];
  model.P =  diag([0, 0, 1e6, 1e6, 1e6, 1e6]);
  model.T0 = 302-model.bin;
end
T = size(test_data.spikes, 2);
A = model.A;
Q = model.Q;
H = model.H;
R = model.R;
x = model.x;
P = model.P;

Y = spikeTrainToSpikeRates(test_data.spikes(:, model.T0:T), model.bin);
for t = 1:size(Y, 2)
  y = Y(:,t);
  xh = A * x;
  Ph = A * P * A' + Q;
  
  z = y - H * xh;
  S = H * Ph * H' + R + 1e-8*eye(size(R));
  K = Ph * H' / S;
  
  x = xh + K * z;
  P = (eye(size(A)) - K*H) * Ph;
end

newModel = model;
newModel.x = x;
newModel.P = P;
newModel.T0 = T+2-model.bin;

px = x(1);
py = x(2);
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end