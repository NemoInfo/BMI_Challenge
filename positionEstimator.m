function [px, py, newModel] = positionEstimator(test_data, model)
if numel(test_data.decodedHandPos) == 0
  model.x = [test_data.startHandPos(1:2); 0; 0; 0; 0];
  model.P =  diag([0, 0, 1e6, 1e6, 1e6, 1e6]);
  model.T0 = 302-model.bin;
  model.xs = [];
end
T = size(test_data.spikes, 2);
A = model.A;
Q = model.Q;
H = model.H;
R = model.R;
x = model.x;
P = model.P;

Y = spikeTrainToSpikeRates(test_data.spikes(:, model.T0:T), model.bin);
xs = zeros(2, size(Y, 2));
for t = 1:size(Y, 2)
  y = Y(:,t);
  xh = A * x;
  Ph = A * P * A' + Q;
  
  z = y - H * xh;
  S = H * Ph * H' + R + 1e-8*eye(size(R));
  K = Ph * H' / S;
  
  x = xh + K * z;
  xs(:, t) = x(1:2)';
  P = (eye(size(A)) - K*H) * Ph;
end
xs = [model.xs xs];

diffs = zeros(8, 1);
for d=1:8
  vh = xs(:,2:end) - xs(:,1:end-1); 
  v = squeeze(model.X(d, 1:size(xs, 2), :))';
  v = v(:,2:end) - v(:, 1:end-1);
  vh = mean(vh, 2);
  v = mean(v, 2);

  diffs(d) = norm(vh - v);
end

[~, d] = min(diffs);
%disp(d);

newModel = model;
newModel.xs = xs;
newModel.x = x;
newModel.P = P;
newModel.T0 = T+2-model.bin;

if T > 440
  px = model.X(d, T, 1);
  py = model.X(d, T, 2);
else
  px = x(1);
  py = x(2);
end

end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end