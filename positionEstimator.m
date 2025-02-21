function [px, py] = positionEstimator(test_data, model)
T = size(test_data.spikes, 2);
X = spikeTrainToSpikeRates(test_data.spikes(:, 1:T), model.bin);

y = [test_data.startHandPos(1:2); 0; 0];
for t = 1:size(X,2)
  z = model.U' * X(:,t);
  y = model.Wz * z + model.Wy * y + model.b;
end

px = y(1);
py = y(2);
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end