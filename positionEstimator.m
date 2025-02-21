function [px, py, newModelParameters] = positionEstimator(test_data, model)
if numel(test_data.decodedHandPos) == 0
  startPos = test_data.startHandPos(1:2);
  model.x = [startPos; 0; 0];
  model.T0 = 302-model.bin;
end
T = size(test_data.spikes, 2);
T0 = model.T0;
newModelParameters = model;

X = spikeTrainToSpikeRates(test_data.spikes(:, 1:T), model.bin);

Z  = model.U' * X;
Yh = model.W * Z + model.b;

px = Yh(1, end);
py = Yh(2, end);
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end