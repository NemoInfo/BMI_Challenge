function [px, py, newModelParameters] = positionEstimator(test_data, modelParameters)

if numel(test_data.decodedHandPos) == 0
  startPos = test_data.startHandPos(1:2);
  modelParameters.y0 = [startPos; 0; 0];
  modelParameters.T0 = 302-modelParameters.bin;
end
T0 = modelParameters.T0;
T = size(test_data.spikes, 2);
X = spikeTrainToSpikeRates(test_data.spikes(:, T0:T), modelParameters.bin);
y = modelParameters.y0;

for t=1:size(X,2)
  y = forward(X(:,t),y,modelParameters);
end

newModelParameters = modelParameters;
newModelParameters.y0 = y;
newModelParameters.T0 =  T+2-modelParameters.bin;

px = y(1);
py = y(2);
end

function [y] = forward(x, y, model)
  z = [x; y];
  h = model.Wi * z + model.bi;
  h = max(h,0);
  y = model.Wo * h + model.bo;
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end