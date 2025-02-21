function [px, py, newModelParameters] = positionEstimator(test_data, modelParameters)
% - test_data:
%     test_data(m).trialID
%         unique trial ID
%     test_data(m).startHandPos
%         2x1 vector giving the [x y] position of the hand at the start
%         of the trial
%     test_data(m).decodedHandPos
%         [2xN] vector giving the hand position estimated by your
%         algorithm during the previous iterations. In this case, N is 
%         the number of times your function has been called previously on
%         the same data sequence.
%     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
%     in this case, t goes from 1 to the current time in steps of 20
%     Example:
%         Iteration 1 (t = 320):
%             test_data.trialID = 1;
%             test_data.startHandPos = [0; 0]
%             test_data.decodedHandPos = []
%             test_data.spikes = 98x320 matrix of spiking activity
%         Iteration 2 (t = 340):
%             test_data.trialID = 1;
%             test_data.startHandPos = [0; 0]
%             test_data.decodedHandPos = [2.3; 1.5]
%             test_data.spikes = 98x340 matrix of spiking activity
% Return Value:
% - [x, y]:
%     current position of the hand
if numel(test_data.decodedHandPos) == 0
  startPos = test_data.startHandPos(1:2);
  %modelParameters.x = ([startPos; 0; 0] - x_mean) ./ x_std;
  modelParameters.x = [startPos; 0; 0];
  modelParameters.P =  diag([0, 0, 1e6, 1e6]);
  modelParameters.T0 = 302-modelParameters.bin;
end
T = size(test_data.spikes, 2);
T0 = modelParameters.T0;
A = modelParameters.A;
Q = modelParameters.Q;
H = modelParameters.H;
R = modelParameters.R;
x = modelParameters.x;
P = modelParameters.P;

Y = spikeTrainToSpikeRates(test_data.spikes(:, T0:T), modelParameters.bin);
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

newModelParameters = modelParameters;
newModelParameters.x = x;
newModelParameters.P = P;
newModelParameters.T0 = T+2-modelParameters.bin;

px = x(1);
py = x(2);
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end