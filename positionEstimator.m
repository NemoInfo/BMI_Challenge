function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

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
    
  % ... compute position at the given timestep.
if numel(test_data.decodedHandPos) == 0
  startPos = test_data.startHandPos(1:2);
  modelParameters.x = [startPos; rand(2,1)*5];
  modelParameters.P = zeros(size(modelParameters.A));
end
A = modelParameters.A;
Q = modelParameters.Q;
H = modelParameters.H;
R = modelParameters.R;
x = modelParameters.x;
P = modelParameters.P;
y = spikeTrainToSpikeRates(test_data.spikes(:, end-19:end), 20);
y = y(:,end);

xh = A * x;
Ph = A * P * A' + Q;

z = y - H * xh;
S = H * Ph * H' + R + 1e6*eye(size(R)); % + 1e-8 * eye(size(R))
K = Ph * H' / S;

newModelParameters = modelParameters;
newModelParameters.x = xh + K * z;
newModelParameters.P = (eye(size(A)) - K*H) * Ph;
% Return Value:

x = newModelParameters.x(1);
y = newModelParameters.x(2);
% - [x, y]:
%     current position of the hand
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  rates = zeros(size(train) - [0 bin]);

  for i = 1:size(train,1)
    for t = bin:size(train,2)
      rates(i, t-bin+1) = sum(train(i, t-bin+1:t));
    end
  end
end