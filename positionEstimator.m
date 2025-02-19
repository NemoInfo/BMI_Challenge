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
  modelParameters.x_est = [startPos; 0; 0];
  modelParameters.P_est = diag([1e-6, 1e-6, 1, 1]);
end
x_est = modelParameters.x_est;
P_est = modelParameters.P_est;

% Kalman filter prediction
A = modelParameters.A;
W = modelParameters.W;
x_pred = A * x_est;
P_pred = A * P_est * A' + W;

% Kalman filter update with latest spikes
H = modelParameters.H;
Q = modelParameters.Q;
z = test_data.spikes(:, end); % Latest spike counts

% Compute innovation covariance S
S = H * P_pred * H' + Q;

% Regularize S to avoid singularity
epsilon = 1e-6; % Small regularization term
S = S + epsilon * eye(size(S));

% Compute Kalman gain using pseudo-inverse
K = P_pred * H' * pinv(S);

% Update state estimate and covariance
y = z - H * x_pred;
x_est = x_pred + K * y;
P_est = (eye(4) - K * H) * P_pred;

% Update state
newModelParameters = modelParameters;
newModelParameters.x_est = x_est;
newModelParameters.P_est = P_est;

% Return estimated position
x = x_est(1);
y = x_est(2);
  
% Return Value:

% - [x, y]:
%     current position of the hand
end