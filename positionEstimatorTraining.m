function [modelParameters] = positionEstimatorTraining(training_data)
%training_data = training_data(1:10,:);
total_samples_A = 0;
total_samples_H = 0;
for i = 1:numel(training_data)
  T = size(training_data(i).handPos, 2);
  total_samples_A = total_samples_A + (T - 2); % For state transitions
  total_samples_H = total_samples_H + (T - 1); % For observation model
end

X_A = zeros(total_samples_A, 4); % For state transitions (current state)
Y_A = zeros(total_samples_A, 4); % For state transitions (next state)
X_H = zeros(total_samples_H, 4); % For observation model (states)
Z_H = zeros(total_samples_H, size(training_data(1, 1).spikes, 1)); % For observation model
counter_A = 1;
counter_H = 1;

wbar = waitbar(0, "Training...");
for i = 1:numel(training_data)
  trial = training_data(i);
  handPos = trial.handPos(1:2, 300:end-100);
  spikes = trial.spikes(:,300:end-100);
  T = size(handPos, 2);
  
  % Compute state vectors [x, y, dx, dy]
  state_vectors = zeros(4, T-1);
  for t = 1:T-1
    pos = handPos(:, t);
    vel = handPos(:, t+1) - handPos(:, t);
    state_vectors(:, t) = [pos; vel];
  end
  
  % Collect data for A (state transition matrix)
  for t = 1:T-2
    X_A(counter_A, :) = state_vectors(:, t)';
    Y_A(counter_A, :) = state_vectors(:, t+1)';
    counter_A = counter_A + 1;
  end
  
  % Collect data for H (observation matrix)
  for t = 1:T-1
    X_H(counter_H, :) = state_vectors(:, t)';
    Z_H(counter_H, :) = spikes(:, t)';
    counter_H = counter_H + 1;
  end
  waitbar(i / numel(training_data), wbar); % Update progress
end
close(wbar);

% Estimate state transition matrix A and process noise W
A = (X_A \ Y_A)';
residuals_A = Y_A - X_A * A';
W = cov(residuals_A, 1);

% Estimate observation matrix H and measurement noise Q
H = (X_H \ Z_H)';
residuals_H = Z_H - X_H * H';
Q = cov(residuals_H, 1);

% Store model parameters
modelParameters.A = A;
modelParameters.H = H;
modelParameters.W = W;
modelParameters.Q = Q;
end