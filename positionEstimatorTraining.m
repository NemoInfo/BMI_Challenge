function [modelParameters] = positionEstimatorTraining(tr)
clc % @TEMPORARY
% Arguments:

% - training_data:
%     training_data(n,k)              (n = trial id,  k = reaching angle)
%     training_data(n,k).trialId      unique number of the trial
%     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
%     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)

% ... train your model
N_neurons = size(tr(1,1).spikes,1);
N_tot_trials = numel(tr);

% Trim data
for i = 1:numel(tr)
  tr(i).spikes = tr(i).spikes(:,300:end-100);
  tr(i).handPos = tr(i).handPos(:,300:end-100);
end

% Train
W = rand(2, N_neurons);
b = rand(2, 1);
X = tr(3).spikes;
eta = 0.1;
epochs = 100;
RMSE = zeros(epochs,1);

figure;
rmse_plot = plot(nan, nan, 'r', 'LineWidth', 2);
%h = waitbar(0, 'Training...');
for epoch = 1:epochs
  for i = 1:numel(tr)
    T = size(tr(i).spikes,2);
    for t = 1:T
      X  = tr(i).spikes(:,t);
      Y  = tr(i).handPos(1:2,t);
      Yh = W * X + b;
      W  = W + eta*2/(T*N_tot_trials) * (Y - Yh) * X';
      b  = b + eta*2/(T*N_tot_trials) * (Y - Yh);
      RMSE(epoch) = RMSE(epoch) + 1/(T*N_tot_trials)*sqrt(sum((Y - Yh).^2));
    end
  end
  set(rmse_plot, 'XData', 1:epoch, 'YData', RMSE(1:epoch));
  drawnow;
  %waitbar(epoch / epochs, h); % Update progress
end
hold off;

modelParameters.W = W;
modelParameters.b = b;
end