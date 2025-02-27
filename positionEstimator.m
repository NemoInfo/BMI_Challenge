function [px, py, newModelParameters] = positionEstimator(test_data, modelParameters)
% - test_data:
%     test_data(m).trialID
%         unique trial ID
%     test_data(m).startHandPos
%         2x1 vector giving the [x y] position of the hand at the start
%         of the trial
%     test_data(m).decodedHandPos
%         [2xN] vector giving the hand position estimated by your
%         algorithm during the previous iterations.
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

% First time initialization
if ~isfield(modelParameters, 'mlp')
    % Initialize MLP with the pretrained parameters
    modelParameters.mlp = MLP(modelParameters.layer_sizes, modelParameters.learning_rate, modelParameters);
    modelParameters.bin = 20; % Default bin size for spike rate calculation
end

% Get current time window
T = size(test_data.spikes, 2);
if numel(test_data.decodedHandPos) == 0
    T0 = 1;
else
    T0 = T - modelParameters.bin + 1;
end

% Convert spike train to spike rates
spike_rates = spikeTrainToSpikeRates(test_data.spikes(:, T0:T), modelParameters.bin);

% Reshape spike_rates to match network input dimensions
% The network expects input as a matrix where each column is a sample
spike_rates = spike_rates'; % Transpose to make features as rows

% Use MLP to predict the position
[px, py] = modelParameters.mlp.predict(spike_rates);

% Update model parameters for next iteration
newModelParameters = modelParameters;
end

function [rates] = spikeTrainToSpikeRates(train, bin)
    kernel = ones(1, bin) / bin;  % averaging kernel
    rates = conv2(train, kernel, 'valid'); 
end