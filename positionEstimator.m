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
    % Initialize FastMLP with the pretrained parameters
    layer_sizes = modelParameters.layers;  % This now contains [input_size, hidden_sizes, output_size]
    modelParameters.mlp = FastMLP(layer_sizes, [], modelParameters);
    modelParameters.window_size = 20; % Window size for spike data
end

% Get current time window
T = size(test_data.spikes, 2);

% Use the most recent window_size samples
T0 = max(1, T - modelParameters.window_size + 1);

% Extract the relevant window of spike data
current_spikes = test_data.spikes(:, T0:T);

% Reshape the input to match the network's expected format
% The network expects input as a column vector [input_size x 1]
input_vector = reshape(current_spikes, [], 1);

% Make prediction
[px, py] = modelParameters.mlp.predict(input_vector);

% Update model parameters for next iteration
newModelParameters = modelParameters;
end

function [rates] = spikeTrainToSpikeRates(train, bin)
    kernel = ones(1, bin) / bin;  % averaging kernel
    rates = conv2(train, kernel, 'valid'); 
end