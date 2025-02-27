classdef MLP
    properties
        layers          % Array of layer sizes
        weights        % Cell array of weight matrices
        biases        % Cell array of bias vectors
        learning_rate % Learning rate for training
        activations   % Storage for forward pass values
    end
    
    methods
        function obj = MLP(layer_sizes, learning_rate, modelParameters)
            
            % if modelParameters is not provided, initialize the network from scratch
            if nargin == 2
                % Constructor: Initialize network
                obj.layers = layer_sizes; % array of layer sizes (n total layers = 1 input + n-2 hidden + 1 output) 
                % it must have at least 3 layers
                if length(layer_sizes) < 3
                    error('MLP must have at least 3 layers. (Input, hidden, output)');
                end
                obj.learning_rate = learning_rate; % learning rate (1 value)
                obj.weights = cell(length(layer_sizes)-1, 1); % cell array of weight matrices (length(layer_sizes)-1 because we don't have a weight matrix for the input layer)
                obj.biases = cell(length(layer_sizes)-1, 1); % cell array of bias vectors (length(layer_sizes)-1 because we don't have a bias vector for the input layer)
                
                % Initialize weights and biases with Xavier initialization
                for i = 1:length(layer_sizes)-1
                    obj.weights{i} = randn(layer_sizes(i+1), layer_sizes(i)) * sqrt(2/layer_sizes(i));
                    obj.biases{i} = zeros(layer_sizes(i+1), 1);
                end
            elseif nargin == 1 || nargin == 3
                % Constructor: Initialize network
                obj.layers = modelParameters.layers; % array of layer sizes (n total layers = 1 input + n-2 hidden + 1 output) 
                obj.weights = modelParameters.weights;
                obj.biases = modelParameters.biases;
                % it must have at least 3 layers
                if length(layer_sizes) < 3
                    error('MLP must have at least 3 layers. (Input, hidden, output)');
                end
                obj.learning_rate = learning_rate;
            end
        end
        
        function [output, activations] = forward(obj, x)
            % Forward pass
            activations = cell(length(obj.layers), 1);
            activations{1} = x;
            
            for i = 1:length(obj.weights)
                z = obj.weights{i} * activations{i} + obj.biases{i};
                if i == length(obj.weights)
                    % Linear activation for output layer (regression)
                    activations{i+1} = z;
                else
                    % ReLU activation for hidden layers
                    activations{i+1} = max(0, z);
                end
            end
            output = activations{end};
        end
        
        function obj = backward(obj, x, y, activations)
            % Backward pass
            m = size(x, 2);  % batch size
            deltas = cell(length(obj.weights), 1);
            
            % Output layer error (MSE derivative)
            error = activations{end} - y;
            deltas{end} = error;
            
            % Backpropagate error
            for i = length(obj.weights)-1:-1:1
                deltas{i} = (obj.weights{i+1}' * deltas{i+1}) .* ...
                           (activations{i+1} > 0);  % ReLU derivative
            end
            
            % Update weights and biases
            for i = 1:length(obj.weights)
                obj.weights{i} = obj.weights{i} - ...
                    obj.learning_rate * (deltas{i} * activations{i}') / m;
                obj.biases{i} = obj.biases{i} - ...
                    obj.learning_rate * mean(deltas{i}, 2);
            end
        end
        
        function [x_pred, y_pred] = predict(obj, x)
            % Make prediction
            output = obj.forward(x);
            x_pred = output(1,:);
            y_pred = output(2,:);
        end
    end
end 