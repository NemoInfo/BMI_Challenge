classdef FastMLP
    properties
        layers          % Array of layer sizes
        weights        % Cell array of weight matrices
        biases        % Cell array of bias vectors
        learning_rate % Learning rate for training
    end
    
    methods
        function obj = FastMLP(layer_sizes, learning_rate, modelParameters)
            % Constructor with optional model parameters loading
            if nargin == 2
                % Initialize new network
                obj.layers = layer_sizes;
                if length(layer_sizes) < 3
                    error('FastMLP must have at least 3 layers (Input, hidden, output)');
                end
                obj.learning_rate = learning_rate;
                
                % Preallocate cells for weights and biases
                num_layers = length(layer_sizes);
                obj.weights = cell(num_layers - 1, 1);
                obj.biases = cell(num_layers - 1, 1);
                
                % Xavier/He initialization for better convergence
                for i = 1:num_layers-1
                    % He initialization for ReLU
                    obj.weights{i} = randn(layer_sizes(i+1), layer_sizes(i)) * sqrt(2/layer_sizes(i));
                    obj.biases{i} = zeros(layer_sizes(i+1), 1);
                end
            elseif nargin == 1 || nargin == 3
                % Load from model parameters
                obj.layers = modelParameters.layers;
                obj.weights = modelParameters.weights;
                obj.biases = modelParameters.biases;
                if length(layer_sizes) < 3
                    error('FastMLP must have at least 3 layers (Input, hidden, output)');
                end
                obj.learning_rate = learning_rate;
            end
        end
        
        function [output, cache] = forward(obj, x)
            % Optimized forward pass with pre-allocation and vectorization
            num_layers = length(obj.layers);
            batch_size = size(x, 2);
            
            % Pre-allocate cache for activations and linear combinations
            cache.Z = cell(num_layers - 1, 1);
            cache.A = cell(num_layers, 1);
            cache.A{1} = x;  % Input layer activation
            
            % Vectorized forward propagation
            for i = 1:num_layers-1
                % Efficient matrix multiplication with broadcasting
                cache.Z{i} = obj.weights{i} * cache.A{i} + obj.biases{i};
                
                if i == num_layers-1
                    % Linear activation for output layer (regression)
                    cache.A{i+1} = cache.Z{i};
                else
                    % Vectorized ReLU for hidden layers
                    cache.A{i+1} = max(0, cache.Z{i});
                end
            end
            
            output = cache.A{end};
        end
        
        function obj = backward(obj, x, y, cache)
            % Optimized backward pass with vectorized operations
            batch_size = size(x, 2);
            num_layers = length(obj.layers);
            
            % Pre-allocate gradients
            dW = cell(num_layers - 1, 1);
            db = cell(num_layers - 1, 1);
            dA = cell(num_layers, 1);
            
            % Output layer error (MSE derivative)
            dA{num_layers} = (cache.A{end} - y) / batch_size;
            
            % Vectorized backward propagation
            for i = num_layers-1:-1:1
                if i == num_layers-1
                    dZ = dA{i+1};  % Output layer uses linear activation
                else
                    % Vectorized ReLU derivative
                    dZ = dA{i+1} .* (cache.Z{i} > 0);
                end
                
                % Compute gradients efficiently
                dW{i} = dZ * cache.A{i}';
                db{i} = sum(dZ, 2);
                
                if i > 1  % Skip for input layer
                    dA{i} = obj.weights{i}' * dZ;
                end
            end
            
            % Vectorized parameter updates
            for i = 1:num_layers-1
                obj.weights{i} = obj.weights{i} - obj.learning_rate * dW{i};
                obj.biases{i} = obj.biases{i} - obj.learning_rate * db{i};
            end
        end
        
        function [x_pred, y_pred] = predict(obj, x)
            % Vectorized prediction
            [output, ~] = obj.forward(x);
            x_pred = output(1, :);
            y_pred = output(2, :);
        end
        
        function loss = compute_loss(~, y_pred, y_true)
            % Vectorized MSE computation
            loss = mean(sum((y_pred - y_true).^2, 1));
        end
    end
end 