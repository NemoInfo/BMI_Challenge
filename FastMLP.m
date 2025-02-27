classdef FastMLP
    properties
        layers          % Array of layer sizes
        weights        % Cell array of weight matrices
        biases        % Cell array of bias vectors
        learning_rate % Learning rate for training
        momentum      % Momentum coefficient for faster training
        velocity_w    % Velocity for weights (momentum)
        velocity_b    % Velocity for biases (momentum)
    end
    
    methods
        function obj = FastMLP(layer_sizes, learning_rate, modelParameters)
            % Clear any existing properties
            obj.layers = [];
            obj.weights = {};
            obj.biases = {};
            obj.learning_rate = 0;
            obj.momentum = 0;
            obj.velocity_w = {};
            obj.velocity_b = {};
            
            if nargin == 2
                % Initialize new network
                obj.layers = layer_sizes;
                if length(layer_sizes) < 3
                    error('FastMLP must have at least 3 layers (Input, hidden, output)');
                end
                obj.learning_rate = learning_rate;
                obj.momentum = 0.9;  % Add momentum for faster convergence
                
                % Preallocate cells
                num_layers = length(layer_sizes);
                obj.weights = cell(num_layers - 1, 1);
                obj.biases = cell(num_layers - 1, 1);
                obj.velocity_w = cell(num_layers - 1, 1);
                obj.velocity_b = cell(num_layers - 1, 1);
                
                % He initialization for better convergence
                for i = 1:num_layers-1
                    fan_in = layer_sizes(i);
                    scale = sqrt(2/fan_in);
                    obj.weights{i} = randn(layer_sizes(i+1), fan_in) * scale;
                    obj.biases{i} = zeros(layer_sizes(i+1), 1);
                    obj.velocity_w{i} = zeros(size(obj.weights{i}));
                    obj.velocity_b{i} = zeros(size(obj.biases{i}));
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
                obj.momentum = 0.9;
                
                % Initialize velocities
                num_layers = length(obj.layers);
                obj.velocity_w = cell(num_layers - 1, 1);
                obj.velocity_b = cell(num_layers - 1, 1);
                for i = 1:num_layers-1
                    obj.velocity_w{i} = zeros(size(obj.weights{i}));
                    obj.velocity_b{i} = zeros(size(obj.biases{i}));
                end
            end
        end
        
        function [output, cache] = forward(obj, x)
            % Fast forward pass with minimal memory allocation
            num_layers = length(obj.layers);
            
            % Pre-allocate cache
            cache.Z = cell(num_layers - 1, 1);
            cache.A = cell(num_layers, 1);
            cache.A{1} = x;
            
            % Forward propagation with optimized operations
            for i = 1:num_layers-1
                % Efficient matrix multiplication with broadcasting
                cache.Z{i} = bsxfun(@plus, obj.weights{i} * cache.A{i}, obj.biases{i});
                
                if i == num_layers-1
                    % Linear output layer (no activation)
                    cache.A{i+1} = cache.Z{i};
                else
                    % Fast ReLU with in-place operation
                    cache.A{i+1} = max(0, cache.Z{i});
                end
            end
            
            output = cache.A{end};
        end
        
        function obj = backward(obj, x, y, cache)
            % Fast backward pass with momentum
            batch_size = size(x, 2);
            num_layers = length(obj.layers);
            
            % Initialize gradient at output layer
            dZ = (cache.A{end} - y) / batch_size;
            
            % Backward propagation with momentum
            for i = num_layers-1:-1:1
                % Compute gradients
                dW = dZ * cache.A{i}';
                db = sum(dZ, 2);
                
                % Update velocities (momentum)
                obj.velocity_w{i} = obj.momentum * obj.velocity_w{i} + obj.learning_rate * dW;
                obj.velocity_b{i} = obj.momentum * obj.velocity_b{i} + obj.learning_rate * db;
                
                % Update parameters
                obj.weights{i} = obj.weights{i} - obj.velocity_w{i};
                obj.biases{i} = obj.biases{i} - obj.velocity_b{i};
                
                % Compute gradient for next layer
                if i > 1
                    dZ = (obj.weights{i}' * dZ) .* (cache.Z{i-1} > 0);
                end
            end
        end
        
        function [x_pred, y_pred] = predict(obj, x)
            % Fast prediction without caching
            num_layers = length(obj.layers);
            a = x;
            
            % Forward pass without storing intermediates
            for i = 1:num_layers-1
                z = bsxfun(@plus, obj.weights{i} * a, obj.biases{i});
                if i == num_layers-1
                    a = z;
                else
                    a = max(0, z);
                end
            end
            
            % Extract predictions
            x_pred = a(1, :);
            y_pred = a(2, :);
        end
        
        function loss = compute_loss(~, y_pred, y_true)
            % Fast MSE computation
            diff = y_pred - y_true;
            loss = mean(sum(diff .* diff, 1));
        end
    end
end 