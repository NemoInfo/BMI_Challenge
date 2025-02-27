classdef FastMLP
    properties
        weights         % Cell array of weight matrices
        biases         % Cell array of bias vectors
        learning_rate  % Learning rate for training
        
        % Adam optimizer parameters
        beta1 = 0.9     % Exponential decay rate for first moment
        beta2 = 0.999   % Exponential decay rate for second moment
        epsilon = 1e-8  % Small constant for numerical stability
        
        % Adam momentum variables
        m_weights      % Cell array of first moments for weights
        v_weights      % Cell array of second moments for weights
        m_biases       % Cell array of first moments for biases
        v_biases       % Cell array of second moments for biases
        
        t = 0          % Time step counter for Adam
    end
    
    methods
        function obj = FastMLP(layer_sizes, ~, modelParameters)
            % Higher learning rate for Adam
            obj.learning_rate = 0.001;
            
            if nargin == 2 || nargin == 1
                % Verify at least input and output layers
                if length(layer_sizes) < 2
                    error('FastMLP requires at least 2 layers (input and output)');
                end
                
                num_layers = length(layer_sizes);
                obj.weights = cell(num_layers - 1, 1);
                obj.biases = cell(num_layers - 1, 1);
                obj.m_weights = cell(num_layers - 1, 1);
                obj.v_weights = cell(num_layers - 1, 1);
                obj.m_biases = cell(num_layers - 1, 1);
                obj.v_biases = cell(num_layers - 1, 1);
                
                % Initialize each layer
                for i = 1:(num_layers-1)
                    % Xavier initialization with scaling factor
                    scale = sqrt(1/layer_sizes(i)) * 0.1;
                    
                    % Initialize weights and biases
                    obj.weights{i} = randn(layer_sizes(i+1), layer_sizes(i)) * scale;
                    obj.biases{i} = zeros(layer_sizes(i+1), 1);
                    
                    % Initialize Adam momentum variables
                    obj.m_weights{i} = zeros(size(obj.weights{i}));
                    obj.v_weights{i} = zeros(size(obj.weights{i}));
                    obj.m_biases{i} = zeros(size(obj.biases{i}));
                    obj.v_biases{i} = zeros(size(obj.biases{i}));
                end
            elseif nargin == 3
                % Load from model parameters
                num_layers = length(modelParameters.weights);
                obj.weights = modelParameters.weights;
                obj.biases = modelParameters.biases;
                
                % Initialize Adam momentum variables
                obj.m_weights = cell(num_layers, 1);
                obj.v_weights = cell(num_layers, 1);
                obj.m_biases = cell(num_layers, 1);
                obj.v_biases = cell(num_layers, 1);
                
                for i = 1:num_layers
                    obj.m_weights{i} = zeros(size(obj.weights{i}));
                    obj.v_weights{i} = zeros(size(obj.weights{i}));
                    obj.m_biases{i} = zeros(size(obj.biases{i}));
                    obj.v_biases{i} = zeros(size(obj.biases{i}));
                end
            end
        end
        
        function [param, m_out, v_out] = adam_update(obj, param, m, v, grad)
            % Update biased first moment estimate
            m_out = obj.beta1 * m + (1 - obj.beta1) * grad;
            % Update biased second raw moment estimate
            v_out = obj.beta2 * v + (1 - obj.beta2) * (grad .* grad);
            
            % Compute bias-corrected first moment estimate
            m_hat = m_out / (1 - obj.beta1^obj.t);
            % Compute bias-corrected second raw moment estimate
            v_hat = v_out / (1 - obj.beta2^obj.t);
            
            % Update parameters
            param = param - obj.learning_rate * m_hat ./ (sqrt(v_hat) + obj.epsilon);
        end
        
        function [output, cache] = forward(obj, x)
            num_layers = length(obj.weights);
            cache.x = cell(num_layers + 1, 1);
            cache.z = cell(num_layers, 1);
            
            % Normalize input
            x_mean = mean(x, 2);
            x_std = std(x, 0, 2) + eps;
            cache.x{1} = (x - x_mean) ./ x_std;
            
            % Forward through all layers
            for i = 1:num_layers
                % Compute pre-activation
                cache.z{i} = obj.weights{i} * cache.x{i} + obj.biases{i};
                
                if i == num_layers
                    % Linear output for last layer
                    cache.x{i+1} = cache.z{i};
                else
                    % ReLU activation for hidden layers
                    cache.x{i+1} = max(0, cache.z{i});
                end
            end
            
            output = cache.x{end};
        end
        
        function obj = backward(obj, ~, y, cache)
            batch_size = size(y, 2);
            obj.t = obj.t + 1;  % Increment time step
            num_layers = length(obj.weights);
            
            % Initialize gradients
            dz = cell(num_layers, 1);
            
            % Output layer gradient
            dz{num_layers} = (cache.x{num_layers+1} - y) / batch_size;
            
            % Backpropagate through hidden layers
            for i = num_layers-1:-1:1
                % Gradient of hidden layer
                da = obj.weights{i+1}' * dz{i+1};
                dz{i} = da .* (cache.z{i} > 0);  % ReLU derivative
            end
            
            % Update all layers
            for i = 1:num_layers
                % Compute weight and bias gradients
                dW = dz{i} * cache.x{i}';
                db = sum(dz{i}, 2);
                
                % Clip gradients
                clip_threshold = 1.0;
                dW = min(max(dW, -clip_threshold), clip_threshold);
                db = min(max(db, -clip_threshold), clip_threshold);
                
                % Update using Adam optimizer
                [obj.weights{i}, obj.m_weights{i}, obj.v_weights{i}] = ...
                    obj.adam_update(obj.weights{i}, obj.m_weights{i}, obj.v_weights{i}, dW);
                [obj.biases{i}, obj.m_biases{i}, obj.v_biases{i}] = ...
                    obj.adam_update(obj.biases{i}, obj.m_biases{i}, obj.v_biases{i}, db);
            end
        end
        
        function [x_pred, y_pred] = predict(obj, x)
            % Normalize input
            x_mean = mean(x, 2);
            x_std = std(x, 0, 2) + eps;
            a = (x - x_mean) ./ x_std;
            
            % Forward pass through all layers
            for i = 1:length(obj.weights)
                z = obj.weights{i} * a + obj.biases{i};
                if i == length(obj.weights)
                    % Linear output for last layer
                    a = z;
                else
                    % ReLU activation for hidden layers
                    a = max(0, z);
                end
            end
            
            % Extract predictions
            x_pred = a(1, :);
            y_pred = a(2, :);
        end
        
        function loss = compute_loss(~, y_pred, y_true)
            % MSE computation with safety checks
            diff = y_pred - y_true;
            squared_diff = diff .* diff;
            
            % Check for invalid values
            valid_mask = ~isnan(squared_diff) & ~isinf(squared_diff);
            if any(valid_mask(:))
                % Apply mask and compute mean
                squared_diff(~valid_mask) = 0;  % Zero out invalid values
                loss = mean(sum(squared_diff, 1));
            else
                loss = Inf;
            end
        end
    end
end 