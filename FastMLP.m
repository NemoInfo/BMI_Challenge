classdef FastMLP
    properties
        input_weights    % Input to first hidden layer weights
        hidden_weights   % First to second hidden layer weights
        output_weights  % Second hidden to output layer weights
        input_biases    % First hidden layer biases
        hidden_biases   % Second hidden layer biases
        output_biases   % Output layer biases
        learning_rate   % Learning rate for training
        
        % Adam optimizer parameters
        beta1 = 0.9     % Exponential decay rate for first moment
        beta2 = 0.999   % Exponential decay rate for second moment
        epsilon = 1e-8  % Small constant for numerical stability
        
        % Adam momentum variables
        m_input_w      % First moment for input weights
        v_input_w      % Second moment for input weights
        m_hidden_w     % First moment for hidden weights
        v_hidden_w     % Second moment for hidden weights
        m_output_w     % First moment for output weights
        v_output_w     % Second moment for output weights
        
        m_input_b      % First moment for input biases
        v_input_b      % Second moment for input biases
        m_hidden_b     % First moment for hidden biases
        v_hidden_b     % Second moment for hidden biases
        m_output_b     % First moment for output biases
        v_output_b     % Second moment for output biases
        
        t = 0          % Time step counter for Adam
    end
    
    methods
        function obj = FastMLP(layer_sizes, ~, modelParameters)
            % Higher learning rate for Adam
            obj.learning_rate = 0.001;
            
            if nargin == 2 || nargin == 1
                % Verify four layers (input, hidden1, hidden2, output)
                if length(layer_sizes) ~= 4
                    error('FastMLP requires exactly 4 layers (input, hidden1, hidden2, output)');
                end
                
                % Initialize weights with scaled Xavier initialization
                input_size = layer_sizes(1);
                hidden1_size = layer_sizes(2);
                hidden2_size = layer_sizes(3);
                output_size = layer_sizes(4);
                
                scale1 = sqrt(1/input_size) * 0.1;
                scale2 = sqrt(1/hidden1_size) * 0.1;
                scale3 = sqrt(1/hidden2_size) * 0.1;
                
                % Initialize weights and biases
                obj.input_weights = randn(hidden1_size, input_size) * scale1;
                obj.hidden_weights = randn(hidden2_size, hidden1_size) * scale2;
                obj.output_weights = randn(output_size, hidden2_size) * scale3;
                obj.input_biases = zeros(hidden1_size, 1);
                obj.hidden_biases = zeros(hidden2_size, 1);
                obj.output_biases = zeros(output_size, 1);
                
                % Initialize Adam momentum variables to zeros
                obj.m_input_w = zeros(size(obj.input_weights));
                obj.v_input_w = zeros(size(obj.input_weights));
                obj.m_hidden_w = zeros(size(obj.hidden_weights));
                obj.v_hidden_w = zeros(size(obj.hidden_weights));
                obj.m_output_w = zeros(size(obj.output_weights));
                obj.v_output_w = zeros(size(obj.output_weights));
                
                obj.m_input_b = zeros(size(obj.input_biases));
                obj.v_input_b = zeros(size(obj.input_biases));
                obj.m_hidden_b = zeros(size(obj.hidden_biases));
                obj.v_hidden_b = zeros(size(obj.hidden_biases));
                obj.m_output_b = zeros(size(obj.output_biases));
                obj.v_output_b = zeros(size(obj.output_biases));
            elseif nargin == 3
                % Load from model parameters
                obj.input_weights = modelParameters.weights{1};
                obj.hidden_weights = modelParameters.weights{2};
                obj.output_weights = modelParameters.weights{3};
                obj.input_biases = modelParameters.biases{1};
                obj.hidden_biases = modelParameters.biases{2};
                obj.output_biases = modelParameters.biases{3};
                
                % Initialize Adam momentum variables to zeros
                obj.m_input_w = zeros(size(obj.input_weights));
                obj.v_input_w = zeros(size(obj.input_weights));
                obj.m_hidden_w = zeros(size(obj.hidden_weights));
                obj.v_hidden_w = zeros(size(obj.hidden_weights));
                obj.m_output_w = zeros(size(obj.output_weights));
                obj.v_output_w = zeros(size(obj.output_weights));
                
                obj.m_input_b = zeros(size(obj.input_biases));
                obj.v_input_b = zeros(size(obj.input_biases));
                obj.m_hidden_b = zeros(size(obj.hidden_biases));
                obj.v_hidden_b = zeros(size(obj.hidden_biases));
                obj.m_output_b = zeros(size(obj.output_biases));
                obj.v_output_b = zeros(size(obj.output_biases));
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
            % Normalize input
            x_mean = mean(x, 2);
            x_std = std(x, 0, 2) + eps;
            x_norm = (x - x_mean) ./ x_std;
            
            % First hidden layer
            z1 = obj.input_weights * x_norm + obj.input_biases;
            a1 = max(0, z1);  % ReLU activation
            
            % Second hidden layer
            z2 = obj.hidden_weights * a1 + obj.hidden_biases;
            a2 = max(0, z2);  % ReLU activation
            
            % Output layer (linear)
            z3 = obj.output_weights * a2 + obj.output_biases;
            
            % Store cache for backward pass
            cache.x_norm = x_norm;
            cache.z1 = z1;
            cache.a1 = a1;
            cache.z2 = z2;
            cache.a2 = a2;
            cache.z3 = z3;
            
            output = z3;
        end
        
        function obj = backward(obj, ~, y, cache)
            batch_size = size(y, 2);
            obj.t = obj.t + 1;  % Increment time step
            
            % Output layer gradient
            dz3 = (cache.z3 - y) / batch_size;
            
            % Second hidden layer gradient
            da2 = obj.output_weights' * dz3;
            dz2 = da2 .* (cache.z2 > 0);  % ReLU derivative
            
            % First hidden layer gradient
            da1 = obj.hidden_weights' * dz2;
            dz1 = da1 .* (cache.z1 > 0);  % ReLU derivative
            
            % Compute weight and bias gradients
            dW3 = dz3 * cache.a2';
            db3 = sum(dz3, 2);
            dW2 = dz2 * cache.a1';
            db2 = sum(dz2, 2);
            dW1 = dz1 * cache.x_norm';
            db1 = sum(dz1, 2);
            
            % Clip gradients
            clip_threshold = 1.0;
            dW3 = min(max(dW3, -clip_threshold), clip_threshold);
            db3 = min(max(db3, -clip_threshold), clip_threshold);
            dW2 = min(max(dW2, -clip_threshold), clip_threshold);
            db2 = min(max(db2, -clip_threshold), clip_threshold);
            dW1 = min(max(dW1, -clip_threshold), clip_threshold);
            db1 = min(max(db1, -clip_threshold), clip_threshold);
            
            % Update using Adam optimizer
            % Output layer
            [obj.output_weights, obj.m_output_w, obj.v_output_w] = ...
                obj.adam_update(obj.output_weights, obj.m_output_w, obj.v_output_w, dW3);
            [obj.output_biases, obj.m_output_b, obj.v_output_b] = ...
                obj.adam_update(obj.output_biases, obj.m_output_b, obj.v_output_b, db3);
            
            % Hidden layer
            [obj.hidden_weights, obj.m_hidden_w, obj.v_hidden_w] = ...
                obj.adam_update(obj.hidden_weights, obj.m_hidden_w, obj.v_hidden_w, dW2);
            [obj.hidden_biases, obj.m_hidden_b, obj.v_hidden_b] = ...
                obj.adam_update(obj.hidden_biases, obj.m_hidden_b, obj.v_hidden_b, db2);
            
            % Input layer
            [obj.input_weights, obj.m_input_w, obj.v_input_w] = ...
                obj.adam_update(obj.input_weights, obj.m_input_w, obj.v_input_w, dW1);
            [obj.input_biases, obj.m_input_b, obj.v_input_b] = ...
                obj.adam_update(obj.input_biases, obj.m_input_b, obj.v_input_b, db1);
        end
        
        function [x_pred, y_pred] = predict(obj, x)
            % Normalize input
            x_mean = mean(x, 2);
            x_std = std(x, 0, 2) + eps;
            x_norm = (x - x_mean) ./ x_std;
            
            % Forward pass through all layers
            a1 = max(0, obj.input_weights * x_norm + obj.input_biases);
            a2 = max(0, obj.hidden_weights * a1 + obj.hidden_biases);
            output = obj.output_weights * a2 + obj.output_biases;
            
            x_pred = output(1, :);
            y_pred = output(2, :);
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