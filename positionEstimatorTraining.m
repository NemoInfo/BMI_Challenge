function [modelParameters] = positionEstimatorTraining(training_data)
    
  % Hyperparameters
  fprintf("Training the FastMLP with the following hyperparameters:\n")
    window_size = 20;  % ms
    input_size = 98 * window_size;  % 98 channels * time steps (flattened)
    hidden_size = [256, 128];  % Two hidden layers
    output_size = 2;  % x,y coordinates
    learning_rate = 0.001;
    epochs = 50;
    batch_size = 128;  % Increased batch size for better vectorization
    fprintf("Window size: %d ms\n", window_size);
    fprintf("Input size: %d\n", input_size);
    fprintf("Hidden size: %s\n", mat2str(hidden_size));
    fprintf("Output size: %d\n", output_size);
    fprintf("Learning rate: %f\n", learning_rate);
    fprintf("Epochs: %d\n", epochs);
    fprintf("Batch size: %d\n", batch_size);
    
    % Create network
    layer_sizes = [input_size, hidden_size, output_size];
    net = FastMLP(layer_sizes, learning_rate);
    
    % Calculate total number of samples to pre-allocate memory
    total_samples = 0;
    for trial = 1:size(training_data, 1)
        for target = 1:size(training_data, 2)
            spikes = training_data(trial, target).spikes(:, 301:end);
            total_samples = total_samples + size(spikes, 2) - window_size + 1;
        end
    end
    
    % Pre-allocate memory
    X_train = zeros(total_samples, input_size);
    Y_train = zeros(total_samples, 2);
    
    % Process each trial and target
    sample_idx = 1;
    for trial = 1:size(training_data, 1)
        for target = 1:size(training_data, 2)
            fprintf("Processing trial %d, target %d\n", trial, target);
            % Get spike data and hand positions after 300ms
            spikes = training_data(trial, target).spikes(:, 301:end);
            pos = training_data(trial, target).handPos(1:2, 301:end);
            
            % Calculate number of windows for this trial
            num_windows = size(spikes, 2) - window_size + 1;
            
            % Create indices for all windows at once
            window_indices = zeros(window_size, num_windows);
            for i = 1:window_size
                window_indices(i, :) = i:(num_windows + i - 1);
            end
            
            % Extract all windows at once
            windows = spikes(:, window_indices);
            
            % Reshape all windows at once
            end_idx = sample_idx + num_windows - 1;
            X_train(sample_idx:end_idx, :) = reshape(windows, 98 * window_size, num_windows)';
            
            % Get corresponding hand positions
            Y_train(sample_idx:end_idx, :) = pos(:, window_size:end)';
            
            % Update sample index
            sample_idx = end_idx + 1;
        end
        
        % Display progress
        if mod(trial, 10) == 0
            fprintf('Processed %d/%d trials\n', trial, size(training_data, 1));
        end
    end
    
    % Training loop with vectorized operations
    n_samples = size(X_train, 1);
    n_batches = floor(n_samples/batch_size);
    
    fprintf('Starting training for %d epochs\n', epochs);
    for epoch = 1:epochs
        % Shuffle the training data
        shuffle_idx = randperm(n_samples);
        X_shuffled = X_train(shuffle_idx, :);
        Y_shuffled = Y_train(shuffle_idx, :);
        
        % Mini-batch training with vectorized operations
        epoch_loss = 0;
        for batch = 1:n_batches
            % Get batch indices
            start_idx = (batch-1)*batch_size + 1;
            end_idx = min(batch*batch_size, n_samples);
            
            % Extract batch
            X_batch = X_shuffled(start_idx:end_idx, :)';
            Y_batch = Y_shuffled(start_idx:end_idx, :)';
            
            % Forward pass
            [output, cache] = net.forward(X_batch);
            
            % Backward pass
            net = net.backward(X_batch, Y_batch, cache);
            
            % Calculate batch loss (MSE)
            batch_loss = net.compute_loss(output, Y_batch);
            epoch_loss = epoch_loss + batch_loss;
        end
        
        % Display epoch progress
        if mod(epoch, 5) == 0
            avg_loss = epoch_loss/n_batches;
            fprintf('Epoch %d/%d - Average Loss: %.4f\n', epoch, epochs, avg_loss);
        end
    end
    
    % Save model parameters
    modelParameters.layers = net.layers;
    modelParameters.weights = net.weights;
    modelParameters.biases = net.biases;
end


