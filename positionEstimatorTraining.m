function [modelParameters] = positionEstimatorTraining(training_data)
    % Clear workspace and persistent variables
    clear global;  % Clear any global variables
    
    % Hyperparameters
    fprintf("Training the FastMLP with the following hyperparameters:\n")
    window_size = 20;  % ms
    input_size = 98 * window_size;  % 98 channels * time steps (flattened)
    hidden_size = [floor(input_size/2), floor(input_size/4)];  % [192*2, 96] achieves an RMSE of 80.1286 mm
    output_size = 2;  % x,y coordinates
    epochs = 50;
    batch_size = 256;  % Larger batch size for better vectorization
    fprintf("Window size: %d ms\n", window_size);
    fprintf("Input size: %d\n", input_size);
    fprintf("Hidden size: %s\n", mat2str(hidden_size));
    fprintf("Output size: %d\n", output_size);
    fprintf("Epochs: %d\n", epochs);
    fprintf("Batch size: %d\n", batch_size);
    
    % Create network (learning rate is fixed in FastMLP)
    layer_sizes = [input_size, hidden_size, output_size];
    net = FastMLP(layer_sizes);
    
    % Calculate total number of samples and preallocate in chunks
    fprintf('Preprocessing data...\n');
    total_samples = 0;
    chunk_size = 1000;  % Process data in chunks to reduce memory pressure
    X_chunks = {};
    Y_chunks = {};
    current_chunk_x = [];
    current_chunk_y = [];
    samples_in_chunk = 0;
    
    % Initialize counters and accumulators
    total_time = 0;
    epoch_loss = 0;
    total_batches = 0;
    
    for trial = 1:size(training_data, 1)
        for target = 1:size(training_data, 2)
            % Get spike data and hand positions after 300ms
            spikes = training_data(trial, target).spikes(:, 301:end);
            pos = training_data(trial, target).handPos(1:2, 301:end);
            
            % Process in one operation per trial
            num_windows = size(spikes, 2) - window_size + 1;
            
            % Create sliding windows efficiently
            windows = zeros(98, window_size, num_windows);
            for w = 1:num_windows
                windows(:,:,w) = spikes(:, w:w+window_size-1);
            end
            
            % Reshape windows and get positions
            X_trial = reshape(permute(windows, [2,1,3]), input_size, num_windows)';
            Y_trial = pos(:, window_size:end)';
            
            % Add to current chunk
            current_chunk_x = [current_chunk_x; X_trial];
            current_chunk_y = [current_chunk_y; Y_trial];
            samples_in_chunk = samples_in_chunk + size(X_trial, 1);
            
            % If chunk is full, store it and create new chunk
            if samples_in_chunk >= chunk_size
                X_chunks{end+1} = current_chunk_x;
                Y_chunks{end+1} = current_chunk_y;
                current_chunk_x = [];
                current_chunk_y = [];
                samples_in_chunk = 0;
            end
            
            total_samples = total_samples + num_windows;
        end
        
        % Display progress
        if mod(trial, 10) == 0
            fprintf('Processed %d/%d trials\n', trial, size(training_data, 1));
        end
    end
    
    % Add final chunk if not empty
    if ~isempty(current_chunk_x)
        X_chunks{end+1} = current_chunk_x;
        Y_chunks{end+1} = current_chunk_y;
    end
    
    % Training loop with chunk-based processing
    fprintf('Starting training for %d epochs\n', epochs);
    total_time = 0;
    for epoch = 1:epochs
        epoch_start = tic;  % Start timing the epoch
        epoch_loss = 0;
        total_batches = 0;
        
        % Process each chunk
        for chunk = 1:length(X_chunks)
            X_train = X_chunks{chunk};
            Y_train = Y_chunks{chunk};
            n_samples = size(X_train, 1);
            n_batches = floor(n_samples/batch_size);
            
            % Shuffle current chunk
            shuffle_idx = randperm(n_samples);
            X_shuffled = X_train(shuffle_idx, :);
            Y_shuffled = Y_train(shuffle_idx, :);
            
            % Mini-batch training
            for batch = 1:n_batches
                % Get batch indices
                start_idx = (batch-1)*batch_size + 1;
                end_idx = min(batch*batch_size, n_samples);
                
                % Extract batch
                X_batch = X_shuffled(start_idx:end_idx, :)';
                Y_batch = Y_shuffled(start_idx:end_idx, :)';
                
                % Forward and backward pass
                [output, cache] = net.forward(X_batch);
                net = net.backward(X_batch, Y_batch, cache);
                
                % Calculate batch loss (with safety check)
                batch_loss = net.compute_loss(output, Y_batch);
                if ~isnan(batch_loss) && ~isinf(batch_loss)  % Only add valid loss values
                    epoch_loss = epoch_loss + batch_loss;
                    total_batches = total_batches + 1;
                end
            end
        end
        
        % Calculate epoch time and update total
        epoch_time = toc(epoch_start);
        total_time = total_time + epoch_time;
        
        % Calculate average loss (with safety check)
        if total_batches > 0
            avg_loss = epoch_loss/total_batches;
        else
            avg_loss = Inf;
            warning('No valid batches in epoch %d', epoch);
        end
        
        % Calculate estimated time remaining
        avg_time_per_epoch = total_time/epoch;
        remaining_epochs = epochs - epoch;
        estimated_time_remaining = avg_time_per_epoch * remaining_epochs;
        
        % Convert estimated time to hours:minutes:seconds
        hours_remaining = floor(estimated_time_remaining/3600);
        minutes_remaining = floor((estimated_time_remaining - hours_remaining*3600)/60);
        seconds_remaining = floor(estimated_time_remaining - hours_remaining*3600 - minutes_remaining*60);
        
        % Display progress with time estimates
        fprintf('Epoch %d/%d - Loss: %.4f - Time: %.2fs - ETA: %02d:%02d:%02d\n', ...
            epoch, epochs, avg_loss, epoch_time, ...
            hours_remaining, minutes_remaining, seconds_remaining);
    end
    
    % Save model parameters
    modelParameters.layers = [input_size, hidden_size, output_size];  % Store layer sizes
    modelParameters.weights = net.weights;  % Already a cell array of weights
    modelParameters.biases = net.biases;   % Already a cell array of biases
end


