function [training_data] = loadTrainingData()
    % Load training data from monkeydata_training.mat file
    % Returns: training_data structure containing the loaded data
    
    % Check if file exists
    if ~exist('monkeydata_training.mat', 'file')
        error('Could not find monkeydata_training.mat in the current directory');
    end
    
    % Load the MAT file
    training_data = load('monkeydata_training.mat');
    
    % Verify data loaded successfully
    if isempty(fieldnames(training_data))
        error('Failed to load data from monkeydata_training.mat');
    end
end

% Create data wrapper instance
data = DataWrapper();

%% PLOTTING HAND POSITION
figure('Name', 'Hand Positions');  % Create new figure with title
% The plot shows that data across 100 trials is star shaped. 
% MAYBE 300 ms
for i = 1:5
    for k = 1:8
        trial_data = data.getTrial(i, k);
        scatter(trial_data.handPos(1,1:500), trial_data.handPos(2,1:500))
        hold on;
    end
end
title('Hand Positions Across Trials')
xlabel('X Position')
ylabel('Y Position')
hold off;

%% PLOTTING TRAIN SPIKES
figure('Name', 'Spike Trains');  % Create new figure with title
% plot the spike trains in .spikes
for i = 1
    for k = 1:8
        subplot(8,1,k)  % Create 8 subplots, one for each trial
        trial_data = data.getTrial(i, k);
        plot(trial_data.spikes(1,:))
        title(['Spike Train for Trial ' num2str(k)])
        ylabel('Spike')
    end
end

%% TRIAL 1: LINEAR REGRESSION
trial_data = data.getTrial(1,1);

% get the handPos
handX = trial_data.handPos(1,300:end);
handY = trial_data.handPos(2,300:end);
Y = [handX ;handY];
Y = Y.';
Yhat = zeros(size(Y));

%declare time steps
time = size(Y,1);

% get the spikes
spikes = trial_data.spikes.';
X = spikes(300:end,:);

% initialize weights and biases
W = rand(2,98);
B = rand(2,1);
eta = 0.01;
epochs = 100;
MSE = zeros(epochs,1);

% Training loop
for epoch = 1:epochs
    for t = 1:time
        % Forward pass
        Yhat(t,:) = (W*X(t,:).' + B).';
        
        % Update weights and bias
        error = (Y(t,:) - Yhat(t,:)).';
        W = W + eta * (error * X(t,:));
        B = B + eta * error;
        
        % Calculate MSE
        MSE(epoch) = MSE(epoch) + sum((Y(t,:) - Yhat(t,:)).^2) / time;
    end
end

% plot MSE
figure('Name', 'Training MSE');  % Create new figure with title
plot(MSE);
title('Mean Squared Error Over Epochs')
xlabel('Epoch')
ylabel('MSE')






