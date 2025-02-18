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


data = loadTrainingData();

% print data
disp(data);

% print data keys
disp(fieldnames(data));

% print data keys
disp(data.trial);

%% PLOTTING HAND POSITION
% The plot shows that data across 100 trials is star shaped. 
% MAYBE 300 ms
for i = 1:5
    for k = 1:8
        scatter(data.trial(i,k).handPos(1,1:500), data.trial(i,k).handPos(2,1:500))
        hold on
    end
end

%% PLOTTING TRAIN SPIKES
% plot the spike trains in .spikes
for i = 1
    for k = 1:8
        plot(data.trial(i,k).spikes(1,:))
        hold on
    end
end


%% TRIAL 1: LINEAR REGRESSION
trial = data.trial(1,1);

% get the handPos
handX = trial.handPos(1,300:end);
handY = trial.handPos(2,300:end);
Y = [handX ;handY];
Y = Y.';

%declare time steps
time = size(Y,1);

% get the spikes
spikes = trial.spikes.';
X = spikes(300:end,:);

% initialize weights and biases
W = rand(2,98);
B = rand(2,1);
eta = 0.01;

%initialize weights and biases
for epoch = 1:100
    for t = 1:time
        Yhat = W*X(t)+B;
        W = W + eta*(2/time*X(t)*(Y(t).'-Yhat))
    end
end





