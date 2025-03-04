% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

function RMSE = testFunction_for_students_MTb()
clc;
trial = load("monkeydata_training.mat");
trial = trial.trial;

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% Select training and testing data (you can choose to split your data in a different way if you wish)
split = 50;
trainingData = trial(ix(1:split),:);
testData = trial(ix(split+1:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

figure
hold on
axis square
grid
colors = [
    1 0 0;       % Red
    0 1 0;       % Green
    0 0 1;       % Blue
    1 1 0;       % Yellow
    1 0 1;       % Magenta
    0 1 1;       % Cyan
    0.5 0 0.5;   % Purple
    1 0.5 0      % Orange
];


for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.0001);
    for direc=randperm(8) 
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), Color=colors(direc,:), LineWidth=1, LineStyle="--");
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times), Color=colors(direc,:), LineWidth=1)
    end
end

RMSE = sqrt(meanSqError/n_predictions)

end
