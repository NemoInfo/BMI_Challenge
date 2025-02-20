clc;
load monkeydata_training.mat;

rng(2013);
ix = randperm(length(trial));

trainingData = trial(ix(1:80),:);
testData = trial(ix(81:end),:);

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

tr = 1;
for d=randperm(8)
  decodedHandPos = [];
  times = 320:20:size(testData(tr,d).spikes, 2);
  for t = times
    test_data.spikes = testData(tr,d).spikes(:,1:t);
    test_data.startHandPos = testData(tr,d).handPos(1:2,1); 
    test_data.decodedHandPos = decodedHandPos;

    [decodedPosX, decodedPosY, newParameters] = positionEstimator(test_data, modelParameters);
    decodedPos = [decodedPosX; decodedPosY];
    decodedHandPos = [decodedHandPos decodedPos];
    modelParameters = newParameters;
  end
  plot(decodedHandPos(1,:),decodedHandPos(2,:), Color=colors(d,:), LineWidth=1.5)
  plot(testData(tr,d).handPos(1,times),testData(tr,d).handPos(2,times), Color=colors(d,:), LineWidth=1)
end