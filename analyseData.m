clc;
load("monkeydata_training.mat");

disp(trial);

%% Hand position data

n_shown = fix(10 / 100 * size(trial,1));
figure;
hold on;
colors = lines(9);
colors = colors(1:end-1,:);
colororder(colors);
for i = 1:n_shown
    for k = 1:8
        scatter(trial(i,k).handPos(1,:), trial(i,k).handPos(2,:));
    end
end
hold off;

figure;
hold on;
colororder(colors);
for i = 1:n_shown
    for k = 1:8
        scatter(trial(i,k).handPos(1,1:300), trial(i,k).handPos(2,1:300));
    end
end
hold off;
% This shows us that the first 300ms are not directed movement, so training
% the model on this might not make sense

figure;
hold on;
colororder(colors);
for i = 1:n_shown
    for k = 1:8
        scatter(trial(i,k).handPos(1,end-100:end), trial(i,k).handPos(2,end-100:end));
    end
end
hold off;
% Same goes for the last 100ms are not directed movement, so training
% the model on this might not make sense

%% Spike train data

figure;
hold on;
num = size(trial(1,1).spikes, 1);

for n = 1:num
    spike_indices = find(trial(1,1).spikes(n,:) == 1);
    if size(spike_indices)
        line([spike_indices(:)'; spike_indices(:)'], [n-0.5; n+0.5], ...
            'Color', "k", 'LineWidth', 1.2);
    end
end
hold off;