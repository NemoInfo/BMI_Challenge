classdef DataWrapper
    properties
        trials      % Cell array to store trial data
        numTrials   % Number of trials
        numTargets  % Number of targets (8 in this case)
        numChannels % Number of neural channels (98)
    end
    
    methods
        function obj = DataWrapper()
            % Constructor: Load and organize the data
            % Check if file exists
            if ~exist('monkeydata_training.mat', 'file')
                error('Could not find monkeydata_training.mat in the current directory');
            end
            
            % Load the MAT file
            raw_data = load('monkeydata_training.mat');
            
            % Verify data loaded successfully
            if isempty(fieldnames(raw_data))
                error('Failed to load data from monkeydata_training.mat');
            end
            
            % Initialize properties
            obj.numTrials = size(raw_data.trial, 1);
            obj.numTargets = size(raw_data.trial, 2);
            obj.numChannels = size(raw_data.trial(1,1).spikes, 1);
            
            % Initialize trials cell array
            obj.trials = cell(obj.numTrials, obj.numTargets);
            
            % Organize data into trials
            for i = 1:obj.numTrials
                for j = 1:obj.numTargets
                    trial_data = struct();
                    trial_data.trialId = [i, j];  % [trial_number, target_number]
                    trial_data.spikes = raw_data.trial(i,j).spikes;  % [98 channels x T]
                    trial_data.handPos = raw_data.trial(i,j).handPos;  % [2 x T]
                    obj.trials{i,j} = trial_data;
                end
            end
        end
        
        function trial_data = getTrial(obj, trial_num, target_num)
            % Get specific trial data
            % Inputs: trial_num (1 to numTrials), target_num (1 to numTargets)
            if trial_num > obj.numTrials || target_num > obj.numTargets
                error('Trial or target number out of bounds');
            end
            trial_data = obj.trials{trial_num, target_num};
        end
        
        function spikes = getSpikes(obj, trial_num, target_num)
            % Get spikes for a specific trial
            trial_data = obj.getTrial(trial_num, target_num);
            spikes = trial_data.spikes;
        end
        
        function pos = getHandPos(obj, trial_num, target_num)
            % Get hand position for a specific trial
            trial_data = obj.getTrial(trial_num, target_num);
            pos = trial_data.handPos;
        end
        
        function plotHandTrajectory(obj, trial_num, target_num)
            % Plot hand trajectory for a specific trial
            pos = obj.getHandPos(trial_num, target_num);
            figure('Name', sprintf('Hand Trajectory - Trial %d, Target %d', trial_num, target_num));
            plot(pos(1,:), pos(2,:));
            title(sprintf('Hand Trajectory - Trial %d, Target %d', trial_num, target_num));
            xlabel('X Position');
            ylabel('Y Position');
            grid on;
        end
        
        function plotSpikeRaster(obj, trial_num, target_num)
            % Plot spike raster for a specific trial
            spikes = obj.getSpikes(trial_num, target_num);
            figure('Name', sprintf('Spike Raster - Trial %d, Target %d', trial_num, target_num));
            imagesc(spikes);
            title(sprintf('Spike Raster - Trial %d, Target %d', trial_num, target_num));
            xlabel('Time (ms)');
            ylabel('Channel');
            colorbar;
        end
    end
end 