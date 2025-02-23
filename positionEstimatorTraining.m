function [model] = positionEstimatorTraining(train)
% RNN model
bin = 300;
model.bin = 300;

progress = waitbar(100, "Computing spike rates...");
for i = 1:numel(train)
  train(i).spikes  = train(i).spikes(:, 301-bin:end);
  train(i).handPos = train(i).handPos(1:2,  299:end);
  train(i).handState = handPosToHandState(train(i).handPos);
  train(i).spikes  = spikeTrainToSpikeRates(train(i).spikes, bin);
  assert(size(train(i).spikes, 2) == size(train(i).handState, 2))
  waitbar(i/numel(train), progress)
end
close(progress)

N = size(train(1).spikes, 1);
S = size(train(1).handState, 1);
H = 100;
eta = 0.01;

model.Wi = randn(H, N+S) * sqrt(2/(N+S));
model.bi = zeros(H,1);
model.Wo = randn(S, H) * sqrt(2/H);
model.bo = zeros(S,1);

epochs = 10;
progress = waitbar(100, "Training...");

losses = zeros(epochs);
for epoch = 1:epochs
  total_loss = 0;
  loss_count = 0;
  order = randperm(numel(train));

  for it = 1:numel(train)
    i = order(it);
    X = train(i).spikes;    % (N,T)
    Y = train(i).handState; % (S,T)
    yp = Y(:,1);
    for t = 2:size(X,2)
      x = X(:,t);
      y = Y(:,t);
  
      % ---- Forward Pass ----
      z = [x; yp];
      h = model.Wi * z + model.bi;
      h = max(h,0); % ReLU
      yh= model.Wo * h + model.bo;
  
      % ----  Loss (MSE)  ----
      L = 0.5 * sum((yh - y).^2);
      total_loss = total_loss + L;
      loss_count = loss_count + 1;
  
      % ----   Backprop   ----
      dL_dyh = yh - y;
      grad_Wo = dL_dyh * h';       % (S x H)
      grad_bo = dL_dyh;            % (S x 1)
      
      dL_dh = model.Wo' * dL_dyh;   % (H x 1)
      d_relu = double(h > 0);       % (H x 1)
      dL_dh_pre = dL_dh .* d_relu;  % (H x 1)
      
      grad_Wi = dL_dh_pre * z';     % (H x (N+S))
      grad_bi = dL_dh_pre;          % (H x 1)
  
      % ------- Parameter Update -------
      model.Wo = model.Wo - eta * clip(grad_Wo,-1,1);
      model.bo = model.bo - eta * clip(grad_bo,-1,1);
      model.Wi = model.Wi - eta * clip(grad_Wi,-1,1);
      model.bi = model.bi - eta * clip(grad_bi,-1,1);
      
      yp = yh;
    end
    waitbar(((epoch-1)*numel(train)+it)/(epochs*numel(train)), progress, ...
      sprintf('Training... Epoch %d | Sample %d', epoch, it));
  end

  losses(epoch) = total_loss / loss_count;
  display(losses(epoch));
end
plot(losses);
close(progress);
end

function [rates] = spikeTrainToSpikeRates(train, bin)
  kernel = ones(1, bin) / bin;  % averaging kernel
  rates = conv2(train, kernel, 'valid'); 
end

function [state] = handPosToHandState(pos)
  vel = pos(:,2:end) - pos(:,1:end-1);
  state = [pos(:,2:end); vel];
end
