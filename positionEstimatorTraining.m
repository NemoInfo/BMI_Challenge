function [modelParameters] = positionEstimatorTraining(train)
% Kalman modeling
% we need to compute A, H, Q and R
bin = 300;

progress = waitbar(100, "Preprocessing training data...");
for i = 1:numel(train)
  train(i).spikes  = train(i).spikes(:, 302:end-100);
  train(i).handPos = train(i).handPos(1:2, 301:end-100);
  train(i).handPos = handPosToHandPosVel(train(i).handPos);
  assert(size(train(i).spikes, 2) == size(train(i).handPos, 2))
  waitbar(i/numel(train), progress)
end
close(progress);

counts = zeros(8, 900);
for d = 1:size(train, 2)
  for i = 1:size(train, 1)
    counts(d,1:size(train(i,d).spikes,2)) = counts(d,1:size(train(i,d).spikes,2)) + 1; 
  end
end

min_data = 30;

T = find(all(((counts >= min_data) == ones(8,1))).',1,'last');
D = size(train, 2);
N = size(train(1).spikes, 1);
P = size(train(1).handPos, 1);

X = zeros(D, P, T);
Y = zeros(D, N, T);
progress = waitbar(100, "Computing peri-stimulus spike rates...");
for t=1:T
  for i=1:size(train,1)
    for d=1:size(train,2)
      if size(train(i,d).spikes, 2) < t, continue, end

      X(d,:,t) = X(d,:,t) + train(i,d).handPos(:, t)';
      Y(d,:,t) = Y(d,:,t) + train(i,d).spikes(:, t)';
    end
  end

  X(:,:,t) = X(:,:,t) ./ counts(:,t);
  Y(:,:,t) = Y(:,:,t) ./ counts(:,t);
  waitbar(t/T, progress)
end
close(progress);

Xp = X(:,:,2:end);
Xm = X(:,:,1:end-1);

Xp  = reshape(permute(Xp,  [2, 1, 3]),  P, D*(T-1));
Xm = reshape(permute(Xm, [2, 1, 3]),  P, D*(T-1));

A = (Xp  * Xm') / (Xm  * Xm');
E = Xp - A * Xm;
Q = (E * E') / (D * (T-1));

X  = reshape(permute(X,  [2, 1, 3]),  P, D*T);
Y  = reshape(permute(Y,  [2, 1, 3]),  N, D*T);

H = (Y*X')/(X*X');
E = Y - H * X;
R = (E * E') / (D*T);

modelParameters.A = A;
modelParameters.bin = bin;
modelParameters.Q = Q;
modelParameters.H = H;
modelParameters.R = R;
end

function [state] = handPosToHandPosVel(pos)
  vel = pos(:,2:end) - pos(:,1:end-1);
  state = [pos(:,2:end); vel];
end
