input_layer_size  = 400;  % 20x20 Input Images
num_labels = 10;          % 10 labels

fprintf('Loading and Visualizing Data ...\n')
load('MNIST-small.mat'); % MNIST Dataset
m = size(X, 1);

rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

% Training
fprintf('\nTraining One-vs-All Logistic Regression...\n')
lambda = 0.1;
[all_theta] = train(X, y, num_labels, lambda);

pred = predict(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% Prediction
fprintf('\nPredicting On Random Image...\n')
rp = randi(m);
X = X(rp,:);
displayData(X);
pred = predict(all_theta, X);
pred=mod(pred, 10);
fprintf('\nPrediction: %f\n', pred);
