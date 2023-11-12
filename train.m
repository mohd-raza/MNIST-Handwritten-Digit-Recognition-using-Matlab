function [all_theta] = train(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for i = 1:num_labels
  fprintf('Training Digit: %d',i-1);
  [all_theta(i,:)] = fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)),initial_theta, options);
end

end
