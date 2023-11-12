function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

J = ((-(sum(y.*log(sigmoid(X*theta)))+sum((1-y).*log(1-sigmoid(X*theta)))))/m)+((lambda/(2*m))*(sum(theta(2:end).^2))) ;
grad = ((X'*(sigmoid(X*theta)-y))/m)+[0;(lambda/m)*theta(2:end)];
grad = grad(:); % Transpose

end
