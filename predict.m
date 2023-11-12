function p = predict(all_theta, X)

m = size(X, 1);
X = [ones(m, 1) X]; % Appending ones for bias term

[~,p] = max(X*all_theta',[],2); % Row-wise maximum

end
