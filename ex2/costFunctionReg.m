function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
sum = 0;
for i = 1:m,
	cry = X(i, :) * theta;
	sum = sum + (-1 * y(i) * log(sigmoid(cry))) - (1 - y(i)) * log(1 - sigmoid(cry));
end;
sum = sum/m;
sum2 = 0;
for i = 2:n,
	sum2 = sum2 + (theta(i)^2);
end
sum2 = sum2 * (lambda/(2*m));
J = sum + sum2;

mul = X * theta;
q = sigmoid(mul);
q = q - y;
grad(1) = X(:, 1)' * q;
grad(1) = grad(1)/m;
for i = 2:n,
	grad(i) = (1/m) * X(:, i)' * q + (lambda/m)*theta(i);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
