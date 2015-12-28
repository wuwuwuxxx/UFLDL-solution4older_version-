function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, rawdata)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

[~, n] = size(rawdata);
data = rawdata;
%forward and save z&a for back propagation
Z1 = W1* data + repmat(b1, 1, n);
A1 = sigmoid(Z1);
Z2 = W2 * A1 + repmat(b2, 1, n);
A2 = sigmoid(Z2);
%sparse parameter
p_mean = sum(A1, 2)/n;
%cost
cost = cost_fun(rawdata - A2);
sparse = sparsityParam*log(sparsityParam./p_mean)+(1-sparsityParam)*log((1-sparsityParam)./(1-p_mean));
cost = cost / n + lambda/2*(norm(W1, 'fro')^2 + norm(W2, 'fro')^2) + beta * sum(sparse);
%back propagation
delta3 = -(rawdata - A2) .* sig_prime(Z2);
p = -sparsityParam./p_mean + (1-sparsityParam)./(1-p_mean);
delta2 = (W2'*delta3 + beta * repmat(p, [1 n])) .* sig_prime(Z1);
%update W and b
delta_w2 = delta3 * A1';
delta_b2 = sum(delta3, 2);
delta_w1 = delta2 * rawdata';
delta_b1 = sum(delta2, 2);

W1grad = (delta_w1) / n + lambda * W1;
W2grad = (delta_w2) / n + lambda * W2;
b1grad = (delta_b1) / n;
b2grad = (delta_b2) / n;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1.0 ./ (1.0 + exp(-x));
end
%prime of the sigmoid function
function prime = sig_prime(x)

    prime = sigmoid(x).*(1-sigmoid(x));
end

function c = cost_fun(x)

    c  = norm(x, 'fro')^2/2;
end
