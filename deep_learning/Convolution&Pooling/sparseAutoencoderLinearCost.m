function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, rawdata)
% -------------------- YOUR CODE HERE --------------------
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

[~, n] = size(rawdata);
data = rawdata;
Z1 = W1* data + repmat(b1, 1, n);
A1 = sigmoid(Z1);
Z2 = W2 * A1 + repmat(b2, 1, n);
A2 = Z2;

p_mean = sum(A1, 2)/n;

cost = cost_fun(rawdata - A2);
sparse = sparsityParam*log(sparsityParam./p_mean)+(1-sparsityParam)*log((1-sparsityParam)./(1-p_mean));
cost = cost / n + lambda/2*(norm(W1, 'fro')^2 + norm(W2, 'fro')^2) + beta * sum(sparse);

delta3 = -(rawdata - A2);
p = -sparsityParam./p_mean + (1-sparsityParam)./(1-p_mean);
delta2 = (W2'*delta3 + beta * repmat(p, [1 n])) .* sig_prime(Z1);

delta_w2 = delta3 * A1';
delta_b2 = sum(delta3, 2);
delta_w1 = delta2 * rawdata';
delta_b1 = sum(delta2, 2);

W1grad = (delta_w1) / n + lambda * W1;
W2grad = (delta_w2) / n + lambda * W2;
b1grad = (delta_b1) / n;
b2grad = (delta_b2) / n;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
% -------------------- YOUR CODE HERE --------------------                                    

end

function sigm = sigmoid(x)
  
    sigm = 1.0 ./ (1.0 + exp(-x));
end

function prime = sig_prime(x)

    prime = sigmoid(x).*(1-sigmoid(x));
end

function c = cost_fun(x)

    c  = norm(x, 'fro')^2/2;
end