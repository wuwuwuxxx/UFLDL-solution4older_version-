function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
%softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end


%cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
layers = numel(stack);
%forward
z{1} = stack{1}.w * data + repmat(stack{1}.b, 1, M);
a{1} = sigmoid(z{1});
for j = 2:layers;
    z{j} = stack{j}.w * a{j-1} + repmat(stack{j}.b, 1, M);
    a{j} = sigmoid(z{j});
end
%forward(get output layer)
zn = (softmaxTheta * a{layers});
p = bsxfun(@rdivide, exp(zn), sum(exp(zn)));
h = log(p) .* groundTruth;
%get cost
cost = -mean(sum(h)) + (lambda / 2) * norm(softmaxTheta, 'fro')^2;
%back propagation
delta{layers+1} = -(groundTruth - p);
delta{layers} = (softmaxTheta' * delta{layers+1}) .* sig_prime(z{layers});
for j = layers:-1:2;
    delta{j-1} = ((stack{j}.w)' * delta{j}) .* sig_prime(z{j-1});
end
%get deltaW and deltaB
softmaxThetaGrad = (delta{layers+1} * a{layers}') / M + lambda*softmaxTheta;
for j = layers:-1:2;
    stackgrad{j}.w = (delta{j} * a{j-1}') / M;
    stackgrad{j}.b = mean(delta{j}, 2);
end
stackgrad{1}.w = delta{1} * data' / M;
stackgrad{1}.b = mean(delta{1}, 2);
% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function prime = sig_prime(x)

    prime = sigmoid(x).*(1-sigmoid(x));
end
