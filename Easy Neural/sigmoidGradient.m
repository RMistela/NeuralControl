function g = sigmoidGradient(z)

g = zeros(size(z));
sigmz = 1.0 ./ (1.0 + exp(-z));
g = sigmz.*(1-sigmz);

end
