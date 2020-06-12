function[out Jo Theta1_grad Theta2_grad] = easy_nn(A,B,y,hidden_layer_size,lambda)

%Input: X Y, y.

%normalize
yr = y';
A = (A - min(A))/(max(A)-min(A));
B = (B - min(B))/(max(B)-min(B));
y = (y - min(y))/(max(y)-min(y));

X = [A B];

input_layer_size = columns(X);
num_labels = 1;
m = size(X, 1);
% You need to return the following variables correctly 

%normalization:

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

Theta1 = reshape(initial_nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(initial_nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Xb = X; %I create Xb as a backpropagation training set.
X = [ones(m,1) X];     
DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));                 
%Remove bias terms which are first columns of theta matrices.
%UnTheta1 = Theta1(:,2:end); %Theta1(all row elements, from 2nd column to the ending one)
%UnTheta2 = Theta2(:,2:end);
%thetas = [UnTheta1(:) ; UnTheta2(:)]; %unroll thetas with removed bias terms.
%Regularization term
%Reg = lambda*(sum(thetas(:).*thetas(:)))/(2*m);
%Cost Functions

%Jf = -1/m * sum(sum((y.*log(h)) + (1-y).*log(1-h))) + Reg;              
                
Th1rg = Theta1;
Th1rg(:,1) = 0;
Th2rg = Theta2;
Th2rg(:,1) = 0;

for t = 1:m
  a1 = Xb(t,:)';
  a1 = [1;a1];
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [ones(1,size(a2,2)) ; a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  %using logical arrays.
  delta3 = a3 - y(t)';
  delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2);
  DELTA1 += delta2*a1';
  DELTA2 += delta3*a2';
endfor
Theta1_grad = (1/m) * DELTA1 + (lambda/m) * Th1rg;
Theta2_grad = (1/m) * DELTA2 + (lambda/m) * Th2rg;
grad = [Theta1_grad(:) ; Theta2_grad(:)];

%Regularization term

a2 = sigmoid(Theta1_grad *  X');
a2 = [ones(1,size(a2,2)) ; a2];
a3 = sigmoid(Theta2_grad * a2);
[k e] = size(a3);
[ip p] = max(a3', [], 2);
h = out = a3;

%hr = out = min(yr) + ((h - min(h))*(max(yr)-min(yr)))/(max(h)-min(h));

plot(y)
hold
plot(h)
%subplot (3, 1, 1)
%plot (yr);
%title('Original output')  
%subplot (3, 1, 2)
%plot (out);
%title('Output')  
%subplot (3, 1, 3)
%plot (X);
%title('Input')  

Reg = lambda*(sum(grad(:).*grad(:)))/(2*m);
%Cost Functions

Jo = -1/m * sum(sum((y.*log(h)) + (1-y).*log(1-h))) + Reg;
end