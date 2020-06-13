function[out Jb Theta1_grad Theta2_grad Theta3_grad] = easy_nn(A,B,y,hidden_layer_size,lambda)

%Input: X Y, y.

%normalize
yr = y;
A = (A - min(A))/(max(A)-min(A));
B = (B - min(B))/(max(B)-min(B));
y = (y - min(y))/(max(y)-min(y));
X = [A B];

input_layer_size = columns(X);
num_labels = 1;
m = size(X, 1);

%random Thetas initialization.

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
Theta3 = randInitializeWeights(hidden_layer_size, num_labels);   
         
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

Xb = X; %I create Xb as a backpropagation training set.
X = [ones(m,1) X];

%Gradient
DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));          
DELTA3 = zeros(size(Theta3));        
             
                
Th1rg = Theta1;
Th1rg(:,1) = 0;
Th2rg = Theta2;
Th2rg(:,1) = 0;
Th3rg = Theta3;
Th3rg(:,1) = 0;

%BackProp
for t = 1:m
  a1 = Xb(t,:)';
  a1 = [1;a1];
  
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [ones(1,size(a2,2)) ; a2];
  
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  a3 = [ones(1,size(a3,2)) ; a3];
  
  z4 = Theta3*a3;
  a4 = sigmoid(z4);
  
  delta4 = a4 - y(t);
  delta3 = Theta3(:,2:end)'*delta4.*sigmoidGradient(z3);
  delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2);
  DELTA1 += delta2*a1';
  DELTA2 += delta3*a2';
  DELTA2 += delta4*a3';
endfor

Theta1_grad = (1/m) * DELTA1 + (lambda/m) * Th1rg;
Theta2_grad = (1/m) * DELTA2 + (lambda/m) * Th2rg;
Theta3_grad = (1/m) * DELTA3 + (lambda/m) * Th3rg;
grad = [Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)];

%Forward Prop
a2 = sigmoid(Theta1_grad * X');
a2 = [ones(1,size(a2,2)) ; a2];
a3 = sigmoid(Theta2_grad * a2);
a3 = [ones(1,size(a3,2)) ; a3];
a4 = sigmoid(Theta3_grad * a3);
[k e] = size(a4);
[ip p] = max(a4', [], 2);
h = out = a4;


%Normalize to output
hr = min(yr) + ((h - min(h))*(max(yr)-min(yr)))/(max(h)-min(h));

%subplot (3, 1, 1)
%plot (yr);
%title('Original output')  
%subplot (3, 1, 2)
%plot (hr);
%title('Output')  
%subplot (3, 1, 3)
%plot (X);
%title('Input')  


%Reg = lambda*(sum(grad(:).*grad(:)))/(2*m);
%Cost Functions
%Jb = -1/m * sum(sum((y.*log(h)) + (1-y).*log(1-h))) + Reg;
SqrErrors = (hr-yr).^2;
Jb = 1/(2*m)*sum(SqrErrors);


end