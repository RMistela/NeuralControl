function[out Jf] = easy(A,B,y,hidden_layer_size,lambda,Theta1, Theta2)

%Input: X Y, y.

%normalize
yr = y';
A = (A .- min(A))/(max(A)-min(A));
B = (B .- min(B))/(max(B)-min(B));
y = (y .- min(y))/(max(y)-min(y));

X = [A B];

input_layer_size = columns(X);
num_labels = 1;
m = size(X, 1);
% You need to return the following variables correctly                  
Xb = X; %I create Xb as a backpropagation training set.
X = [ones(m,1) X];                   
             
a2 = sigmoid(Theta1 * X');
a2 = [ones(1,size(a2,2)) ; a2];
a3 = sigmoid(Theta2 * a2);
[k e] = size(a3);
[ip p] = max(a3', [], 2);
h = out = a3;
%Remove bias terms which are first columns of theta matrices.
%UnTheta1 = Theta1(:,2:end); %Theta1(all row elements, from 2nd column to the ending one)
%UnTheta2 = Theta2(:,2:end);
thetas = [Theta1(:) ; Theta2(:)]; %unroll thetas with removed bias terms.
%re-normalize
out = h = min(yr) + ((h - min(h))*(max(yr)-min(yr)))/(max(h)-min(h));

h = h + mean(abs(h-yr));
plot(yr)
hold
plot(h)

%subplot (3, 1, 1)
%plot (yr);
%title('Original output')  
%subplot (3, 1, 2)
%plot (h);
%title('Output')  
%subplot (3, 1, 3)
%plot (X);
%title('Input') 

Reg = lambda*(sum(thetas(:).*thetas(:)))/(2*m);
%Cost Functions

Jf = -1/m * sum(sum((yr*log(h)) + (1-yr).*log(1-h))) + Reg;              

end