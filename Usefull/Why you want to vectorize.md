Why vectorize?
==============

Previously i was iterating through entire dataset one by one incrementing deltas at the end of the loop.

```Matlab
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
```
I was satisfied for a while, but waiting for the algorithm to go through thousands of neural networks drove me mad:
following output can be obtained by using tic toc commands:
```Matlab
tic();
  %Do something
toc();

Elapsed time is 1.1198 seconds.
Elapsed time is 1.03619 seconds.
Elapsed time is 1.01856 seconds.
Elapsed time is 1.04025 seconds.
Elapsed time is 1.06346 seconds.
Elapsed time is 1.11821 seconds.
```
Quick math:
10 000 examples *  1s +/- 0.5s to compute gives me roughly 2hrs and 46 min (painfully slow)

NOW
===

Vectorization is a technique allowing you to use highly optimized vector calculus to carry iteration for you.

```Matlab
  a1 = [ones(m,1) X]';
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [ones(1,size(a2,2)) ; a2];
  
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  a3 = [ones(1,size(a3,2)) ; a3];
  
  z4 = Theta3*a3;
  a4 = sigmoid(z4);
  
  delta4 = a4 .- y;
  delta3 = Theta3(:,2:end)'*delta4.*sigmoidGradient(z3);
  delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2);
  DELTA1 = delta2*a1';
  DELTA2 = delta3*a2';
  DELTA2 = delta4*a3';
```
Now the time required to burst through all parameters takes aproximately 1075 sec, which is 10 times faster than a loop.
