 Neural PID controller (WIP)
 ===============================
This project contains my progress in implementing knowledge obtained during the Andrews Ng machine learning 
course to design neural network emulating PID controller embedded in my [Stewart Platform](https://github.com/Kompan15/Stewart-Platform-Ball-Ballancer/blob/master/ReadMe.md "Ball balancer") :

<p align="center">

<img src="https://i.imgur.com/8plFr77.gif" width="50%" height="50%">

</p>

Embedded PID handling the control:

<p align="center">
<img src="https://github.com/Kompan15/NeuralControl/blob/master/Pictures/PID.svg" alt="alt text" width="70%" height="70%">
</p>


I wanted it to be a complete project before publishing, but the opportunity presented itself a bit earlier.
Having wired to PID output, I collected as much data as I could while the platform was working and exported it to octave for further processing.
Neural network to implement:

<p align="center">
<img src="https://github.com/Kompan15/NeuralControl/blob/master/Pictures/Neural.svg" alt="alt text" width="60%" height="60%">
</p>

And the plan is to make it like that:

<p align="center">
<img src="https://github.com/Kompan15/NeuralControl/blob/master/Pictures/NeuralPid.svg" alt="alt text" width="70%" height="70%">
</p>

Three scripts to handle this so far:

1. Forward Propagation Algorithm (Neural network taking data and weights matrix as input)

```Matlab
for t = 1:size(X, 1);
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
  ...
```

2. Backpropagation algorithm (Cost function minimalization)

```Matlab
...
  delta4 = a4 - y(t);
  delta3 = Theta3(:,2:end)'*delta4.*sigmoidGradient(z3);
  delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2);
  DELTA1 += delta2*a1';
  DELTA2 += delta3*a2';
  DELTA2 += delta4*a3';
endfor
```

3. Optimizer (Basically loop within a loop trying out different hidden layers sizes + lambdas looking for smallest cost function value)
```Matlab
for n = 1:length(lambda_vec)
  for i = 1:length(layers_vec)
    [M Jb Theta1 Theta2 Theta3] = easy_nn(A,B,y',layers_vec(i),lambda_vec(n));
    if Jb<Jmin
      Jmin = Jb;
      layermin = layers_vec(i);
      lambdamin = lambda_vec(n);
      Theta1min = Theta1;
      Theta2min = Theta2;
      Theta3min = Theta3;
      easy(A,B,y,layermin,lambdamin, Theta1min, Theta2min, Theta3min, count);
      count++
    endif
  endfor
  endfor
end
```

In my case it means approximately 9987 networks going straight to :toilet:, 13 pretenders and one winner:
You can clearly see that increasing amount of data fed into the neural network is simultaneously decreasing the penalty (cost function):

<p align="center">
 
<img src="https://media.giphy.com/media/Q60hDsd4RiWXPdjzeK/giphy.gif" width="30%" height="30%">

<img src="https://media.giphy.com/media/XDKqws1GpZ8TVbCWDr/giphy.gif" width="30%" height="30%">

<img src="https://media.giphy.com/media/WodPDRvlgnHkT5m1Hl/giphy.gif" width="30%" height="30%">

</p>

This seems (for me) like an manifestation of a famous quote:

>         It's not who has the best algorithm that wins, It's who has the most data     

to be continued!

