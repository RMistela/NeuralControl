function[layermin lambdamin Theta1min Theta2min Theta3min] = optimize(A,B,y)

count = 1; %Print plot counter.
Jmin = inf 
layermin = 0;
lambdamin = 0;
lambda_vec = [0.1:0.1:10]';
layers_vec = [5:1:100];
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