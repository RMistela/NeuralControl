function[layermin lambdamin Theta1min Theta2min Theta3min] = optimize(A,B,y)

figu = 1;
Jmin = inf;
layermin = 0;
lambdamin = 0;
lambda_vec = [0.1:0.1:10]';
layers_vec = [5:1:100];
for n = 1:length(lambda_vec)
  for i = 1:length(layers_vec)
    [M J Theta1 Theta2 Theta3] = easy_nn(A,B,y',layers_vec(i),lambda_vec(n));
    if J<Jmin
      Jmin = J
      layermin = layers_vec(n)
      lambdamin = lambda_vec(i)
      Theta1min = Theta1;
      Theta2min = Theta2;
      Theta3min = Theta3;
      easy(A,B,y,layermin,lambdamin, Theta1min, Theta2min, Theta3min);
    endif
  endfor
  endfor

end