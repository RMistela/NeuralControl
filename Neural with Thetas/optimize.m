function[layermin lambdamin Theta1min Theta2min Theta3min] = optimizer(A,B,y)

Jmin = inf;
layermin = 0;
lambdamin = 0;
lambda_vec = [0:0.1:100]';
layers_vec = [1:1:100];
for n = 1:length(lambda_vec)
  for i = 1:length(layers_vec)
    [M J Theta1 Theta2 Theta3] = easy_nn(A,B,y',layers_vec(n),lambda_vec(i));
    if J<Jmin
      Jmin = J
      layermin = layers_vec(n)
      lambdamin = lambda_vec(i)
      Theta1min = Theta1;
      Theta2min = Theta2;
      Theta3min = Theta3;
    endif
  endfor
  endfor

end