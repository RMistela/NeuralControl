function[layermin lambdamin Theta1 Theta2] = optimizer(A,B,y)

Jmin = inf;
layermin = 0;
lambdamin = 0;
lambda_vec = [0:0.1:10];
layers_vec = [0:1:100];
for n = 1:length(layers_vec)
  for i = 1:length(lambda_vec)
    [M J Theta1 Theta2] = easy_nn(A,B,y',layers_vec(n),lambda_vec(i));
    if J<Jmin
      J
      Jmin = J;
      layermin = layers_vec(n)
      lambdamin = lambda_vec(i)
    endif
  endfor
  endfor

end