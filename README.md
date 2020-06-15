 Neural PID controller (WIP)
 ===============================
This project contains my progress in implementing knowlegde obtained during the Andrews Ng machine learning 
course to design neural network emulating PID controller embedded in my [stewart platform](https://github.com/Kompan15/Stewart-Platform-Ball-Ballancer/blob/master/ReadMe.md "Ball balancer") :

<p align="center">

<img src="https://i.imgur.com/8plFr77.gif" width="50%" height="50%">

</p>

Embedded PID handling the control:

<p align="center">
<img src="https://github.com/Kompan15/NeuralControl/blob/master/Pictures/PID.svg" alt="alt text" width="70%" height="70%">
</p>


I wanted it to be a finished project before publishing, but opportunity presented itself a bit earlier.
Having wired to PID output, I collected as much data as i could while platform was working and exported it to octave for further processing.
Neural network to implement:

<p align="center">
<img src="https://github.com/Kompan15/NeuralControl/blob/master/Pictures/Neural.svg" alt="alt text" width="70%" height="70%">
</p>

And the plan is to make it like that:

<p align="center">
<img src="https://github.com/Kompan15/NeuralControl/blob/master/Pictures/NeuralPid.svg" alt="alt text" width="70%" height="70%">
</p>

Three scripts to handle this so far:
1.Backpropagation algorithm (Cost function minimalization)
2.Forward Propagation Algorithm (Neural network taking data and weights matrix as input)
3.Optimizer (Basically loop within a loop trying out diffrent hidden layers sizes + lambdas looking for smallest cost function value)

In my case it means approximately 9987 networks going straight to :toilet:, 13 pretenders and one winner:
You can clearly see that increasing amount of data fed into the neural network is simultaneously decreasing the penalty (cost function):

<p align="center">
 
<img src="https://media.giphy.com/media/Q60hDsd4RiWXPdjzeK/giphy.gif" width="50%" height="50%">

<img src="https://media.giphy.com/media/XDKqws1GpZ8TVbCWDr/giphy.gif" width="50%" height="50%">

<img src="https://media.giphy.com/media/WodPDRvlgnHkT5m1Hl/giphy.gif" width="50%" height="50%">

</p>

This seems (for me) like an manifestation of a famous quote:

> It's not who has the best algorithm that wins,It's who has the most data          

