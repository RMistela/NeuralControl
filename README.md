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
I took embedded pid output, collected as much data as i could while platform was working and exported it to octave for further processing.
There i wrote three scripts doing the job for me:

1.Backpropagation algorithm (Cost function minimalization)
2.Forward Propagation Algorithm (Neural network taking data and weights matrix as input)
3.optimizer (Basically loop within a loop trying out diffrent hidden layers sizes + lambdas used to compute regularization terms)


<p align="center">

<img src="https://media.giphy.com/media/Q60hDsd4RiWXPdjzeK/giphy.gif" width="40%" height="40%">
<img src="https://media.giphy.com/media/XDKqws1GpZ8TVbCWDr/giphy.gif" width="40%" height="40%">
<img src="https://media.giphy.com/media/WodPDRvlgnHkT5m1Hl/giphy.gif" width="40%" height="40%">

</p>
