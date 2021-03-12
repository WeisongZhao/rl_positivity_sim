Inspired by Andrew, I open a GitHub page to have an open discussion on the deconvolution topic, all opinions are my own and may not be right. Please just leave a message on Twitter/GitHub-issue if you are interested. 

## Inverse problem

First, I intend to talk about a little background of the inverse problem and deconvolution.

To me, the deconvolution is actually a classical machine learning method but not an optics method, that estimate the hidden parameter (real signal) from the measured parameter (camera image). 


<h4 align="center">Sparse-SIM
Ax = y. It is an inverse problem if without noise.<br><br>
Ax + n = y. It is an ill-posed inverse problem if with noise</h4>


What we're doing is trying to estimate the maximum possible x from the observed data y. If the A is a gaussian function, point source, and in noise-free condition, so the x and y are one–to–one correspondence.

In the machine learning or convex optimization field, 

We always start by asking ourselves, is this problem convex, is there a global optimal value. If there is no noise, then x and y can be one–to–one correspondence. No matter what method we use, we just need to find Ax that is absolutely equal to y. 

There are many solutions, including the Bayesian-based Richardson-Lucy deconvolution, which will be discussed below. If the computing power is sufficient, even particle swarm (PSO) or genetic algorithm (GA) are effective choices. We can define the x is the parameters to be optimized for GA/PSO, and the optimization will stop when find x for Ax – y = 0.

## Frequentism and Bayesianism for the Richardson-Lucy (RL) deconvolution

Lucy's RL article is the very first successful try on machine learning applying to the optical imaging. However, the logic used in the Lucy's article is very flexible, so it can be a little misleading in many cases. What I intend to do in this section is to give a small insight that RL is a branch of classical machine learning.

The history of RL is very interesting. In fact, for centuries, the Frequentism and the Bayesianism have been at odds, and that seems to echo the debate that we have today about the applications of the machine learning on the optical imaging.

To summarize the differences: Frequentism considers probabilities to be related to frequencies of real or hypothetical events. Bayesianism considers probabilities to measure degrees of knowledge. Frequentist analyses generally proceed through use of point estimates and **maximum likelihood** approaches. That talk about the RL to show that difference in specific.
