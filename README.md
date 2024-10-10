# Coupling 
This is a course project of the course « Simulation and Monte Carlo Methods » at ENSAE Paris

# The Coupled Rejection Sampler

This repository contains the codes of the experimental results presented in the paper "CouplingMC" by me and two Classmates. 
This project start from the works of Adrien Corenflos and Simo Särkkä [1]. 

Firstly, we implemented the approach proposed by Adrien Corenflos and Simo Särkkä [1] in their paper and we compared it with the simpler approach known as Thoriston's algorithm.
The experiments concerned in particular the case of Gaussians tails. 

Secondly the Thoriston's algorithm was analysed in a theoretical point of view and his link with rejction sampling methods was shown through a theoretical proof. 
In fact we considered the modified version of the Thoriston's algorithm which include a factor C very useful to control the running time variance of this algorithm. 

Finally, the behavior of the running time of the rejection coupling algorithm proposed by Adrien Corenflos and Simo Särkkä [1], was analysed empirically. We have graphically represent the distribution
of the running time and we showned through statistical test that it folloms a geometric distribution. The curse of dimensionality had also been analysed. 

### Codes

See the `coupled_rejection_sampler` folder. The code it contains is the implementation of our paper illustrations.

### References

.. [1]: Corenflos, A. and Särkkä, S., “The Coupled Rejection Sampler”, <i>arXiv e-prints</i>, 2022. (https://arxiv.org/abs/2201.09585)
