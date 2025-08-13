---
layout: page
title: Damped Pendulum
mathjax: true
---

## Problem Overview
The damped pendulum is a classic non-linear ODE that is often taught to undergraduates. It descirbes the moition of 
pendulum with angle $\theta$, as it oscillates due to gravity. Since we are discussing a damped pendulum, there is a 
damping factor, $k$, that can be thought of as air resistance or friction. The equation takes the following form.

$$ml^2\frac{d^2\theta}{dt^2}+kl^{2}\frac{d\theta}{dt}+mglsin(\theta) = 0$$

Where $m$ is the mass of the pendulum, $l$ is the length of the pendulum, $g$ is gravity.
## PINN Setup
In the PINN presented, the following variables are the inputs and outputs:

### Inputs: 
- Time, $t$
- Damping factor, $\frac{k}{m}$

### Output:
- Angle, $\theta$

### Assumptions:
- $\frac{g}{l} = 1$
- No Forcing

### Loss Function
The loss function in this PINN is split into three parts. The first is the standard MSE of the output of the model.
Secondly, the residual is included in the loss function. The residual of the PINN  is calculated using the models predicted
angle, $\theta_{NN}$.

$$\frac{d^2\theta_{NN}}{dt^2}+\frac{k}{m}\frac{d\theta_{NN}}{dt}+\frac{g}{l}sin(\theta) = R$$

The derivatives are calculated using automatic differentiation (AD). However, when using AD with the scaled inputs the derivative
calculated is the derivative with respect to the scaled time.

$$\frac{\theta_{NN}}{d\tau}$$

The scaling function used in this algorithm is the sklearn StandardScalar.

$$\tau = \frac{t-\mu}{\sigma}$$

Thus, by applying the chain rule, the derivative with respect to time is the following.

$$\frac{d\theta_{NN}}{dt} = \frac{1}{\sigma}\frac{d\theta_{NN}}{d\tau}$$

$$\frac{d^2\theta_{NN}}{dt^2} = \frac{1}{\sigma^2}\frac{d^2\theta_{NN}}{d\tau^2}$$

The loss from the residual is then calculated by the MSE of the residual.
The last part of the loss function is the initial condition restriction. In it, the loss is calculated from the predicted
initial conditions and the actual initial conditions. Since this is a second order ODE, two initial conditions are required.
The MSE of each of the initial conditions loss is then added together to get the loss condition.
The total loss is then the following:

$$Loss = \Delta\theta_{MSE} + R_{MSE}+\Delta\theta(0,\frac{k}{m})_{MSE}+\Delta\frac{d\theta}{dt}(0,\frac{k}{m})_{MSE}$$