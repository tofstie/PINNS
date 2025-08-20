---
layout: page
title: Damped Pendulum
mathjax: true
---

## Problem Overview
The linear advection equation is a common PDE of interest due to its simplicity. 
It has the following form, with a variable of interest, $u$, and a wave speed, $\mathbf{a}$.
$$\begin{split}\frac{du}{dt} + \mathbf{a}\nabla \cdot u = 0 \\
u(\mathbf{x},0) = u_{o}(\mathbf{x})
\end{split}$$

The analytical solution of the equation is known using the method of lines. This makes this an easy problem to calculate the accuracy of the method.
This analytical solution is 
found to be the following.
$$ u = u_{o}(\mathbf{x} + \mathbf{a}t)$$
## PINN Setup
In the PINN presented, the following variables are the inputs and outputs:

### Inputs: 
- Time, $t$
- Location on x axis, $x$
- If Dim= 2: Location on y axis, $y$

### Output:
- Variable of Interest, $u$

### Assumptions:
- No forcing function
- Periodic boundary conditions along $x = 1$ and $y=1$

### Loss Function
The loss function in this PINN is split into three parts. The first is the standard MSE of the output of the model.
Secondly, the residual is included in the loss function. The residual of the PINN  is calculated using the models predicted
angle, $u_{NN}$.

$$\frac{du_{NN}}{dt} + \mathbf{a}\nabla \cdot u_{NN} = R$$

The derivatives are calculated using automatic differentiation (AD). However, when using AD with the scaled inputs the derivative
calculated is the derivative with respect to the scaled time.

$$\frac{u_{NN}}{d\tau}$$

The scaling function used in this algorithm is the sklearn StandardScalar.

$$\tau = \frac{t-\mu}{\sigma}$$

Thus, by applying the chain rule, the derivative with respect to time is the following.

$$\frac{du_{NN}}{dt} = \frac{1}{\sigma}\frac{du_{NN}}{d\tau}$$


The loss from the residual is then calculated by the MSE of the residual.
The last part of the loss function is the initial condition restriction. In it, the loss is calculated from the predicted
initial conditions and the actual initial conditions. Since this is a second order ODE, two initial conditions are required.
The MSE of each of the initial conditions loss is then added together to get the loss condition.
The total loss is then the following:

$$Loss = \Delta\theta_{MSE} + R_{MSE}+\Delta u(\mathbf{x},0)_{MSE}$$