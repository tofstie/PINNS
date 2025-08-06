---
layout: page
title: Damped Pendulum
mathjax: true
---

## Problem Overview
The damped pendulum is a classic non-linear PDE that is often taught to undergraduates. It descirbes the moition of 
pendulum with angle $\theta$, as it oscillates due to gravity. Since we are discussing a damped pendulum, there is a 
damping factor, $k$, that can be thought of as air resistance or friction. The equation takes the following form.

$$ml^2\frac{d^2\theta}{dt^2}+kl^{2}\frac{d\theta}{dt}+mglsin(\theta) = 0$$

Where $m$ is the mass of the pendulum, $l$ is the length of the pendulum, $g$ is gravity.
## PINN Setup
In the PINN presented, the following variables are the inputs and outputs:

### Inputs: 
- Time, $t$
- Damping factor, $k$

### Output:
- Angle, $\theta$