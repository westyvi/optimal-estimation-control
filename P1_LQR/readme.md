# Project 1: Linear Quadratic Regulator (LQR)

## Overview
Simulates spacecraft optimal guidance for a rendesvous operation using continuous and discrete LQR controllers (finite/infinite horizon) and the Clohessy-Wiltshire linearized equations for orbital proximity operations. Implements and compares multiple LQR designs (labeled as cases) with varying state and control cost matrices.

## Results
![Z-time_Trajectory](<plots/z_time_finite horizon discrete LQR.png>)

*Z trajectory for finite horizon continuous LQR: the controller drives the chaser spacecraft to the origin, centered at the target spacecraft. Higher state costs/lower control costs result in more control input and faster state convergence.*

## More Results
The full set of plots is already available in the `plots/` folder. You can also navigate to this directory and run `P1.py` to regenerate them if desired.

## Commentary 
Firstly, the LQR guidance/control performance is very similar between all finite/infinite horizon continous/discrete designs. The largest difference is for the continous time, finite-horizon LQR, which shows a zero control input at the first time step followed by a large negative control. In contrast, all three other design architectures show a large negative control input at the first time step. This results in a slightly slower response for the continous time finite-horizon LQR, but is most likely an artifact of how the LQR was implemented for this case than the continuous-time, finite-horizon LQR architecture itself. Nonetheless, this LQR design still yields an effective and valid solution to the problem. All three other architectures are nearly indiscernible in control and state histories between architectures. 

In all four architectures, the performance is impacted predictably by relative scaling between the state cost matrix Q and the control cost matrix R. As R is weighted more heavily (case 1 -> case 3), the LQR commands smaller accelerations and the state takes a longer time to settle to the zero value. Conversely, smaller R weightings result in very fast response times in driving the states to zero, but very large control input. It can also be seen that, for this problem, penalizing the control input has a very large (an order of magnitude) effect on the maximum control commands while it has a much smaller effect on the settling time of the states (doubling for case 1 -> case 2, tripling for case 2 -> case 3). It follows that if control is expensive for this application (i.e. if the spacecraft has limited fuel) or if the vehicle has low thrust and cannot produce the large accelerations commanded by case 1, reducing the control effort by increasing R will have a proportionally small detriment on the state error. 
