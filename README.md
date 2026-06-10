#Interesting Physics Visualizations

Honestly, I got tired of deriving endless equations on paper and just plotting a depressing, static `plt.plot(theta)` for lab reports. It’s boring, and it doesn't give you any actual intuition for how these systems behave. 

So, I’m building this repo to turn complex physics equations into real-time, frame-by-frame animations. If a script in here doesn't actively move and show the physical system, it shouldn't be here.

---

## What's Inside (and What I'm Working On)

Instead of just creating empty placeholder folders, I'm implementing these one by one using proper numerical integration (`solve_ivp`) and updating them into real animations:

* **Double Pendulum:** Done. Two rods, two masses, and a lot of chaos. It solves the ODEs and tracks a live trail behind the second bob so you can see exactly when determinism goes out the window.
* **Lorenz Attractor:** A full 3D chaotic system that traces the classic butterfly pattern over time.
* **Three-Body Problem:** Three gravitational masses locked in a toxic relationship until one inevitably gets yeeted out of the frame.
* **N-Body Simulator:** A gravity-driven free-for-all where a bunch of masses pull on each other to see if they'll collapse or accidentally form a galaxy.
* **The Pendulum Variants:** Working on the **Inverted Pendulum**, **Elastic Pendulum**, and the **Kapitza Pendulum** (where vibrating the pivot upside down somehow defies gravity).
* **Rigid Body & Mechanics Oddballs:** Restricted Three-Body, Rattleback, and a Falling Chain.

---

## Setup

Standard routine. Clone it and install the basic math/plotting libraries:

```bash
git clone [https://github.com/Fortnight-wow/Interesting_Physics_Visualizations-.git](https://github.com/Fortnight-wow/Interesting_Physics_Visualizations-.git)
cd Interesting_Physics_Visualizations-
pip install -r requirements.txt
