import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 9.81
L = 0.5
A = 0.04
OMEGA = 90.0


def derivatives(t, state):
    theta, theta_dot = state
    theta_acc = -(G / L) * np.sin(theta) + (A * OMEGA**2 * np.cos(OMEGA * t) / L) * np.sin(theta)
    return [theta_dot, theta_acc]


state0 = [np.pi - 0.15, 0.0]
t = np.linspace(0, 15, 2500)
sol = solve_ivp(derivatives, [0, 15], state0, t_eval=t, max_step=0.002)

theta = sol.y[0]
pivot_y = A * np.cos(OMEGA * t)
bob_x = L * np.sin(theta)
bob_y = pivot_y + L * np.cos(theta)

TRAIL_LENGTH = 150

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.9, 0.6)
ax.set_aspect('equal')
ax.set_title('Kapitza Pendulum (Dynamically Stabilized Inverted)')
ax.set_xlabel('x')
ax.set_ylabel('y')

pivot, = ax.plot([], [], 's', color='dimgray', markersize=8)
rod, = ax.plot([], [], lw=2, color='steelblue')
bob, = ax.plot([], [], 'o', color='crimson', markersize=10)
trail, = ax.plot([], [], lw=1, color='crimson', alpha=0.6)


def update(frame):
    pivot.set_data([0], [pivot_y[frame]])
    rod.set_data([0, bob_x[frame]], [pivot_y[frame], bob_y[frame]])
    bob.set_data([bob_x[frame]], [bob_y[frame]])
    start = max(0, frame - TRAIL_LENGTH)
    trail.set_data(bob_x[start:frame], bob_y[start:frame])
    return pivot, rod, bob, trail


ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
