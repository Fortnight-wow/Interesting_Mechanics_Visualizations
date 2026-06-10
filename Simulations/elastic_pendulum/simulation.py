import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 9.81
M = 1.0
L0 = 1.0
K = 40.0


def derivatives(t, state):
    length, theta, length_dot, theta_dot = state
    spring = -K * (length - L0)
    radial = spring + M * G * np.cos(theta) + M * length * theta_dot**2
    length_acc = radial / M
    angular_acc = -(2 * length_dot * theta_dot + G * np.sin(theta)) / length
    return [length_dot, theta_dot, length_acc, angular_acc]


state0 = [L0 + 0.35, np.pi / 3, 0.0, 0.4]
t = np.linspace(0, 30, 3000)
sol = solve_ivp(derivatives, [0, 30], state0, t_eval=t, max_step=0.02)

length = sol.y[0]
theta = sol.y[1]
bob_x = length * np.sin(theta)
bob_y = -length * np.cos(theta)

TRAIL_LENGTH = 150

fig, ax = plt.subplots(figsize=(6, 6))
margin = 0.5
ax.set_xlim(bob_x.min() - margin, bob_x.max() + margin)
ax.set_ylim(bob_y.min() - margin, 0.5)
ax.set_aspect('equal')
ax.set_title('Elastic Pendulum')
ax.set_xlabel('x')
ax.set_ylabel('y')

spring_line, = ax.plot([], [], lw=2, color='steelblue')
bob, = ax.plot([], [], 'o', color='crimson', markersize=10)
trail, = ax.plot([], [], lw=1, color='crimson', alpha=0.6)


def update(frame):
    spring_line.set_data([0, bob_x[frame]], [0, bob_y[frame]])
    bob.set_data([bob_x[frame]], [bob_y[frame]])
    start = max(0, frame - TRAIL_LENGTH)
    trail.set_data(bob_x[start:frame], bob_y[start:frame])
    return spring_line, bob, trail


ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
