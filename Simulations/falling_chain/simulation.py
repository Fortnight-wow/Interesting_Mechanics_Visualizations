import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 30
TABLE_LINKS = 14
M = 0.1
LINK = 0.1
K = 600.0
C = 8.0
G = 9.81


def spring_force(pos_i, pos_j, vel_i, vel_j):
    delta = pos_j - pos_i
    dist = np.linalg.norm(delta)
    stretch = dist - LINK
    direction = delta / dist
    rel_speed = np.dot(vel_j - vel_i, direction)
    return (K * stretch + C * rel_speed) * direction


def derivatives(t, state):
    pos = state[: 2 * (N - 1)].reshape(N - 1, 2)
    vel = state[2 * (N - 1) :].reshape(N - 1, 2)
    full_pos = np.vstack([[0.0, 0.0], pos])
    full_vel = np.vstack([[0.0, 0.0], vel])
    acc = np.zeros((N, 2))
    acc[:, 1] = -G

    for i in range(N - 1):
        force = spring_force(full_pos[i], full_pos[i + 1], full_vel[i], full_vel[i + 1])
        acc[i] += force / M
        acc[i + 1] -= force / M

    return np.concatenate([vel.ravel(), acc[1:].ravel()])


positions = np.zeros((N - 1, 2))
for i in range(1, TABLE_LINKS + 1):
    positions[i - 1] = [-i * LINK, 0.0]
for i in range(TABLE_LINKS + 1, N):
    positions[i - 1] = [0.0, -(i - TABLE_LINKS) * LINK]

state0 = np.concatenate([positions.ravel(), np.zeros(2 * (N - 1))])
t = np.linspace(0, 2.5, 1250)
sol = solve_ivp(derivatives, [0, 2.5], state0, t_eval=t, max_step=0.002)

free_coords = sol.y[: 2 * (N - 1)].reshape(N - 1, 2, -1)
xs = np.vstack([np.zeros(len(t)), free_coords[:, 0, :]])
ys = np.vstack([np.zeros(len(t)), free_coords[:, 1, :]])

TRAIL_LENGTH = 80

fig, ax = plt.subplots(figsize=(6, 8))
margin = 0.4
ax.set_xlim(xs.min() - margin, xs.max() + margin)
ax.set_ylim(ys.min() - margin, 0.5)
ax.set_aspect('equal')
ax.set_title('Falling Chain')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axhline(0, color='black', lw=2)

chain_line, = ax.plot([], [], lw=2, color='steelblue')
tip_trail, = ax.plot([], [], lw=1, color='crimson', alpha=0.6)


def update(frame):
    chain_line.set_data(xs[:, frame], ys[:, frame])
    start = max(0, frame - TRAIL_LENGTH)
    tip_trail.set_data(xs[-1, start:frame], ys[-1, start:frame])
    return chain_line, tip_trail


ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
