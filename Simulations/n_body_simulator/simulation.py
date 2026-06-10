import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 1.0
N = 10
SOFTENING = 0.08
MASSES = np.full(N, 0.2)
MASSES[0] = 50.0

positions = np.zeros((N, 2))
velocities = np.zeros((N, 2))
positions[0] = [0.0, 0.0]
velocities[0] = [0.0, 0.0]

for i in range(1, N):
    angle = 2 * np.pi * (i - 1) / (N - 1)
    radius = 2.0 + 0.5 * (i - 1)
    positions[i] = [radius * np.cos(angle), radius * np.sin(angle)]
    speed = np.sqrt(G * MASSES[0] / radius) * 0.98
    velocities[i] = [-speed * np.sin(angle), speed * np.cos(angle)]

TRAIL_LENGTH = 100


def n_body(t, state):
    pos = state[: 2 * N].reshape(N, 2)
    vel = state[2 * N :].reshape(N, 2)
    acc = np.zeros((N, 2))

    for i in range(N):
        for j in range(N):
            if i != j:
                r = pos[j] - pos[i]
                dist_sq = np.dot(r, r) + SOFTENING**2
                acc[i] += G * MASSES[j] * r / dist_sq**1.5

    return np.concatenate([vel.ravel(), acc.ravel()])


state0 = np.concatenate([positions.ravel(), velocities.ravel()])
t = np.linspace(0, 25, 3000)
sol = solve_ivp(n_body, [0, 25], state0, t_eval=t, rtol=1e-8, atol=1e-10)

trajectories = sol.y[: 2 * N].reshape(N, 2, -1)
xs = trajectories[:, 0, :]
ys = trajectories[:, 1, :]

fig, ax = plt.subplots(figsize=(8, 8))
margin = 0.5
ax.set_xlim(xs.min() - margin, xs.max() + margin)
ax.set_ylim(ys.min() - margin, ys.max() + margin)
ax.set_aspect('equal')
ax.set_title('N-Body Simulator')
ax.set_xlabel('x')
ax.set_ylabel('y')

colors = plt.cm.tab10(np.linspace(0, 1, N))
sizes = [14 if i == 0 else 7 for i in range(N)]

bodies = []
trails = []
for i in range(N):
    body, = ax.plot([], [], 'o', color=colors[i], markersize=sizes[i])
    trail, = ax.plot([], [], lw=0.8, color=colors[i], alpha=0.6)
    bodies.append(body)
    trails.append(trail)


def update(frame):
    for i in range(N):
        bodies[i].set_data([xs[i, frame]], [ys[i, frame]])
        start = max(0, frame - TRAIL_LENGTH)
        trails[i].set_data(xs[i, start:frame], ys[i, start:frame])
    return bodies + trails


ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
