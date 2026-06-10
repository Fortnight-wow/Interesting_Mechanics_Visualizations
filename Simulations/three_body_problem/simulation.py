import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 1.0
MASSES = np.array([1.0, 1.0, 1.0])

# Figure-eight orbit initial conditions (Chenciner & Montgomery)
POS0 = np.array([
    [-0.97000436, 0.24308753],
    [0.0, 0.0],
    [0.97000436, -0.24308753],
])
VEL0 = np.array([
    [0.4662036850, 0.4323657300],
    [-0.93240737, -0.86473146],
    [0.4662036850, 0.4323657300],
])


def three_body(t, state):
    positions = state[:6].reshape(3, 2)
    velocities = state[6:].reshape(3, 2)
    accelerations = np.zeros((3, 2))

    for i in range(3):
        for j in range(3):
            if i != j:
                r = positions[j] - positions[i]
                dist = np.linalg.norm(r)
                accelerations[i] += G * MASSES[j] * r / dist**3

    return np.concatenate([velocities.ravel(), accelerations.ravel()])


state0 = np.concatenate([POS0.ravel(), VEL0.ravel()])
t = np.linspace(0, 25, 3000)
sol = solve_ivp(three_body, [0, 25], state0, t_eval=t)

xs = sol.y[[0, 2, 4], :]
ys = sol.y[[1, 3, 5], :]

COLORS = ['crimson', 'steelblue', 'goldenrod']
TRAIL_LENGTH = 150

fig, ax = plt.subplots(figsize=(8, 8))
margin = 0.3
ax.set_xlim(xs.min() - margin, xs.max() + margin)
ax.set_ylim(ys.min() - margin, ys.max() + margin)
ax.set_aspect('equal')
ax.set_title('Three Body Problem')
ax.set_xlabel('x')
ax.set_ylabel('y')

bodies = []
trails = []
for i in range(3):
    body, = ax.plot([], [], 'o', color=COLORS[i], markersize=10)
    trail, = ax.plot([], [], lw=1, color=COLORS[i], alpha=0.7)
    bodies.append(body)
    trails.append(trail)


def update(frame):
    for i in range(3):
        bodies[i].set_data([xs[i, frame]], [ys[i, frame]])
        start = max(0, frame - TRAIL_LENGTH)
        trails[i].set_data(xs[i, start:frame], ys[i, start:frame])
    return bodies + trails


ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
