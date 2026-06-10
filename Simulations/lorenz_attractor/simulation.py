import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz(t, state):
    x, y, z = state
    return [
        SIGMA * (y - x),
        RHO * x - y - x * z,
        x * y - BETA * z,
    ]


state0 = [1.0, 1.0, 1.0]
t = np.linspace(0, 40, 4000)
sol = solve_ivp(lorenz, [0, 40], state0, t_eval=t)

x, y, z = sol.y

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

margin = 2.0
ax.set_xlim(x.min() - margin, x.max() + margin)
ax.set_ylim(y.min() - margin, y.max() + margin)
ax.set_zlim(z.min() - margin, z.max() + margin)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')

point, = ax.plot([], [], [], 'o', color='crimson', markersize=6)
trail, = ax.plot([], [], [], lw=0.7, color='steelblue', alpha=0.85)

TRAIL_LENGTH = 200


def update(frame):
    point.set_data([x[frame]], [y[frame]])
    point.set_3d_properties([z[frame]])

    start = max(0, frame - TRAIL_LENGTH)
    trail.set_data(x[start:frame], y[start:frame])
    trail.set_3d_properties(z[start:frame])

    ax.view_init(elev=25, azim=frame * 0.15)

    return point, trail


ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=False)
plt.show()
