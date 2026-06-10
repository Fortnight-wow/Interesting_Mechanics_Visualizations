import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MU = 0.1


def potential_derivatives(x, y):
    r1_sq = (x + MU) ** 2 + y**2
    r2_sq = (x - 1 + MU) ** 2 + y**2
    r1 = np.sqrt(r1_sq)
    r2 = np.sqrt(r2_sq)
    dudx = x - (1 - MU) * (x + MU) / r1**3 - MU * (x - 1 + MU) / r2**3
    dudy = y - (1 - MU) * y / r1**3 - MU * y / r2**3
    return dudx, dudy


def crtbp(t, state):
    x, y, vx, vy = state
    dudx, dudy = potential_derivatives(x, y)
    ax = 2 * vy + dudx
    ay = -2 * vx + dudy
    return [vx, vy, ax, ay]


state0 = [0.82, 0.0, 0.0, 0.35]
t = np.linspace(0, 40, 4000)
sol = solve_ivp(crtbp, [0, 40], state0, t_eval=t, rtol=1e-9, atol=1e-11)

x = sol.y[0]
y = sol.y[1]

primary1 = (-MU, 0)
primary2 = (1 - MU, 0)

TRAIL_LENGTH = 200

fig, ax = plt.subplots(figsize=(8, 6))
margin = 0.15
ax.set_xlim(x.min() - margin, x.max() + margin)
ax.set_ylim(y.min() - margin, y.max() + margin)
ax.set_aspect('equal')
ax.set_title('Restricted Three Body Problem (Rotating Frame)')
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.plot(*primary1, 'o', color='goldenrod', markersize=14, label='Primary 1')
ax.plot(*primary2, 'o', color='steelblue', markersize=10, label='Primary 2')

probe, = ax.plot([], [], 'o', color='crimson', markersize=8)
trail, = ax.plot([], [], lw=0.8, color='crimson', alpha=0.7)


def update(frame):
    probe.set_data([x[frame]], [y[frame]])
    start = max(0, frame - TRAIL_LENGTH)
    trail.set_data(x[start:frame], y[start:frame])
    return probe, trail


ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
