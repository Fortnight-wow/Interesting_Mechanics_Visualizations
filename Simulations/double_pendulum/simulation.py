import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 9.81
L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0


def derivatives(t, state):
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1

    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) ** 2
    den2 = (L2 / L1) * den1

    dtheta1 = omega1
    dtheta2 = omega2

    domega1 = ((M2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
                + M2 * G * np.sin(theta2) * np.cos(delta)
                + M2 * L2 * omega2**2 * np.sin(delta)
                - (M1 + M2) * G * np.sin(theta1)) / den1)

    domega2 = ((-M2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
                + (M1 + M2) * (G * np.sin(theta1) * np.cos(delta)
                - L1 * omega1**2 * np.sin(delta)
                - G * np.sin(theta2))) / den2)

    return [dtheta1, domega1, dtheta2, domega2]

state0 = [np.pi / 2, 0, np.pi / 2 + 0.01, 0]
t = np.linspace(0, 25, 3000)
sol = solve_ivp(derivatives, [0, 25], state0, t_eval=t)

theta1 = sol.y[0]
theta2 = sol.y[2]

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)

x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.set_title('Double Pendulum Animation')

line, = ax.plot([], [], lw=2)
trail, = ax.plot([], [])


def update(frame):
    line.set_data([0, x1[frame], x2[frame]],
                  [0, y1[frame], y2[frame]])

    start = max(0, frame - 150)
    trail.set_data(x2[start:frame], y2[start:frame])

    return line, trail

ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
