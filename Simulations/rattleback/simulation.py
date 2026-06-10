import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

I1 = 1.0
I2 = 1.15
I3 = 1.3
DAMP_POS = 0.4
DAMP_NEG = 0.05
WOBBLE_DAMP = 0.05


def spin_damping(omega):
    return DAMP_POS if omega > 0 else DAMP_NEG


def derivatives(t, state):
    w1, w2, w3 = state
    dw1 = ((I2 - I3) / I1) * w2 * w3 - WOBBLE_DAMP * w1
    dw2 = ((I3 - I1) / I2) * w3 * w1 - WOBBLE_DAMP * w2
    dw3 = ((I1 - I2) / I3) * w1 * w2 - spin_damping(w3) * w3
    return [dw1, dw2, dw3]


state0 = [2.0, 2.0, 8.0]
t = np.linspace(0, 15, 3000)
sol = solve_ivp(derivatives, [0, 15], state0, t_eval=t, max_step=0.005)

w3 = sol.y[2]
angle = np.zeros(len(t))
for i in range(1, len(t)):
    angle[i] = angle[i - 1] + w3[i] * (t[i] - t[i - 1])

body = np.array([
    [-0.35, -0.12],
    [0.45, -0.08],
    [0.55, 0.0],
    [0.35, 0.12],
    [-0.45, 0.08],
])

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_title('Rattleback (Celt Spin Reversal)')
ax.set_xlabel('x')
ax.set_ylabel('y')

patch = Polygon(body, closed=True, color='saddlebrown', ec='black')
ax.add_patch(patch)
spin_text = ax.text(-1.05, 1.0, '', fontsize=10)


def update(frame):
    rotated = Affine2D().rotate(angle[frame]).translate(0, 0) + ax.transData
    patch.set_transform(rotated)
    spin_text.set_text(f'ω₃ = {w3[frame]:.2f} rad/s')
    return patch, spin_text


ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
