import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- transforms ---
def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]])

def scale2d(sx, sy):
    return np.array([[sx, 0],[0, sy]])

# --- initial polygon (square) ---
base = np.array([[-1, -1],
                 [ 1, -1],
                 [ 1,  1],
                 [-1,  1]], dtype=float)

# state
dragging_idx = None
theta = 0.0
sx, sy = 1.0, 1.0

def apply_transform(P):
    M = rot2d(theta) @ scale2d(sx, sy)
    return P @ M.T

# --- figure & artists ---
plt.close("all")
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

transformed = apply_transform(base)

orig_line,   = ax.plot(*base.T, "-o", lw=1.5, alpha=0.5, label="Original")
trans_line,  = ax.plot(*transformed.T, "-o", lw=2.0, label="Transformed")

ax.set_aspect("equal", adjustable="box")
ax.grid(True, ls=":", alpha=0.5)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_title("Drag the original vertices • Use sliders to rotate/scale")
ax.legend(loc="upper left")

# --- sliders ---
ax_theta = plt.axes([0.15, 0.12, 0.7, 0.03])
ax_sx    = plt.axes([0.15, 0.08, 0.7, 0.03])
ax_sy    = plt.axes([0.15, 0.04, 0.7, 0.03])

s_theta = Slider(ax_theta, 'Angle (°)', -180.0, 180.0, valinit=0.0)
s_sx    = Slider(ax_sx,    'Scale X',    0.1,   3.0,   valinit=1.0)
s_sy    = Slider(ax_sy,    'Scale Y',    0.1,   3.0,   valinit=1.0)

def update_from_sliders(_):
    global theta, sx, sy
    theta = np.deg2rad(s_theta.val)
    sx = s_sx.val
    sy = s_sy.val
    T = apply_transform(base)
    trans_line.set_data(*T.T)
    fig.canvas.draw_idle()

s_theta.on_changed(update_from_sliders)
s_sx.on_changed(update_from_sliders)
s_sy.on_changed(update_from_sliders)

# --- hit-testing helpers for dragging ---
def nearest_vertex(event, pts, tol=0.15):
    if event.xdata is None or event.ydata is None:
        return None
    d = np.hypot(pts[:,0] - event.xdata, pts[:,1] - event.ydata)
    i = np.argmin(d)
    return int(i) if d[i] < tol else None

def on_press(event):
    global dragging_idx
    if event.inaxes != ax: return
    dragging_idx = nearest_vertex(event, base)

def on_release(event):
    global dragging_idx
    dragging_idx = None

def on_move(event):
    if dragging_idx is None: return
    if event.inaxes != ax or event.xdata is None or event.ydata is None: return
    # update the dragged vertex (original polygon)
    base[dragging_idx] = [event.xdata, event.ydata]
    orig_line.set_data(*base.T)
    # update transformed polygon
    T = apply_transform(base)
    trans_line.set_data(*T.T)
    fig.canvas.draw_idle()

cid1 = fig.canvas.mpl_connect('button_press_event', on_press)
cid2 = fig.canvas.mpl_connect('button_release_event', on_release)
cid3 = fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()
