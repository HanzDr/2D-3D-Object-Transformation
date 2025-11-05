import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- transforms ----------
def rot_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def scale3d(sx, sy, sz):
    return np.diag([sx, sy, sz])

# ---------- geometry ----------
BASE = np.array([
    [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
    [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]
], dtype=float)
EDGES = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]

# state
verts = BASE.copy()
ax_deg = ay_deg = az_deg = 0.0
sx = sy = sz = 1.0
tx = ty = tz = 0.0

def apply_transform(P):
    rx, ry, rz = np.deg2rad(ax_deg), np.deg2rad(ay_deg), np.deg2rad(az_deg)
    M = rot_x(rx) @ rot_y(ry) @ rot_z(rz) @ scale3d(sx, sy, sz)
    return (P @ M.T) + np.array([tx, ty, tz])

# ---------- figure ----------
plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.22)

# draw initial
T = apply_transform(verts)
lines = []
for a,b in EDGES:
    ln, = ax.plot(*zip(T[a], T[b]), lw=2)
    lines.append(ln)

ax.set_box_aspect([1,1,1])
ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_zlim(-4,4)
ax.set_title("3D: rotate/scale/translate with sliders • Drag mouse to orbit/zoom")

# ---------- sliders ----------
def add_slider(y, label, vmin, vmax, vinit):
    axsl = plt.axes([0.12, y, 0.76, 0.03])
    return Slider(axsl, label, vmin, vmax, valinit=vinit)

s_rx = add_slider(0.17, "rotX (°)", -180, 180, 0)
s_ry = add_slider(0.13, "rotY (°)", -180, 180, 0)
s_rz = add_slider(0.09, "rotZ (°)", -180, 180, 0)

s_sx = add_slider(0.05, "scaleX", 0.2, 3.0, 1.0)
s_sy = add_slider(0.01, "scaleY", 0.2, 3.0, 1.0)


def on_change(_):
    global ax_deg, ay_deg, az_deg, sx, sy, sz, tx, ty, tz
    ax_deg, ay_deg, az_deg = s_rx.val, s_ry.val, s_rz.val
    sx, sy = s_sx.val, s_sy.val
    # sz = s_sz.val if you enabled it
    T = apply_transform(verts)
    for (a,b), ln in zip(EDGES, lines):
        ln.set_data_3d(*zip(T[a], T[b]))
    fig.canvas.draw_idle()

for s in (s_rx, s_ry, s_rz, s_sx, s_sy):
    s.on_changed(on_change)

drag_idx = None

def project_points(P):

    xy = []
    for p in P:
        x2, y2, _ = ax.proj_transform(p[0], p[1], p[2], ax.get_proj())
        xy.append([x2, y2])
    return np.array(xy)

def nearest_vertex(event, pts2d, tol=10):
    # event.x, event.y are display (pixel) coords
    d = np.hypot(pts2d[:,0] - event.x, pts2d[:,1] - event.y)
    i = np.argmin(d)
    return int(i) if d[i] < tol else None

def on_press(event):
    global drag_idx
    if event.inaxes != ax: return
    pts2d = project_points(apply_transform(verts))
    drag_idx = nearest_vertex(event, pts2d)

def on_release(event):
    global drag_idx
    drag_idx = None

def on_move(event):
    if drag_idx is None or event.inaxes != ax: return
    # Move the vertex in view plane: approximate by changing X/Y in camera space.
    # For simplicity, update the *original* verts in world X/Y only:
    dx = (ax.get_xlim()[1]-ax.get_xlim()[0]) * (event.step if hasattr(event,'step') else 0)  # fallback noop
    # Better approach: recompute from event.x/event.y delta projected back; here we keep it simple:
    # Use data coords directly when mouse moves:
    if event.xdata is None or event.ydata is None: return
    # Snap Z to its original value to avoid depth wobble
    z0 = verts[drag_idx, 2]
    verts[drag_idx] = [event.xdata, event.ydata, z0]
    on_change(None)

# Comment these three lines if you don't want dragging:
cid1 = fig.canvas.mpl_connect("button_press_event", on_press)
cid2 = fig.canvas.mpl_connect("button_release_event", on_release)
cid3 = fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.show()
