# Pick-and-Place Simulations — PyBullet

Two standalone PyBullet GUI simulations of a robotic arm picking and stacking six coloured cubes on a wooden table. Both are fully self-contained — no training or model files required.

| File | Arm | IK | Dependencies |
|---|---|---|---|
| `4DOFpickandplace.py` | Custom 4-joint arm built from primitive shapes | Analytical (pure Python geometry) | `pybullet`, `numpy` |
| `7DOFpickandplace.py` | Real Kuka iiwa URDF (same model as `main1.py`) | PyBullet built-in solver | `pybullet`, `numpy` |

---

## How to Run

```bash
# 4-joint primitive arm
python 4DOFpickandplace.py

# Kuka iiwa 7-DOF URDF arm
python 7DOFpickandplace.py
```

A PyBullet GUI window opens immediately. No model files are needed.

Optional — gripper camera frame saving:

```bash
pip install imageio
```

---

## Scene (shared by both files)

- Wooden table, 0.75 m height, four legs
- Six cubes on the table surface:

| Colour | Start position (x, y) |
|---|---|
| Red    | (−0.25, 0.20) |
| Blue   | ( 0.10, 0.20) |
| Green  | (−0.10, 0.00) |
| Yellow | ( 0.25, 0.00) |
| Purple | (−0.25,−0.20) |
| Orange | ( 0.10,−0.20) |

- Arm base: (−0.42, 0.30) on the table surface
- Stack target: (0.48, 0.32) — cubes stacked vertically

---

## `4DOFpickandplace.py` — Custom 4-Joint Arm

### Arm Design

The arm is built entirely from PyBullet primitive shapes (cylinders and boxes) — no URDF. It has **4 revolute joints**:

| Joint | Description | Range |
|---|---|---|
| J0 | Base yaw — rotates entire arm in horizontal plane | −π to π |
| J1 | Shoulder pitch | 0.1 to 0.9π |
| J2 | Elbow pitch | 0 to 0.85π |
| J3 | Wrist pitch — auto-levelled to keep EE vertical | −π/2 to π/2 |

Link lengths: pedestal 0.06 m · upper arm 0.22 m · forearm 0.18 m · hand 0.10 m

### Inverse Kinematics

Solved analytically using the 2-link geometric method (law of cosines). J3 is derived automatically to keep the end-effector pointing downward. No external IK library needed.

### GUI Sliders

| Slider | Function |
|---|---|
| J0 Base Yaw | Manual joint control |
| J1 Shoulder Pitch | Manual joint control |
| J2 Elbow Pitch | Manual joint control |
| J3 Wrist Pitch | Manual joint control |
| AUTO PICK & PLACE | Set to 1 to start automated sequence |
| Speed | Animation speed (0.5 – 5.0×) |
| CAPTURE CAMERA | Set to 1 to save gripper camera frames |

---

## `7DOFpickandplace.py` — Kuka iiwa 7-DOF URDF Arm

### Arm Design

Loads `kuka_iiwa/model.urdf` from `pybullet_data` — the exact same robot model used in `main1.py` via `custom_env.py`. The URDF gives a realistic Kuka iiwa appearance with proper link geometry and joint limits.

A pair of visual orange finger boxes is added manually (the plain Kuka URDF has no built-in gripper).

### Joints

| Joint | Description | Limits |
|---|---|---|
| J0 | Base rotation | −0.967 to 0.967 rad |
| J1 | Shoulder pitch | −2.0 to 2.0 rad |
| J2 | Upper-arm roll | −2.96 to 2.96 rad |
| J3 | Elbow pitch | 0.19 to 2.29 rad |
| J4 | Forearm roll | −2.96 to 2.96 rad |
| J5 | Wrist pitch | −2.09 to 2.09 rad |
| J6 | Wrist roll | −3.05 to 3.05 rad |

Joint limits match `pybullet_envs/bullet/kuka.py`.

### Inverse Kinematics

Uses `p.calculateInverseKinematics` with:
- Target orientation: EE pointing straight down (`[0, π, 0]` Euler)
- Rest pose: `[0, 0, 0, π/2, 0, −π/2, 0]`
- 200 iterations, residual threshold 0.005 m

### GUI Sliders

| Slider | Function |
|---|---|
| J0 – J6 | Manual joint control (7 sliders, one per joint) |
| AUTO PICK & PLACE | Set to 1 to start automated sequence |
| Speed | Animation speed (0.5 – 5.0×) |
| CAPTURE CAMERA | Set to 1 to save gripper camera frames |

---

## Auto Pick-and-Place State Machine (both files)

```
idle → hover_pick → descend → grasp → lift → carry → place_down → release → retreat → idle
```

Each cube is processed in order. When all 6 are stacked, the arm resets all cubes to their original positions and repeats.

### State Descriptions

| State | What happens |
|---|---|
| `idle` | Compute IK target to hover above next cube |
| `hover_pick` | Interpolate joints toward hover position |
| `descend` | Slowly lower to cube height |
| `grasp` | Close gripper, attach cube to EE |
| `lift` | Raise arm to lift height (TABLE + 0.35 m) |
| `carry` | Move horizontally to hover above stack |
| `place_down` | Slowly lower to stack position |
| `release` | Snap cube to exact stack position, open gripper |
| `retreat` | Return to lift height before next cube |

---

## Gripper Camera (both files)

When **CAPTURE CAMERA** slider = 1, a virtual 64 × 64 RGB + depth camera at the gripper tip activates each frame.

- Images saved to `camera_frames/session_<timestamp>/` every 10 frames (requires `imageio`)
- Depth linearised and stored in metres
- A `README.md` summary written on exit

Camera: FOV 60° · near 0.01 m · far 2.0 m · resolution 64 × 64

---

## Shared Source Files

| File | Role |
|---|---|
| `4DOFpickandplace.py` | 4-joint primitive arm simulation |
| `7DOFpickandplace.py` | Kuka iiwa 7-DOF URDF simulation |
| `camera_visualizer.py` | Optional frame saver used by both |

---

## Closing the Simulation

Close the PyBullet window or press **Ctrl-C**. Both scripts catch the disconnect and exit cleanly.

