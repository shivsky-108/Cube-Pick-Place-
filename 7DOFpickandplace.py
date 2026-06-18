"""
PyBullet — Table + Kuka iiwa 7-DOF + Cube Pick & Place
=======================================================
Run:  python pickandplace.py

The arm is the same Kuka iiwa 7-DOF model used in main1.py
(kuka_iiwa/model.urdf from pybullet_data), controlled via
PyBullet's built-in inverse-kinematics solver.

Controls (GUI sliders):
  J0-J6 sliders        -> manually pose all 7 joints
  AUTO PICK & PLACE    -> set to 1 to start automated sequence
  Speed                -> animation speed multiplier
"""

import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  CAMERA VISUALIZER (optional)
# ═══════════════════════════════════════════════════════════════
try:
    from camera_visualizer import CameraVisualizer
    CAMERA_VIZ_AVAILABLE = True
except ImportError:
    CAMERA_VIZ_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
#  CONNECT & SCENE
# ═══════════════════════════════════════════════════════════════
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(
    cameraDistance=2.2,
    cameraYaw=50,
    cameraPitch=-28,
    cameraTargetPosition=[0, 0, 0.6],
)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def make_box(half, pos, color, mass=0, friction=0.8):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    bid = p.createMultiBody(mass, col, vis, pos)
    p.changeDynamics(bid, -1, lateralFriction=friction, restitution=0.1)
    return bid

# ═══════════════════════════════════════════════════════════════
#  GRIPPER CAMERA UTILITIES
# ═══════════════════════════════════════════════════════════════
CAMERA_WIDTH  = 64
CAMERA_HEIGHT = 64
CAMERA_FOV    = 60.0
CAMERA_NEAR   = 0.01
CAMERA_FAR    = 2.0
CAMERA_OFFSET = 0.08

def capture_gripper_camera(ee_pos, j0):
    camera_pos = [
        ee_pos[0] + math.cos(j0) * CAMERA_OFFSET,
        ee_pos[1] + math.sin(j0) * CAMERA_OFFSET,
        ee_pos[2],
    ]
    camera_orn = p.getQuaternionFromEuler([math.pi / 4, -math.pi / 2.5, j0])
    target = [
        camera_pos[0] + 0.5 * math.cos(j0),
        camera_pos[1] + 0.5 * math.sin(j0),
        camera_pos[2] - 0.3,
    ]
    view_matrix = p.computeViewMatrix(camera_pos, target, [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(
        CAMERA_FOV, CAMERA_WIDTH / CAMERA_HEIGHT, CAMERA_NEAR, CAMERA_FAR
    )
    _, _, rgb, depth, _ = p.getCameraImage(
        CAMERA_WIDTH, CAMERA_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb_arr   = np.array(rgb, dtype=np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))[:, :, :3]
    depth_arr = np.array(depth, dtype=np.float32).reshape((CAMERA_HEIGHT, CAMERA_WIDTH))
    depth_m   = CAMERA_FAR * CAMERA_NEAR / (CAMERA_FAR - (CAMERA_FAR - CAMERA_NEAR) * depth_arr)
    return {"rgb": rgb_arr, "depth": depth_m, "camera_pos": camera_pos, "camera_orn": camera_orn}

# ═══════════════════════════════════════════════════════════════
#  TABLE
# ═══════════════════════════════════════════════════════════════
p.loadURDF("plane.urdf")

TABLE_H   = 0.75
TOP_THICK = 0.04
LEG_H     = TABLE_H - TOP_THICK

make_box([0.65, 0.45, TOP_THICK / 2], [0, 0, TABLE_H - TOP_THICK / 2], [0.55, 0.35, 0.15, 1.0])
for lx, ly in [(0.55, 0.37), (-0.55, 0.37), (0.55, -0.37), (-0.55, -0.37)]:
    make_box([0.03, 0.03, LEG_H / 2], [lx, ly, LEG_H / 2], [0.40, 0.25, 0.08, 1.0])

# ═══════════════════════════════════════════════════════════════
#  CUBES
# ═══════════════════════════════════════════════════════════════
CUBE_H = 0.055
CUBE_Z = TABLE_H + CUBE_H

cube_configs = [
    ([-0.25,  0.20], [0.85, 0.20, 0.20, 1.0]),   # red
    ([ 0.10,  0.20], [0.20, 0.55, 0.85, 1.0]),   # blue
    ([-0.10,  0.00], [0.25, 0.75, 0.35, 1.0]),   # green
    ([ 0.25,  0.00], [0.95, 0.75, 0.10, 1.0]),   # yellow
    ([-0.25, -0.20], [0.75, 0.25, 0.85, 1.0]),   # purple
    ([ 0.10, -0.20], [0.95, 0.50, 0.10, 1.0]),   # orange
]

cube_ids = []
for (cx, cy), col in cube_configs:
    cid = make_box([CUBE_H] * 3, [cx, cy, CUBE_Z], col, mass=0.3, friction=1.5)
    cube_ids.append(cid)

# ═══════════════════════════════════════════════════════════════
#  KUKA iiwa 7-DOF ARM  (same URDF as main1.py / custom_env.py)
# ═══════════════════════════════════════════════════════════════
ARM_BASE_X = -0.42
ARM_BASE_Y =  0.30
ARM_BASE_Z =  TABLE_H   # arm stands on table surface

kuka_id = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z],
    useFixedBase=True,
)

NUM_JOINTS   = 7
KUKA_EE_LINK = 6   # lbr_iiwa_link_7 — matches pybullet_envs kuka.py

# Disable default joint damping so resetJointState has full control
for _i in range(NUM_JOINTS):
    p.setJointMotorControl2(kuka_id, _i, p.VELOCITY_CONTROL, force=0)

# Visual gripper fingers — Kuka URDF has no built-in gripper
_bright_org = [1.0, 0.5, 0.0, 1.0]
_I          = p.getQuaternionFromEuler([0, 0, 0])
finger_l    = make_box([0.012, 0.008, 0.04], [ARM_BASE_X - 0.025, ARM_BASE_Y, ARM_BASE_Z + 0.1], _bright_org)
finger_r    = make_box([0.012, 0.008, 0.04], [ARM_BASE_X + 0.025, ARM_BASE_Y, ARM_BASE_Z + 0.1], _bright_org)

# IK parameters from pybullet_envs/bullet/kuka.py
IK_LOWER  = [-0.967, -2.000, -2.960,  0.190, -2.960, -2.090, -3.050]
IK_UPPER  = [ 0.967,  2.000,  2.960,  2.290,  2.960,  2.090,  3.050]
IK_RANGES = [ 5.800,  4.000,  5.800,  4.000,  5.800,  4.000,  6.000]
IK_REST   = [ 0.000,  0.000,  0.000,  0.500 * math.pi, 0.000, -0.500 * math.pi, 0.000]

# Orientation that keeps EE pointing straight down for grasping
EE_DOWN_ORN = p.getQuaternionFromEuler([0, math.pi, 0])

# ── IK ──────────────────────────────────────────────────────────
def ik(target_x, target_y, target_z):
    """Return 7 joint angles that place the EE at (x,y,z) pointing down."""
    joint_poses = p.calculateInverseKinematics(
        kuka_id,
        KUKA_EE_LINK,
        [target_x, target_y, target_z],
        targetOrientation=EE_DOWN_ORN,
        lowerLimits=IK_LOWER,
        upperLimits=IK_UPPER,
        jointRanges=IK_RANGES,
        restPoses=IK_REST,
        maxNumIterations=200,
        residualThreshold=0.005,
    )
    return tuple(float(a) for a in joint_poses[:NUM_JOINTS])

# ── FK / arm update ─────────────────────────────────────────────
def update_arm(joint_angles, gripper_open=True):
    """Set all 7 joints, update visual fingers, return EE world position."""
    for i, angle in enumerate(joint_angles):
        p.resetJointState(kuka_id, i, angle)

    link_state = p.getLinkState(kuka_id, KUKA_EE_LINK, computeForwardKinematics=True)
    ee_pos = list(link_state[4])   # world-frame link origin position
    ee_orn = link_state[5]         # world-frame link orientation

    # Spread fingers along the EE local X-axis
    rot = p.getMatrixFromQuaternion(ee_orn)
    # Column 0 of the rotation matrix = local X in world frame
    rx, ry, rz = rot[0], rot[3], rot[6]
    gap    = 0.045 if gripper_open else 0.016
    fl_pos = [ee_pos[0] + rx * gap, ee_pos[1] + ry * gap, ee_pos[2] + rz * gap]
    fr_pos = [ee_pos[0] - rx * gap, ee_pos[1] - ry * gap, ee_pos[2] - rz * gap]
    p.resetBasePositionAndOrientation(finger_l, fl_pos, ee_orn)
    p.resetBasePositionAndOrientation(finger_r, fr_pos, ee_orn)

    return ee_pos

# ═══════════════════════════════════════════════════════════════
#  GUI SLIDERS
# ═══════════════════════════════════════════════════════════════
sl_j = [
    p.addUserDebugParameter(f"J{i}", IK_LOWER[i], IK_UPPER[i], IK_REST[i])
    for i in range(NUM_JOINTS)
]
sl_auto = p.addUserDebugParameter("AUTO PICK & PLACE (0=off 1=on)", 0, 1, 0)
sl_spd  = p.addUserDebugParameter("Speed", 0.5, 5.0, 2.0)
sl_cam  = p.addUserDebugParameter("CAPTURE CAMERA (0=off 1=on)", 0, 1, 0)

# ═══════════════════════════════════════════════════════════════
#  STATE MACHINE
# ═══════════════════════════════════════════════════════════════
PLACE_X      =  0.48
PLACE_Y      =  0.32
PLACE_BASE_Z =  CUBE_Z
HOVER_ABOVE  =  0.18
LIFT_HEIGHT  =  TABLE_H + 0.35
INTERP_RATE  =  0.04

state       = 'idle'
cube_idx    = 0
held_cube   = None
step_t      = 0
phase_j     = tuple(IK_REST)
target_j    = tuple(IK_REST)
place_count = 0
ee          = None

cube_initial_positions = [p.getBasePositionAndOrientation(cid) for cid in cube_ids]

def lerp_joints(cur, tgt, rate):
    return tuple(c + max(-rate, min(rate, t - c)) for c, t in zip(cur, tgt))

def joints_close(a, b, tol=0.015):
    return all(abs(x - y) < tol for x, y in zip(a, b))

# ═══════════════════════════════════════════════════════════════
#  CAMERA VISUALIZER
# ═══════════════════════════════════════════════════════════════
cam_viz = (
    CameraVisualizer(save_dir="camera_frames", save_enabled=True, display_enabled=False)
    if CAMERA_VIZ_AVAILABLE else None
)

print("=" * 60)
print("  PyBullet — Table + Kuka iiwa 7-DOF + Pick & Place")
print("  Set 'AUTO PICK & PLACE' slider to 1 to start!")
print("  Close the window to quit.")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════
try:
    while True:
        auto_on = p.readUserDebugParameter(sl_auto) > 0.5
        speed   = p.readUserDebugParameter(sl_spd)
        rate    = INTERP_RATE * speed

        # ── Manual mode ─────────────────────────────────────────
        if not auto_on:
            joint_vals = tuple(p.readUserDebugParameter(sl) for sl in sl_j)
            ee      = update_arm(joint_vals, gripper_open=True)
            phase_j = joint_vals
            state   = 'idle'

        # ── Auto pick-and-place ──────────────────────────────────
        else:
            if cube_idx >= len(cube_ids):
                cube_idx    = 0
                place_count = 0
                for cid, (pos, orn) in zip(cube_ids, cube_initial_positions):
                    p.resetBasePositionAndOrientation(cid, pos, orn)

            cube_id     = cube_ids[cube_idx]
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            cx, cy, cz  = cube_pos

            if state == 'idle':
                state    = 'hover_pick'
                target_j = ik(cx, cy, cz + HOVER_ABOVE)

            elif state == 'hover_pick':
                phase_j = lerp_joints(phase_j, target_j, rate)
                ee      = update_arm(phase_j, gripper_open=True)
                if joints_close(phase_j, target_j):
                    state    = 'descend'
                    target_j = ik(cx, cy, cz + CUBE_H * 0.5)

            elif state == 'descend':
                phase_j = lerp_joints(phase_j, target_j, rate * 0.6)
                ee      = update_arm(phase_j, gripper_open=True)
                if joints_close(phase_j, target_j):
                    state = 'grasp'

            elif state == 'grasp':
                ee        = update_arm(phase_j, gripper_open=False)
                p.resetBasePositionAndOrientation(cube_id, [cx, cy, cz], [0, 0, 0, 1])
                held_cube = cube_id
                state     = 'lift'
                target_j  = ik(cx, cy, LIFT_HEIGHT)

            elif state == 'lift':
                phase_j = lerp_joints(phase_j, target_j, rate)
                ee      = update_arm(phase_j, gripper_open=False)
                if held_cube is not None:
                    p.resetBasePositionAndOrientation(
                        held_cube, [ee[0], ee[1], ee[2] - CUBE_H], [0, 0, 0, 1]
                    )
                if joints_close(phase_j, target_j):
                    place_z  = PLACE_BASE_Z + place_count * (CUBE_H * 2 + 0.005)
                    state    = 'carry'
                    target_j = ik(PLACE_X, PLACE_Y, place_z + HOVER_ABOVE)

            elif state == 'carry':
                phase_j = lerp_joints(phase_j, target_j, rate)
                ee      = update_arm(phase_j, gripper_open=False)
                if held_cube is not None:
                    p.resetBasePositionAndOrientation(
                        held_cube, [ee[0], ee[1], ee[2] - CUBE_H], [0, 0, 0, 1]
                    )
                if joints_close(phase_j, target_j):
                    place_z  = PLACE_BASE_Z + place_count * (CUBE_H * 2 + 0.005)
                    state    = 'place_down'
                    target_j = ik(PLACE_X, PLACE_Y, place_z + CUBE_H * 0.5)

            elif state == 'place_down':
                phase_j = lerp_joints(phase_j, target_j, rate * 0.6)
                ee      = update_arm(phase_j, gripper_open=False)
                if held_cube is not None:
                    p.resetBasePositionAndOrientation(
                        held_cube, [ee[0], ee[1], ee[2] - CUBE_H], [0, 0, 0, 1]
                    )
                if joints_close(phase_j, target_j):
                    state = 'release'

            elif state == 'release':
                place_z = PLACE_BASE_Z + place_count * (CUBE_H * 2 + 0.005)
                if held_cube is not None:
                    p.resetBasePositionAndOrientation(
                        held_cube, [PLACE_X, PLACE_Y, place_z], [0, 0, 0, 1]
                    )
                ee          = update_arm(phase_j, gripper_open=True)
                held_cube   = None
                place_count += 1
                state       = 'retreat'
                target_j    = ik(cx, cy, LIFT_HEIGHT)

            elif state == 'retreat':
                phase_j = lerp_joints(phase_j, target_j, rate)
                ee      = update_arm(phase_j, gripper_open=True)
                if joints_close(phase_j, target_j):
                    cube_idx += 1
                    state    = 'idle'

        # ── Camera capture ───────────────────────────────────────
        cam_enabled = p.readUserDebugParameter(sl_cam) > 0.5
        if cam_enabled and ee is not None:
            try:
                cam_data = capture_gripper_camera(ee, phase_j[0])
                rgb, depth = cam_data['rgb'], cam_data['depth']
                if cam_viz is not None:
                    cam_viz.process_frame(
                        cam_data, state=state,
                        action_info=f"Cube {cube_idx}" if auto_on else "Manual",
                    )
                if step_t % 30 == 0:
                    print(f"  Camera: RGB {rgb.shape}  depth [{depth.min():.3f}, {depth.max():.3f}] m")
                    if cam_viz:
                        print(f"  Frames saved: {cam_viz.frame_count // 10}")
            except Exception:
                pass

        step_t += 1
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

except (p.error, KeyboardInterrupt):
    pass
finally:
    if cam_viz is not None:
        cam_viz.save_summary()
    p.disconnect()
    print("Simulation ended.")
