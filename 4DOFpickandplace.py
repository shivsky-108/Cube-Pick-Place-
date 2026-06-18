"""
PyBullet — Table + 4-DOF Robotic Arm + Cube Pick & Place
=========================================================
Run:  python table_robot_arm.py

Controls (GUI sliders):
  • Joint 0–3 sliders  → manually pose the arm
  • [AUTO PICK&PLACE]  → set to 1 to start the automated sequence
  • Speed              → animation speed multiplier

The arm automatically picks each cube one-by-one and stacks
them in the back-right corner of the table.
"""

import pybullet as p
import pybullet_data
import time
import math

# ═══════════════════════════════════════════════════════════════
#  IMPORTS - CAMERA VISUALIZATION
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
    cameraTargetPosition=[0, 0, 0.6]
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

def make_cylinder(radius, length, pos, orn, color, mass=0):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, pos, orn)

# ═══════════════════════════════════════════════════════════════
#  GRIPPER CAMERA UTILITIES
# ═══════════════════════════════════════════════════════════════
import numpy as np

# Camera parameters
CAMERA_WIDTH = 64
CAMERA_HEIGHT = 64
CAMERA_FOV = 60.0
CAMERA_NEAR = 0.01
CAMERA_FAR = 2.0
CAMERA_OFFSET = 0.08  # How far ahead of gripper to mount camera (meters)

def get_camera_pose(ee_pos, j0):
    """
    Compute camera position and orientation at gripper tip.
    
    Args:
        ee_pos: End-effector position [x, y, z]
        j0: Base yaw joint angle (rad)
    
    Returns:
        (camera_pos, camera_orn): Position and quaternion orientation
    """
    # Camera points along the gripper approach direction (end-effector Z-axis)
    # Compute the forward direction from j0
    cam_offset_x = math.cos(j0) * CAMERA_OFFSET
    cam_offset_y = math.sin(j0) * CAMERA_OFFSET
    cam_offset_z = 0  # Slightly ahead in the grasp direction
    
    camera_pos = [
        ee_pos[0] + cam_offset_x,
        ee_pos[1] + cam_offset_y,
        ee_pos[2] + cam_offset_z
    ]
    
    # Camera orientation: pointing downward along gripper approach
    # Use j0 for yaw, point slightly downward
    camera_orn = p.getQuaternionFromEuler([math.pi/4, -math.pi/2.5, j0])
    
    return camera_pos, camera_orn

def get_camera_matrix(width, height, fov=60.0):
    """Compute camera intrinsic matrix for reference."""
    aspect = width / height
    f = (height / 2.0) / math.tan(math.radians(fov / 2.0))
    
    fx = f
    fy = f
    cx = width / 2.0
    cy = height / 2.0
    
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

def capture_gripper_camera(ee_pos, j0, ee_orn=None):
    """
    Capture RGB and depth images from gripper-mounted camera.
    
    Args:
        ee_pos: End-effector position [x, y, z]
        j0: Base yaw joint angle (rad)
        ee_orn: Optional end-effector orientation quaternion
    
    Returns:
        {
            'rgb': RGB image (64, 64, 3) as numpy array uint8
            'depth': Depth map (64, 64) as numpy array float32 in meters
            'camera_pos': Camera position
            'camera_orn': Camera orientation
        }
    """
    camera_pos, camera_orn = get_camera_pose(ee_pos, j0)
    
    # Compute view matrix
    target = [
        camera_pos[0] + 0.5 * math.cos(j0),
        camera_pos[1] + 0.5 * math.sin(j0),
        camera_pos[2] - 0.3
    ]
    
    up_vector = [0, 0, 1]
    
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target,
        cameraUpVector=up_vector
    )
    
    # Compute projection matrix
    aspect = CAMERA_WIDTH / CAMERA_HEIGHT
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV,
        aspect=aspect,
        nearVal=CAMERA_NEAR,
        farVal=CAMERA_FAR
    )
    
    # Get camera image
    w, h, rgb, depth, seg = p.getCameraImage(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # Convert to numpy arrays
    rgb_array = np.array(rgb, dtype=np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))[:,:,:3]
    
    # Convert depth to meters
    depth_array = np.array(depth, dtype=np.float32).reshape((CAMERA_HEIGHT, CAMERA_WIDTH))
    depth_meters = CAMERA_FAR * CAMERA_NEAR / (CAMERA_FAR - (CAMERA_FAR - CAMERA_NEAR) * depth_array)
    
    return {
        'rgb': rgb_array,
        'depth': depth_meters,
        'camera_pos': camera_pos,
        'camera_orn': camera_orn
    }

# ═══════════════════════════════════════════════════════════════
#  TABLE
# ═══════════════════════════════════════════════════════════════
p.loadURDF("plane.urdf")

TABLE_H   = 0.75
TOP_THICK = 0.04
LEG_H     = TABLE_H - TOP_THICK
wood      = [0.55, 0.35, 0.15, 1.0]
dark_wood = [0.40, 0.25, 0.08, 1.0]

make_box([0.65, 0.45, TOP_THICK/2], [0, 0, TABLE_H - TOP_THICK/2], wood)

for lx, ly in [(0.55, 0.37), (-0.55, 0.37), (0.55, -0.37), (-0.55, -0.37)]:
    make_box([0.03, 0.03, LEG_H/2], [lx, ly, LEG_H/2], dark_wood)

# ═══════════════════════════════════════════════════════════════
#  CUBES  (placed on table surface)
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
    cid = make_box([CUBE_H]*3, [cx, cy, CUBE_Z], col, mass=0.3, friction=1.5)
    cube_ids.append(cid)

# ═══════════════════════════════════════════════════════════════
#  4-DOF ROBOTIC ARM  (BEEFY & COLORFUL)
# ═══════════════════════════════════════════════════════════════
ARM_BASE_X = -0.42   
ARM_BASE_Y =  0.30
ARM_BASE_Z =  TABLE_H  

L0 = 0.06   # base pedestal height
L1 = 0.22   # upper arm
L2 = 0.18   # forearm
L3 = 0.10   # wrist+hand

# -- NEW VIBRANT COLOR PALETTE --
carbon_blk  = [0.15, 0.15, 0.15, 1.0]
neon_blue   = [0.0, 0.7, 1.0, 1.0]
neon_pink   = [1.0, 0.1, 0.6, 1.0]
white_shell = [0.95, 0.95, 0.95, 1.0]
bright_org  = [1.0, 0.5, 0.0, 1.0]
toxic_green = [0.2, 0.9, 0.2, 1.0]

arm_parts = {}

def quat_from_euler(r, p_, y):
    return p.getQuaternionFromEuler([r, p_, y])

def build_cylinder_link(r, length, pos, orn, color, mass=0):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=length)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=length, rgbaColor=color)
    bid = p.createMultiBody(mass, col, vis, pos, orn)
    return bid

def build_box_link(half, pos, orn, color, mass=0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    bid = p.createMultiBody(mass, col, vis, pos, orn)
    return bid

# Base disk (Bigger, Darker)
arm_parts['base'] = build_cylinder_link(
    0.12, 0.03, [ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z + 0.015],
    quat_from_euler(0, 0, 0), carbon_blk
)
# Pedestal (Thicker, Green)
arm_parts['pedestal'] = build_cylinder_link(
    0.06, L0, [ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z + 0.03 + L0/2],
    quat_from_euler(0, 0, 0), toxic_green
)
# Shoulder hub (Massive, Pink)
arm_parts['shoulder_hub'] = build_cylinder_link(
    0.07, 0.09, [ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z + 0.03 + L0],
    quat_from_euler(math.pi/2, 0, 0), neon_pink
)
# Upper arm (Thick Shell, White)
arm_parts['upper'] = build_cylinder_link(
    0.045, L1, [ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z + 0.03 + L0 + L1/2],
    quat_from_euler(0, 0, 0), white_shell
)
# Elbow hub (Big, Blue)
arm_parts['elbow_hub'] = build_cylinder_link(
    0.06, 0.08, [ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z + 0.03 + L0 + L1],
    quat_from_euler(math.pi/2, 0, 0), neon_blue
)
# Forearm (Thick Shell, White)
arm_parts['forearm'] = build_cylinder_link(
    0.035, L2, [ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z + 0.03 + L0 + L1 + L2/2],
    quat_from_euler(0, 0, 0), white_shell
)
# Wrist hub (Pink)
arm_parts['wrist_hub'] = build_cylinder_link(
    0.05, 0.06, [ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z + 0.03 + L0 + L1 + L2],
    quat_from_euler(math.pi/2, 0, 0), neon_pink
)
# Hand / end-effector body (Chunky, Carbon)
arm_parts['hand'] = build_box_link(
    [0.03, 0.045, 0.04],
    [ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z + 0.03 + L0 + L1 + L2 + 0.04],
    quat_from_euler(0, 0, 0), carbon_blk
)
# Gripper fingers (Thicker, Bright Orange)
arm_parts['finger_l'] = build_box_link(
    [0.012, 0.008, 0.04],
    [ARM_BASE_X - 0.025, ARM_BASE_Y, ARM_BASE_Z + 0.03 + L0 + L1 + L2 + 0.085],
    quat_from_euler(0, 0, 0), bright_org
)
arm_parts['finger_r'] = build_box_link(
    [0.012, 0.008, 0.04],
    [ARM_BASE_X + 0.025, ARM_BASE_Y, ARM_BASE_Z + 0.03 + L0 + L1 + L2 + 0.085],
    quat_from_euler(0, 0, 0), bright_org
)

# ═══════════════════════════════════════════════════════════════
#  FORWARD KINEMATICS
# ═══════════════════════════════════════════════════════════════
def update_arm(j0, j1, j2, j3, gripper_open=True):
    bx, by, bz = ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z

    # ── base ───────────────────────────────────────────────────
    p.resetBasePositionAndOrientation(arm_parts['base'], [bx, by, bz + 0.015], quat_from_euler(0, 0, j0))
    p.resetBasePositionAndOrientation(arm_parts['pedestal'], [bx, by, bz + 0.03 + L0/2], quat_from_euler(0, 0, j0))

    # ── shoulder ───────────────────────────────────────────────
    sh_pos = [bx, by, bz + 0.03 + L0]
    p.resetBasePositionAndOrientation(arm_parts['shoulder_hub'], sh_pos, quat_from_euler(math.pi/2, 0, j0))

    # ── upper arm ──────────────────────────────────────────────
    ux = math.cos(j0) * math.sin(j1)
    uy = math.sin(j0) * math.sin(j1)
    uz = math.cos(j1)

    ua_tip = [sh_pos[0] + ux * L1, sh_pos[1] + uy * L1, sh_pos[2] + uz * L1]
    ua_mid = [sh_pos[0] + ux * L1/2, sh_pos[1] + uy * L1/2, sh_pos[2] + uz * L1/2]

    pitch_ua = -(math.pi/2 - j1)
    p.resetBasePositionAndOrientation(arm_parts['upper'], ua_mid, quat_from_euler(0, pitch_ua, j0))
    p.resetBasePositionAndOrientation(arm_parts['elbow_hub'], ua_tip, quat_from_euler(math.pi/2, 0, j0))

    # ── forearm ────────────────────────────────────────────────
    fa_angle = j1 + j2
    fx = math.cos(j0) * math.sin(fa_angle)
    fy = math.sin(j0) * math.sin(fa_angle)
    fz = math.cos(fa_angle)

    fa_tip = [ua_tip[0] + fx * L2, ua_tip[1] + fy * L2, ua_tip[2] + fz * L2]
    fa_mid = [ua_tip[0] + fx * L2/2, ua_tip[1] + fy * L2/2, ua_tip[2] + fz * L2/2]
    
    pitch_fa = -(math.pi/2 - fa_angle)
    p.resetBasePositionAndOrientation(arm_parts['forearm'], fa_mid, quat_from_euler(0, pitch_fa, j0))
    p.resetBasePositionAndOrientation(arm_parts['wrist_hub'], fa_tip, quat_from_euler(math.pi/2, 0, j0))

    # ── hand / wrist ───────────────────────────────────────────
    wa_angle = fa_angle + j3
    wx = math.cos(j0) * math.sin(wa_angle)
    wy = math.sin(j0) * math.sin(wa_angle)
    wz = math.cos(wa_angle)

    hand_offset = L3 * 0.5
    hand_pos = [fa_tip[0] + wx * hand_offset, fa_tip[1] + wy * hand_offset, fa_tip[2] + wz * hand_offset]
    ee_pos = [fa_tip[0] + wx * L3, fa_tip[1] + wy * L3, fa_tip[2] + wz * L3]
    
    pitch_w = -(math.pi/2 - wa_angle)
    p.resetBasePositionAndOrientation(arm_parts['hand'], hand_pos, quat_from_euler(0, pitch_w, j0))

    # ── fingers ────────────────────────────────────────────────
    gap = 0.045 if gripper_open else 0.016 # Adjusted slightly for the thicker fingers
    perp_x = -math.sin(j0)
    perp_y =  math.cos(j0)

    fl_pos = [ee_pos[0] + perp_x*gap, ee_pos[1] + perp_y*gap, ee_pos[2] + 0.01]
    fr_pos = [ee_pos[0] - perp_x*gap, ee_pos[1] - perp_y*gap, ee_pos[2] + 0.01]

    p.resetBasePositionAndOrientation(arm_parts['finger_l'], fl_pos, quat_from_euler(0, pitch_w, j0))
    p.resetBasePositionAndOrientation(arm_parts['finger_r'], fr_pos, quat_from_euler(0, pitch_w, j0))

    return ee_pos 

# ═══════════════════════════════════════════════════════════════
#  INVERSE KINEMATICS
# ═══════════════════════════════════════════════════════════════
def ik(target_x, target_y, target_z):
    dx = target_x - ARM_BASE_X
    dy = target_y - ARM_BASE_Y
    j0 = math.atan2(dy, dx)

    sh_z = ARM_BASE_Z + 0.03 + L0
    horiz = math.sqrt(dx**2 + dy**2)
    vert  = target_z - sh_z

    D = math.sqrt(horiz**2 + vert**2)
    D = max(0.01, min(D, L1 + L2 - 0.01))  

    cos_j2 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_j2 = max(-1.0, min(1.0, cos_j2))
    j2_raw = math.acos(cos_j2)   

    alpha = math.atan2(vert, horiz)
    beta  = math.acos(max(-1, min(1, (D**2 + L1**2 - L2**2) / (2 * D * L1))))

    j1 = math.pi/2 - (alpha + beta)   
    j2 = math.pi - j2_raw             

    j3 = -(j1 + j2) + math.pi/2      

    return j0, j1, j2, j3

# ═══════════════════════════════════════════════════════════════
#  GUI SLIDERS
# ═══════════════════════════════════════════════════════════════
sl_j0   = p.addUserDebugParameter("J0 Base Yaw",       -math.pi,   math.pi,    0.0)
sl_j1   = p.addUserDebugParameter("J1 Shoulder Pitch",  0.1,        math.pi*0.9, math.pi/4)
sl_j2   = p.addUserDebugParameter("J2 Elbow Pitch",     0.0,        math.pi*0.85, math.pi/3)
sl_j3   = p.addUserDebugParameter("J3 Wrist Pitch",    -math.pi/2, math.pi/2,  0.0)
sl_auto = p.addUserDebugParameter("AUTO PICK & PLACE (0=off 1=on)", 0, 1, 0)
sl_spd  = p.addUserDebugParameter("Speed",              0.5,        5.0,        2.0)
sl_cam  = p.addUserDebugParameter("CAPTURE CAMERA (0=off 1=on)",  0, 1, 0)

# ═══════════════════════════════════════════════════════════════
#  AUTO PICK & PLACE STATE MACHINE
# ═══════════════════════════════════════════════════════════════
PLACE_X =  0.48   
PLACE_Y =  0.32
PLACE_BASE_Z = CUBE_Z

HOVER_ABOVE  = 0.18   
LIFT_HEIGHT  = TABLE_H + 0.35  

state       = 'idle'
cube_idx    = 0
held_cube   = None
step_t      = 0
phase_j     = (0.0, math.pi/4, math.pi/3, 0.0)
target_j    = (0.0, math.pi/4, math.pi/3, 0.0)
INTERP_RATE = 0.04
place_count = 0
ee          = None

# Store spawn positions so cubes can be reset when the cycle repeats
cube_initial_positions = [
    p.getBasePositionAndOrientation(cid) for cid in cube_ids
]

def lerp_joints(cur, tgt, rate):
    return tuple(c + max(-rate, min(rate, t - c)) for c, t in zip(cur, tgt))

def joints_close(a, b, tol=0.015):
    return all(abs(x - y) < tol for x, y in zip(a, b))

# ═══════════════════════════════════════════════════════════════
#  CAMERA VISUALIZER SETUP
# ═══════════════════════════════════════════════════════════════
if CAMERA_VIZ_AVAILABLE:
    cam_viz = CameraVisualizer(
        save_dir="camera_frames",
        save_enabled=True,      # Save frames to disk
        display_enabled=False   # Don't display in real-time (slows down simulation)
    )
else:
    cam_viz = None

print("=" * 60)
print("  PyBullet — Table + 4-DOF Arm + Pick & Place")
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

        if not auto_on:
            j0 = p.readUserDebugParameter(sl_j0)
            j1 = p.readUserDebugParameter(sl_j1)
            j2 = p.readUserDebugParameter(sl_j2)
            j3 = p.readUserDebugParameter(sl_j3)
            ee = update_arm(j0, j1, j2, j3, gripper_open=True)
            phase_j = (j0, j1, j2, j3)
            state = 'idle'

        else:
            if cube_idx >= len(cube_ids):
                cube_idx = 0
                place_count = 0
                for cid, (pos, orn) in zip(cube_ids, cube_initial_positions):
                    p.resetBasePositionAndOrientation(cid, pos, orn)

            cube_id = cube_ids[cube_idx]
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            cx, cy, cz = cube_pos

            if state == 'idle':
                state = 'hover_pick'
                target_j = ik(cx, cy, cz + HOVER_ABOVE)

            elif state == 'hover_pick':
                phase_j = lerp_joints(phase_j, target_j, rate)
                ee = update_arm(*phase_j, gripper_open=True)
                if joints_close(phase_j, target_j):
                    state = 'descend'
                    target_j = ik(cx, cy, cz + CUBE_H * 0.5)

            elif state == 'descend':
                phase_j = lerp_joints(phase_j, target_j, rate * 0.6)
                ee = update_arm(*phase_j, gripper_open=True)
                if joints_close(phase_j, target_j):
                    state = 'grasp'

            elif state == 'grasp':
                ee = update_arm(*phase_j, gripper_open=False)
                p.resetBasePositionAndOrientation(cube_id, [cx, cy, cz], [0,0,0,1])
                held_cube = cube_id
                state = 'lift'
                target_j = ik(cx, cy, LIFT_HEIGHT)

            elif state == 'lift':
                phase_j = lerp_joints(phase_j, target_j, rate)
                ee = update_arm(*phase_j, gripper_open=False)
                if held_cube is not None:
                    p.resetBasePositionAndOrientation(
                        held_cube, [ee[0], ee[1], ee[2] - CUBE_H], [0,0,0,1]
                    )
                if joints_close(phase_j, target_j):
                    place_z = PLACE_BASE_Z + place_count * (CUBE_H * 2 + 0.005)
                    state = 'carry'
                    target_j = ik(PLACE_X, PLACE_Y, place_z + HOVER_ABOVE)

            elif state == 'carry':
                phase_j = lerp_joints(phase_j, target_j, rate)
                ee = update_arm(*phase_j, gripper_open=False)
                if held_cube is not None:
                    p.resetBasePositionAndOrientation(
                        held_cube, [ee[0], ee[1], ee[2] - CUBE_H], [0,0,0,1]
                    )
                if joints_close(phase_j, target_j):
                    place_z = PLACE_BASE_Z + place_count * (CUBE_H * 2 + 0.005)
                    state = 'place_down'
                    target_j = ik(PLACE_X, PLACE_Y, place_z + CUBE_H * 0.5)

            elif state == 'place_down':
                phase_j = lerp_joints(phase_j, target_j, rate * 0.6)
                ee = update_arm(*phase_j, gripper_open=False)
                if held_cube is not None:
                    p.resetBasePositionAndOrientation(
                        held_cube, [ee[0], ee[1], ee[2] - CUBE_H], [0,0,0,1]
                    )
                if joints_close(phase_j, target_j):
                    state = 'release'

            elif state == 'release':
                place_z = PLACE_BASE_Z + place_count * (CUBE_H * 2 + 0.005)
                if held_cube is not None:
                    p.resetBasePositionAndOrientation(
                        held_cube, [PLACE_X, PLACE_Y, place_z], [0,0,0,1]
                    )
                ee = update_arm(*phase_j, gripper_open=True)
                held_cube = None
                place_count += 1
                state = 'retreat'
                target_j = ik(cx, cy, LIFT_HEIGHT)   

            elif state == 'retreat':
                phase_j = lerp_joints(phase_j, target_j, rate)
                ee = update_arm(*phase_j, gripper_open=True)
                if joints_close(phase_j, target_j):
                    cube_idx += 1
                    state = 'idle'

        # ─────────────────────────────────────────────────────────
        #  CAMERA CAPTURE
        # ─────────────────────────────────────────────────────────
        cam_enabled = p.readUserDebugParameter(sl_cam) > 0.5
        if cam_enabled and ee is not None:
            try:
                cam_data = capture_gripper_camera(ee, phase_j[0] if len(phase_j) > 0 else 0)
                rgb = cam_data['rgb']
                depth = cam_data['depth']
                
                # Process for visualization
                if cam_viz is not None:
                    cam_viz.process_frame(
                        cam_data,
                        state=state,
                        action_info=f"Cube {cube_idx}" if auto_on else "Manual"
                    )
                
                # Camera info logging (every 30 frames)
                if step_t % 30 == 0:
                    print(f"  📷 Camera: RGB shape={rgb.shape}, Depth range=[{depth.min():.3f}, {depth.max():.3f}]m")
                    print(f"     Position: [{cam_data['camera_pos'][0]:.3f}, {cam_data['camera_pos'][1]:.3f}, {cam_data['camera_pos'][2]:.3f}]")
                    if cam_viz:
                        print(f"     Frames saved: {cam_viz.frame_count // 10}")
            except Exception as e:
                pass  # Silently skip camera errors
        
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