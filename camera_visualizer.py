"""
Camera Feed Visualization for tablecube.py
Displays and saves gripper camera images during pick-and-place operations
"""
import numpy as np
import os
from datetime import datetime

class CameraVisualizer:
    """Saves and optionally displays gripper camera frames."""
    
    def __init__(self, save_dir="camera_frames", save_enabled=True, display_enabled=False):
        """
        Args:
            save_dir: Directory to save frame images
            save_enabled: Whether to save frames to disk
            display_enabled: Whether to display in real-time (requires matplotlib/OpenCV)
        """
        self.save_dir = save_dir
        self.save_enabled = save_enabled
        self.display_enabled = display_enabled
        self.frame_count = 0
        self.current_state = None
        
        if save_enabled:
            # Create timestamped directory for this session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(save_dir, f"session_{timestamp}")
            os.makedirs(self.session_dir, exist_ok=True)
            print(f"📸 Camera frames will be saved to: {self.session_dir}/")
        
        if display_enabled:
            try:
                import matplotlib.pyplot as plt
                self.plt = plt
                print("  • Real-time display enabled (matplotlib)")
            except ImportError:
                print("  ⚠ matplotlib not available for display")
                self.display_enabled = False
    
    def process_frame(self, cam_data, state="", action_info=""):
        """
        Process and save/display camera frame.
        
        Args:
            cam_data: Dict from capture_gripper_camera() with 'rgb' and 'depth'
            state: Current simulation state (e.g., 'grasp', 'carry')
            action_info: Additional info (e.g., 'Cube 2')
        """
        rgb = cam_data['rgb']
        depth = cam_data['depth']
        
        self.frame_count += 1
        self.current_state = state
        
        if self.save_enabled:
            self._save_frame(rgb, depth, state, action_info)
        
        if self.display_enabled:
            self._display_frame(rgb, depth, state)
    
    def _save_frame(self, rgb, depth, state, action_info):
        """Save RGB and depth images."""
        # Normalize depth to 0-255 for visualization
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
        
        # Save only every 10th frame to avoid excessive disk I/O
        if self.frame_count % 10 == 0:
            try:
                import imageio
                
                frame_id = self.frame_count // 10
                rgb_path = os.path.join(self.session_dir, f"frame_{frame_id:04d}_rgb.png")
                depth_path = os.path.join(self.session_dir, f"frame_{frame_id:04d}_depth.png")
                
                imageio.imwrite(rgb_path, rgb)
                imageio.imwrite(depth_path, depth_norm)
                
                # Save metadata
                with open(os.path.join(self.session_dir, f"frame_{frame_id:04d}_info.txt"), 'w') as f:
                    f.write(f"Frame: {frame_id}\n")
                    f.write(f"Global step: {self.frame_count}\n")
                    f.write(f"State: {state}\n")
                    f.write(f"Action: {action_info}\n")
                    f.write(f"Depth range: {depth.min():.3f} - {depth.max():.3f} m\n")
                    f.write(f"RGB range: [{rgb.min()}, {rgb.max()}]\n")
                    
            except ImportError:
                pass  # imageio not available
    
    def _display_frame(self, rgb, depth, state):
        """Display frame in real-time (requires matplotlib)."""
        if not self.display_enabled or not hasattr(self, 'plt'):
            return
        
        try:
            self.plt.figure(figsize=(10, 4))
            
            # RGB
            self.plt.subplot(1, 2, 1)
            self.plt.imshow(rgb)
            self.plt.title(f"RGB View - {state}")
            self.plt.axis('off')
            
            # Depth
            self.plt.subplot(1, 2, 2)
            self.plt.imshow(depth, cmap='viridis')
            self.plt.title(f"Depth Map [{depth.min():.2f}-{depth.max():.2f}m]")
            self.plt.colorbar()
            
            self.plt.tight_layout()
            self.plt.draw()
            self.plt.pause(0.01)
            self.plt.close()
        except Exception as e:
            pass  # Display error, continue silently
    
    def save_summary(self):
        """Create a summary document of captured frames."""
        if not self.save_enabled or not hasattr(self, 'session_dir'):
            return
        
        summary_path = os.path.join(self.session_dir, "README.md")
        with open(summary_path, 'w') as f:
            f.write("# Gripper Camera Frame Capture\n\n")
            f.write(f"**Total frames captured:** {self.frame_count}\n")
            f.write(f"**Saved frames:** {self.frame_count // 10}\n")
            f.write(f"**Final state:** {self.current_state}\n\n")
            f.write("## Files\n")
            f.write("- `frame_XXXX_rgb.png`: RGB image from gripper camera\n")
            f.write("- `frame_XXXX_depth.png`: Normalized depth map (0-255)\n")
            f.write("- `frame_XXXX_info.txt`: Frame metadata (state, depth range, etc.)\n\n")
            f.write("## Viewing Frames\n")
            f.write("```python\n")
            f.write("import cv2\n")
            f.write("rgb = cv2.imread('frame_0001_rgb.png')\n")
            f.write("depth = cv2.imread('frame_0001_depth.png', cv2.IMREAD_GRAYSCALE)\n")
            f.write("cv2.imshow('RGB', rgb)\n")
            f.write("cv2.imshow('Depth', depth)\n")
            f.write("cv2.waitKey(0)\n")
            f.write("```\n")
        
        print(f"📄 Summary saved to: {summary_path}")


def create_camera_video(frame_dir, output_video="camera_feed.mp4", fps=24):
    """
    Create video from saved camera frames.
    
    Args:
        frame_dir: Directory containing frame_XXXX_rgb.png files
        output_video: Output video filename
        fps: Frames per second
    """
    try:
        import imageio
        import glob
        
        # Find all RGB frames
        frames = sorted(glob.glob(os.path.join(frame_dir, "frame_????_rgb.png")))
        
        if not frames:
            print("❌ No frames found to create video")
            return
        
        print(f"📹 Creating video from {len(frames)} frames...")
        
        # Read first frame to get dimensions
        first_frame = imageio.imread(frames[0])
        
        # Create video
        with imageio.get_writer(output_video, fps=fps) as writer:
            for frame_path in frames:
                frame = imageio.imread(frame_path)
                writer.append_data(frame)
        
        print(f"✓ Video saved to: {output_video}")
        
    except ImportError:
        print("❌ imageio not available. Install with: pip install imageio")


# Example integration into tablecube.py:
"""
# At the top of tablecube.py, after imports:
from camera_visualizer import CameraVisualizer

# Initialize visualizer
cam_viz = CameraVisualizer(
    save_dir="camera_frames",
    save_enabled=True,      # Save frames to disk
    display_enabled=False   # Don't display in real-time (too slow with GUI)
)

# In main loop, after camera capture:
if cam_enabled and ee is not None:
    try:
        cam_data = capture_gripper_camera(ee, phase_j[0] if len(phase_j) > 0 else 0)
        
        # NEW: Process frame for visualization
        cam_viz.process_frame(
            cam_data,
            state=state,
            action_info=f"Cube {cube_idx}, Step {step_t}"
        )
        
        # ... rest of logging code ...
        
# At the end (in finally block):
cam_viz.save_summary()
"""
