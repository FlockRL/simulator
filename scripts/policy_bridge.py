"""
Policy control bridge: Converts RL policy actions to drone commands.

Implements multiple control approaches:
- Approach A: Velocity control (acceleration → velocity setpoints)
- Approach B: Hover control (acceleration → body-frame hover commands)
- Approach C: Attitude control (acceleration → roll/pitch/thrust)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class PolicyBridge(ABC):
    """Abstract base class for policy-to-drone control bridges."""
    
    def __init__(
        self,
        max_acceleration: float = 5.0,
        control_dt: float = 0.033333,  # 30 Hz
        action_scale: float = 1.0,
        velocity_limit: float = 1.0
    ):
        """
        Initialize policy bridge.
        
        Args:
            max_acceleration: Maximum acceleration from policy (m/s²)
            control_dt: Control timestep (seconds)
            action_scale: Scale factor for actions (for conservative testing)
            velocity_limit: Maximum velocity magnitude (m/s)
        """
        self.max_acceleration = max_acceleration
        self.control_dt = control_dt
        self.action_scale = action_scale
        self.velocity_limit = velocity_limit
        
        # State tracking
        self.velocity_estimate = np.zeros(3, dtype=np.float32)
        self.last_command_time = 0
        
    @abstractmethod
    def convert_action(
        self,
        policy_action: np.ndarray,
        drone_state: Dict
    ) -> Dict:
        """
        Convert policy action to drone command.
        
        Args:
            policy_action: Policy output (3D acceleration, normalized to [-1, 1])
            drone_state: Current drone state dict with position, velocity, etc.
            
        Returns:
            Dict with command type and parameters
        """
        pass
    
    def build_observation(
        self,
        drone_state: Dict,
        goal_vector: np.ndarray,
        goal_distance: float
    ) -> np.ndarray:
        """
        Build policy observation from drone state.
        
        The policy expects: [vx, vy, vz, goal_dx, goal_dy, goal_dz, goal_dist]
        
        Args:
            drone_state: Current drone state
            goal_vector: Vector from position to goal (dx, dy, dz)
            goal_distance: Euclidean distance to goal
            
        Returns:
            Observation array matching policy training
        """
        # Extract velocity (m/s)
        velocity = np.array(drone_state.get("velocity", [0, 0, 0]), dtype=np.float32)
        
        # Build observation: [vel(3), goal_vector(3), goal_distance(1)]
        obs = np.concatenate([
            velocity,
            goal_vector,
            [goal_distance]
        ], dtype=np.float32)
        
        return obs


class VelocityControlBridge(PolicyBridge):
    """
    Approach A: Velocity Control
    
    Integrates policy accelerations to velocity setpoints.
    Requires: Position/velocity state estimation
    Commands: send_velocity_world_setpoint(vx, vy, vz, yaw_rate)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.integrated_velocity = np.zeros(3, dtype=np.float32)
        
    def convert_action(
        self,
        policy_action: np.ndarray,
        drone_state: Dict
    ) -> Dict:
        """
        Convert acceleration action to velocity setpoint.
        
        Policy outputs normalized actions in [-1, 1].
        Scale to actual accelerations and integrate.
        """
        # Scale action to acceleration (m/s²)
        # Policy action is normalized [-1, 1], scale to [-max_acc, max_acc]
        acceleration = policy_action * self.max_acceleration * self.action_scale
        
        # Get current velocity from drone
        current_velocity = np.array(drone_state.get("velocity", [0, 0, 0]), dtype=np.float32)
        
        # Integrate: v_new = v_current + a * dt
        desired_velocity = current_velocity + acceleration * self.control_dt
        
        # Apply velocity limits
        velocity_magnitude = np.linalg.norm(desired_velocity)
        if velocity_magnitude > self.velocity_limit:
            desired_velocity = desired_velocity / velocity_magnitude * self.velocity_limit
        
        # Store for next iteration
        self.integrated_velocity = desired_velocity
        
        return {
            "type": "velocity_world",
            "vx": float(desired_velocity[0]),
            "vy": float(desired_velocity[1]),
            "vz": float(desired_velocity[2]),
            "yaw_rate": 0.0  # No yaw control for now
        }


class HoverControlBridge(PolicyBridge):
    """
    Approach B: Hover Control
    
    Converts accelerations to body-frame hover commands.
    Requires: Altitude + IMU
    Commands: send_hover_setpoint(vx_body, vy_body, yaw_rate, height)
    """
    
    def __init__(self, target_height: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.target_height = target_height
        self.integrated_velocity = np.zeros(3, dtype=np.float32)
        
    def convert_action(
        self,
        policy_action: np.ndarray,
        drone_state: Dict
    ) -> Dict:
        """
        Convert acceleration to hover command.
        
        Uses world-frame accelerations but commands body-frame velocities.
        Height is maintained separately.
        """
        # Scale action to acceleration
        acceleration = policy_action * self.max_acceleration * self.action_scale
        
        # Get current velocity
        current_velocity = np.array(drone_state.get("velocity", [0, 0, 0]), dtype=np.float32)
        
        # Integrate horizontal only (X, Y)
        desired_velocity = current_velocity + acceleration * self.control_dt
        
        # Apply limits
        velocity_magnitude_xy = np.linalg.norm(desired_velocity[:2])
        if velocity_magnitude_xy > self.velocity_limit:
            desired_velocity[:2] = desired_velocity[:2] / velocity_magnitude_xy * self.velocity_limit
        
        # Get current yaw for coordinate transform
        yaw = drone_state.get("attitude", [0, 0, 0])[2]  # yaw in degrees
        yaw_rad = np.radians(yaw)
        
        # Transform world frame to body frame
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        
        vx_body = desired_velocity[0] * cos_yaw + desired_velocity[1] * sin_yaw
        vy_body = -desired_velocity[0] * sin_yaw + desired_velocity[1] * cos_yaw
        
        # Use goal Z for target height
        target_height = self.target_height
        
        return {
            "type": "hover",
            "vx": float(vx_body),
            "vy": float(vy_body),
            "yaw_rate": 0.0,
            "height": float(target_height)
        }


class AttitudeControlBridge(PolicyBridge):
    """
    Approach C: Attitude Control
    
    Converts accelerations to roll/pitch/thrust commands.
    Most complex, requires careful tuning.
    Commands: send_setpoint(roll, pitch, yaw_rate, thrust)
    """
    
    def __init__(
        self,
        hover_thrust: int = 42000,
        max_angle: float = 15.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hover_thrust = hover_thrust  # PWM value for hover
        self.max_angle = max_angle  # Maximum tilt angle (degrees)
        self.gravity = 9.81
        
    def convert_action(
        self,
        policy_action: np.ndarray,
        drone_state: Dict
    ) -> Dict:
        """
        Convert acceleration to attitude command.
        
        This is approximate and may need tuning for the specific drone.
        """
        # Scale action to acceleration
        acceleration = policy_action * self.max_acceleration * self.action_scale
        
        # Decompose acceleration into horizontal and vertical
        acc_horizontal = acceleration[:2]  # X, Y
        acc_vertical = acceleration[2]      # Z
        
        # Compute required tilt angles (small angle approximation)
        # For small angles: tan(theta) ≈ theta ≈ a_horizontal / g
        desired_pitch = -np.arctan2(acc_horizontal[0], self.gravity)
        desired_roll = np.arctan2(acc_horizontal[1], self.gravity)
        
        # Convert to degrees
        desired_pitch_deg = np.degrees(desired_pitch)
        desired_roll_deg = np.degrees(desired_roll)
        
        # Clamp to safe limits
        desired_pitch_deg = np.clip(desired_pitch_deg, -self.max_angle, self.max_angle)
        desired_roll_deg = np.clip(desired_roll_deg, -self.max_angle, self.max_angle)
        
        # Compute thrust adjustment for vertical acceleration
        # T = m * (g + a_z), normalized by hover thrust
        thrust_adjustment = int(acc_vertical / self.gravity * self.hover_thrust * 0.5)
        thrust = self.hover_thrust + thrust_adjustment
        
        # Clamp thrust to safe range
        thrust = int(np.clip(thrust, 20000, 60000))
        
        return {
            "type": "attitude",
            "roll": float(desired_roll_deg),
            "pitch": float(desired_pitch_deg),
            "yaw_rate": 0.0,
            "thrust": thrust
        }


def create_bridge(approach: str, **kwargs) -> PolicyBridge:
    """
    Create appropriate policy bridge based on approach.
    
    Args:
        approach: "velocity_control", "hover_control", or "attitude_control"
        **kwargs: Parameters for the specific bridge
        
    Returns:
        PolicyBridge instance
    """
    if approach == "velocity_control":
        return VelocityControlBridge(**kwargs)
    elif approach == "hover_control":
        return HoverControlBridge(**kwargs)
    elif approach == "attitude_control":
        return AttitudeControlBridge(**kwargs)
    else:
        raise ValueError(f"Unknown approach: {approach}")


# Safe default configurations for testing
DEFAULT_CONFIGS = {
    "conservative": {
        "action_scale": 0.3,
        "velocity_limit": 0.5,
        "max_acceleration": 5.0
    },
    "normal": {
        "action_scale": 0.7,
        "velocity_limit": 1.0,
        "max_acceleration": 5.0
    },
    "full": {
        "action_scale": 1.0,
        "velocity_limit": 2.0,
        "max_acceleration": 5.0
    }
}
