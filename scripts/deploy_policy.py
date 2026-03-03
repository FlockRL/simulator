#!/usr/bin/env python3
"""
Deploy trained RL policy to BCube ESP32-S2 drone.

This script:
1. Loads a trained PPO policy
2. Connects to the drone
3. Runs the policy control loop
4. Logs flight data for analysis

Usage:
    # WiFi connection (BCube ESP32-S2)
    python scripts/deploy_policy.py udp://192.168.4.1:2390 --goal 1.0 0.0 0.5
    python scripts/deploy_policy.py udp://192.168.4.1:2390 --preset hover
    
    # Radio connection (with dongle)
    python scripts/deploy_policy.py radio://0/80/2M --goal 1.0 0.0 0.5
    python scripts/deploy_policy.py radio://0/80/2M --config drone_deployment/config.yml
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import yaml

# Add paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "crazyflie-clients-python" / "src"))
sys.path.insert(0, str(REPO_ROOT / "simulator"))

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from stable_baselines3 import PPO

from goal_handler import create_goal_handler, SingleGoalHandler
from policy_bridge import create_bridge, DEFAULT_CONFIGS


class DroneController:
    """Main controller for policy deployment."""
    
    def __init__(
        self,
        scf: SyncCrazyflie,
        uri: str,
        policy_path: str,
        goal_handler,
        bridge,
        config: dict
    ):
        self.scf = scf
        self.cf = scf.cf
        self.uri = uri
        self.goal_handler = goal_handler
        self.bridge = bridge
        self.config = config
        
        # Load policy
        print(f"Loading policy from {policy_path}...")
        self.model = PPO.load(policy_path)
        print("✓ Policy loaded")
        
        # State tracking
        self.drone_state = {
            "position": np.zeros(3),
            "velocity": np.zeros(3),
            "attitude": np.zeros(3),
            "battery": 0.0,
            "timestamp": 0
        }
        
        # Logging
        self.flight_log = []
        self.log_conf = None
        
        # Safety
        self.battery_threshold = config.get("safety", {}).get("battery_threshold", 3.0)
        self.max_flight_time = config.get("safety", {}).get("max_flight_time", 60.0)
        self.boundary_limits = config.get("safety", {}).get("boundary_limits", None)
        self.start_time = None
        self.safety_grace_period = 2.0  # Don't check boundaries for first 2 seconds
        
    def setup_logging(self):
        """Setup drone state logging."""
        print("Setting up state logging...")
        
        # Reduced log config - BCube can't handle too many variables at once
        self.log_conf = LogConfig(name='PolicyControl', period_in_ms=33)  # 30 Hz
        
        # Add only essential variables (max ~6-7 floats for ESP-Drone)
        try:
            self.log_conf.add_variable('stateEstimate.x', 'float')
            self.log_conf.add_variable('stateEstimate.y', 'float')
            self.log_conf.add_variable('stateEstimate.z', 'float')
            self.log_conf.add_variable('stateEstimate.vx', 'float')
            self.log_conf.add_variable('stateEstimate.vy', 'float')
            self.log_conf.add_variable('pm.vbat', 'float')
            print("✓ Logging configured")
        except KeyError as e:
            print(f"⚠ Warning: Could not add log variable: {e}")
        
        self.log_conf.data_received_cb.add_callback(self._log_callback)
        
        try:
            self.cf.log.add_config(self.log_conf)
            self.log_conf.start()
        except AttributeError as e:
            print(f"⚠ Warning: Logging setup failed: {e}")
            print("Will attempt to continue without full logging...")
        
    def _log_callback(self, timestamp, data, logconf):
        """Handle incoming log data."""
        self.drone_state.update({
            "position": np.array([
                data.get('stateEstimate.x', 0.0),
                data.get('stateEstimate.y', 0.0),
                data.get('stateEstimate.z', 0.0)
            ], dtype=np.float32),
            "velocity": np.array([
                data.get('stateEstimate.vx', 0.0),
                data.get('stateEstimate.vy', 0.0),
                0.0  # vz not logged to save space
            ], dtype=np.float32),
            "attitude": np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Not logged
            "battery": data.get('pm.vbat', 0.0),
            "timestamp": timestamp
        })
        
    def check_safety(self) -> tuple[bool, str]:
        """
        Check safety conditions.
        
        Returns:
            (is_safe, reason_if_unsafe)
        """
        # Battery check
        print(f"Checking safety: battery={self.drone_state['battery']:.2f}V")
        if self.drone_state["battery"] < self.battery_threshold:
            return False, f"Low battery: {self.drone_state['battery']:.2f}V"
        
        # Time limit
        if self.start_time and (time.time() - self.start_time) > self.max_flight_time:
            return False, f"Max flight time exceeded: {self.max_flight_time}s"
        if self.boundary_limits and self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.safety_grace_period:  # Only check after grace period
                pos = self.drone_state["position"]
                limits = self.boundary_limits
                
                if not (limits["x_min"] <= pos[0] <= limits["x_max"]):
                    return False, f"X boundary violated: {pos[0]:.2f}"
                if not (limits["y_min"] <= pos[1] <= limits["y_max"]):
                    return False, f"Y boundary violated: {pos[1]:.2f}"
                if not (limits["z_min"] <= pos[2] <= limits["z_max"]):
                    return False, f"Z boundary violated: {pos[2]:.2f}"
        
        return True, ""
    
    def send_command(self, command: dict):
        """Send command to drone based on type."""
        cmd_type = command["type"]
        
        if cmd_type == "velocity_world":
            self.cf.commander.send_velocity_world_setpoint(
                command["vx"],
                command["vy"],
                command["vz"],
                command["yaw_rate"]
            )
        elif cmd_type == "hover":
            self.cf.commander.send_hover_setpoint(
                command["vx"],
                command["vy"],
                command["yaw_rate"],
                command["height"]
            )
        elif cmd_type == "attitude":
            self.cf.commander.send_setpoint(
                command["roll"],
                command["pitch"],
                command["yaw_rate"],
                command["thrust"]
            )
        else:
            raise ValueError(f"Unknown command type: {cmd_type}")
    
    def emergency_stop(self):
        """Execute emergency stop."""
        print("\n🛑 EMERGENCY STOP")
        for _ in range(20):
            self.cf.commander.send_stop_setpoint()
            time.sleep(0.01)
    
    def run_control_loop(self):
        """Main control loop."""
        print("\n" + "=" * 60)
        print("STARTING POLICY CONTROL")
        print("=" * 60)
        
        self.start_time = time.time()
        control_dt = self.bridge.control_dt
        
        try:
            step_count = 0
            
            while not self.goal_handler.is_complete():
                step_start = time.time()
                
                # Check safety
                is_safe, reason = self.check_safety()
                if not is_safe:
                    print(f"\n⚠ Safety violation: {reason}")
                    break
                
                # Get current position in simulator coordinates
                current_pos_real = self.drone_state["position"]
                current_pos_sim = self.goal_handler.real_to_sim(current_pos_real)
                
                # Check goal reached
                if self.goal_handler.check_goal_reached(current_pos_sim):
                    print(f"\n✓ Goal {self.goal_handler.current_goal_idx} reached!")
                    if self.goal_handler.is_complete():
                        print("🎉 All goals completed!")
                        break
                
                # Compute goal information
                goal_vector = self.goal_handler.compute_goal_vector(current_pos_sim)
                goal_distance = self.goal_handler.compute_goal_distance(current_pos_sim)
                
                # Build observation for policy
                obs = self.bridge.build_observation(
                    self.drone_state,
                    goal_vector,
                    goal_distance
                )
                
                # Get action from policy
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Convert to drone command
                command = self.bridge.convert_action(action, self.drone_state)
                
                # Send command
                self.send_command(command)
                
                # Log data
                log_entry = {
                    "step": step_count,
                    "time": time.time() - self.start_time,
                    "position": self.drone_state["position"].tolist(),
                    "velocity": self.drone_state["velocity"].tolist(),
                    "goal_vector": goal_vector.tolist(),
                    "goal_distance": float(goal_distance),
                    "action": action.tolist(),
                    "command": command,
                    "battery": float(self.drone_state["battery"])
                }
                self.flight_log.append(log_entry)
                
                # Print progress
                if step_count % 10 == 0:
                    print(f"Step {step_count:4d} | "
                          f"Pos: ({current_pos_sim[0]:.2f}, {current_pos_sim[1]:.2f}, {current_pos_sim[2]:.2f}) | "
                          f"Goal dist: {goal_distance:.2f}m | "
                          f"Bat: {self.drone_state['battery']:.2f}V")
                
                step_count += 1
                
                # Sleep to maintain control frequency
                elapsed = time.time() - step_start
                sleep_time = max(0, control_dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
        except Exception as e:
            print(f"\n\n✗ Error during control: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.emergency_stop()
            
        # Print summary
        print("\n" + "=" * 60)
        print("FLIGHT SUMMARY")
        print("=" * 60)
        
        progress = self.goal_handler.get_progress()
        print(f"Goals reached: {progress['goals_reached']} / {progress['total_goals']}")
        print(f"Flight time: {time.time() - self.start_time:.1f}s")
        print(f"Total steps: {len(self.flight_log)}")
        print(f"Final battery: {self.drone_state['battery']:.2f}V")
        
    def save_flight_log(self, output_path: Path):
        """Save flight log to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        log_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "drone_uri": self.uri,
                "policy_path": str(self.config.get("policy_path", "unknown")),
                "approach": self.config.get("approach", "unknown"),
                "flight_time": time.time() - self.start_time if self.start_time else 0
            },
            "goal_progress": self.goal_handler.get_progress(),
            "flight_log": self.flight_log
        }
        
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n✓ Flight log saved to: {output_path}")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Deploy RL policy to BCube drone")
    parser.add_argument("uri", nargs='?', default="udp://192.168.4.1:2390",
                       help="Drone URI (default: udp://192.168.4.1:2390)")
    parser.add_argument("--config", type=str, help="Config YAML file")
    parser.add_argument("--policy", type=str, 
                       default="experiments/memorized_model_jerk/model.zip",
                       help="Path to policy model")
    parser.add_argument("--approach", choices=["velocity_control", "hover_control", "attitude_control"],
                       default="velocity_control", help="Control approach")
    parser.add_argument("--preset", choices=["hover", "forward_1m", "square_1m", "figure_eight"],
                       help="Goal preset")
    parser.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "Z"),
                       help="Custom goal position")
    parser.add_argument("--safety-level", choices=["conservative", "normal", "full"],
                       default="conservative", help="Safety/aggressiveness level")
    parser.add_argument("--output", type=str, help="Output path for flight log")
    parser.add_argument("--cache", type=str, default="./cache", help="Cache directory")
    
    args = parser.parse_args()
    
    # Load or build config
    if args.config:
        config = load_config(Path(args.config))
    else:
        config = {
            "approach": args.approach,
            "policy_path": args.policy,
            "safety": {
                "battery_threshold": 3.2,
                "max_flight_time": 120.0,
                "boundary_limits": {
                    "x_min": -2.0, "x_max": 2.0,
                    "y_min": -2.0, "y_max": 2.0,
                    "z_min": -0.5, "z_max": 2.5  # Allow ground level
                }
            }
        }
    
    # Setup goal handler
    if args.preset:
        goal_handler = create_goal_handler(args.preset)
    elif args.goal:
        goal_handler = SingleGoalHandler(tuple(args.goal))
    else:
        # Default hover goal
        goal_handler = SingleGoalHandler((0.0, 0.0, 0.5))
    
    # Setup policy bridge
    bridge_config = DEFAULT_CONFIGS[args.safety_level]
    bridge = create_bridge(args.approach, **bridge_config)
    
    print("=" * 60)
    print("POLICY DEPLOYMENT CONFIGURATION")
    print("=" * 60)
    print(f"Policy: {args.policy}")
    print(f"Approach: {args.approach}")
    print(f"Safety level: {args.safety_level}")
    print(f"Goal: {goal_handler.get_current_goal()}")
    print(f"Drone URI: {args.uri}")
    print("=" * 60)
    
    response = input("\nReady to deploy? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted")
        return
    
    # Initialize and connect
    print(f"\nConnecting to {args.uri}...")
    cflib.crtp.init_drivers()
    
    try:
        with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache=args.cache)) as scf:
            print("✓ Connected!")
            
            # Create controller
            controller = DroneController(
                scf,
                args.uri,
                args.policy,
                goal_handler,
                bridge,
                config
            )
            
            # Setup logging
            controller.setup_logging()
            time.sleep(1)  # Let logging stabilize
            
            # Run control loop
            controller.run_control_loop()
            
            # Save log
            if args.output:
                output_path = Path(args.output)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"flight_logs/flight_{timestamp}.json")
            
            controller.save_flight_log(output_path)
            
            # Cleanup
            controller.log_conf.stop()
            controller.log_conf.delete()
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
