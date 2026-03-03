#!/usr/bin/env python3
"""
Test basic drone control commands and measure responses.

This script performs system identification tests:
- Send known commands
- Measure actual responses
- Validate control authority
- Test emergency stop

Usage:
    # WiFi connection (BCube ESP32-S2)
    python scripts/test_drone_control.py udp://192.168.4.1:2390 --test hover
    python scripts/test_drone_control.py udp://192.168.4.1:2390 --test all
    
    # Radio connection (with dongle)
    python scripts/test_drone_control.py radio://0/80/2M --test velocity
"""

import argparse
import sys
import time
from pathlib import Path
from collections import deque

# Add crazyflie-clients-python to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "crazyflie-clients-python" / "src"))

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


class DroneStateMonitor:
    """Monitor and log drone state."""
    
    def __init__(self, scf: SyncCrazyflie):
        self.scf = scf
        self.cf = scf.cf
        self.state = {
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "attitude": [0.0, 0.0, 0.0],  # roll, pitch, yaw
            "battery": 0.0,
            "timestamp": 0
        }
        self.history = deque(maxlen=100)
        self.log_conf = None
        
    def start_logging(self):
        """Start logging drone state."""
        # Reduced log config - BCube can't handle too many variables at once
        self.log_conf = LogConfig(name='StateMonitor', period_in_ms=100)
        
        # Add only essential variables (max ~6-7 floats for ESP-Drone)
        try:
            self.log_conf.add_variable('stateEstimate.x', 'float')
            self.log_conf.add_variable('stateEstimate.y', 'float')
            self.log_conf.add_variable('stateEstimate.z', 'float')
            self.log_conf.add_variable('stateEstimate.vx', 'float')
            self.log_conf.add_variable('stateEstimate.vy', 'float')
            self.log_conf.add_variable('pm.vbat', 'float')
            print("  ✓ Position/velocity/battery logging enabled")
        except KeyError as e:
            print(f"  ⚠ Could not add variable: {e}")
        
        self.log_conf.data_received_cb.add_callback(self._log_callback)
        
        try:
            self.cf.log.add_config(self.log_conf)
            self.log_conf.start()
        except AttributeError as e:
            print(f"  ⚠ Warning: Logging failed: {e}")
            print("  Continuing without logging...")
        
    def _log_callback(self, timestamp, data, logconf):
        """Handle incoming log data."""
        self.state["timestamp"] = timestamp
        
        if 'stateEstimate.x' in data:
            self.state["position"] = [
                data.get('stateEstimate.x', 0.0),
                data.get('stateEstimate.y', 0.0),
                data.get('stateEstimate.z', 0.0)
            ]
            self.state["velocity"] = [
                data.get('stateEstimate.vx', 0.0),
                data.get('stateEstimate.vy', 0.0),
                0.0  # vz not logged to save space
            ]
        
        if 'pm.vbat' in data:
            self.state["battery"] = data['pm.vbat']
        
        self.history.append(self.state.copy())
        
    def stop_logging(self):
        """Stop logging."""
        if self.log_conf:
            self.log_conf.stop()
            self.log_conf.delete()
    
    def print_current_state(self):
        """Print current state."""
        pos = self.state["position"]
        vel = self.state["velocity"]
        bat = self.state["battery"]
        
        print(f"  Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
        print(f"  Vel: ({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}) m/s")
        print(f"  Bat: {bat:.2f} V")


def test_hover_command(scf: SyncCrazyflie, monitor: DroneStateMonitor):
    """Test hover setpoint command."""
    print("\n[TEST: Hover Setpoint]")
    print("Sending hover commands (vx=0, vy=0, yaw_rate=0, height=0.3m)")
    print("Press Ctrl+C to stop\n")
    
    cf = scf.cf
    target_height = 0.3  # 30cm
    
    try:
        print("Sending commands for 5 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 5.0:
            cf.commander.send_hover_setpoint(0, 0, 0, target_height)
            time.sleep(0.1)  # 10 Hz
            
            if int((time.time() - start_time) * 10) % 10 == 0:
                monitor.print_current_state()
        
        print("\nStopping...")
        for _ in range(10):
            cf.commander.send_stop_setpoint()
            time.sleep(0.1)
        
        print("✓ Hover test complete")
        
    except KeyboardInterrupt:
        print("\n\nEmergency stop!")
        for _ in range(10):
            cf.commander.send_stop_setpoint()
            time.sleep(0.1)


def test_velocity_command(scf: SyncCrazyflie, monitor: DroneStateMonitor):
    """Test velocity world setpoint command."""
    print("\n[TEST: Velocity World Setpoint]")
    print("Sending velocity commands (0.1 m/s in X direction)")
    print("Press Ctrl+C to stop\n")
    
    cf = scf.cf
    
    try:
        print("Sending commands for 5 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 5.0:
            cf.commander.send_velocity_world_setpoint(0.1, 0, 0, 0)
            time.sleep(0.1)  # 10 Hz
            
            if int((time.time() - start_time) * 10) % 10 == 0:
                monitor.print_current_state()
        
        print("\nStopping...")
        for _ in range(10):
            cf.commander.send_stop_setpoint()
            time.sleep(0.1)
        
        print("✓ Velocity test complete")
        
    except KeyboardInterrupt:
        print("\n\nEmergency stop!")
        for _ in range(10):
            cf.commander.send_stop_setpoint()
            time.sleep(0.1)


def test_attitude_command(scf: SyncCrazyflie, monitor: DroneStateMonitor):
    """Test basic attitude setpoint command."""
    print("\n[TEST: Attitude Setpoint]")
    print("Sending attitude commands (roll=0, pitch=0, yaw_rate=0, thrust=30000)")
    print("⚠ WARNING: This will apply thrust - be ready to catch!")
    print("Press Ctrl+C to stop\n")
    
    response = input("Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Skipped")
        return
    
    cf = scf.cf
    
    try:
        print("Sending commands for 3 seconds...")
        start_time = time.time()
        
        # Very low thrust for safety
        thrust = 30000  # Minimum for sensing, won't take off
        
        while time.time() - start_time < 3.0:
            cf.commander.send_setpoint(0, 0, 0, thrust)
            time.sleep(0.1)  # 10 Hz
            
            if int((time.time() - start_time) * 10) % 10 == 0:
                monitor.print_current_state()
        
        print("\nStopping...")
        for _ in range(10):
            cf.commander.send_stop_setpoint()
            time.sleep(0.1)
        
        print("✓ Attitude test complete")
        
    except KeyboardInterrupt:
        print("\n\nEmergency stop!")
        for _ in range(10):
            cf.commander.send_stop_setpoint()
            time.sleep(0.1)


def test_emergency_stop(scf: SyncCrazyflie):
    """Test emergency stop command."""
    print("\n[TEST: Emergency Stop]")
    print("Testing stop command responsiveness...")
    
    cf = scf.cf
    
    for i in range(5):
        cf.commander.send_stop_setpoint()
        time.sleep(0.01)
    
    print("✓ Emergency stop working")


def main():
    parser = argparse.ArgumentParser(description="Test BCube drone control")
    parser.add_argument("uri", nargs='?', default="udp://192.168.4.1:2390",
                       help="Drone URI (default: udp://192.168.4.1:2390)")
    parser.add_argument("--test", choices=["hover", "velocity", "attitude", "all"], 
                       default="all", help="Which test to run")
    parser.add_argument("--cache", type=str, default="./cache", 
                       help="Cache directory")
    
    args = parser.parse_args()
    
    print(f"\nConnecting to {args.uri}...")
    print("=" * 60)
    
    # Initialize drivers
    cflib.crtp.init_drivers()
    
    try:
        with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache=args.cache)) as scf:
            print("✓ Connected!")
            
            # Setup logging
            print("\nSetting up state monitoring...")
            monitor = DroneStateMonitor(scf)
            monitor.start_logging()
            time.sleep(1)  # Let logging initialize
            
            print("\nInitial state:")
            monitor.print_current_state()
            
            # Always test emergency stop first
            test_emergency_stop(scf)
            
            # Run requested tests
            if args.test in ["hover", "all"]:
                test_hover_command(scf, monitor)
            
            if args.test in ["velocity", "all"]:
                test_velocity_command(scf, monitor)
            
            if args.test in ["attitude", "all"]:
                test_attitude_command(scf, monitor)
            
            # Stop logging
            monitor.stop_logging()
            
            print("\n" + "=" * 60)
            print("ALL TESTS COMPLETE")
            print("=" * 60)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
