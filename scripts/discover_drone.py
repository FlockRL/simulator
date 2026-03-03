#!/usr/bin/env python3
"""
Discover BCube ESP32-S2 drone capabilities.

This script connects to a BCube drone and queries its:
- Available parameters (settings)
- Available log variables (sensors)
- Supported command types
- Firmware version

Usage:
    # For WiFi/UDP connection (BCube ESP32-S2)
    python scripts/discover_drone.py udp://192.168.4.1:2390
    
    # For radio connection (if you have Crazyradio dongle)
    python scripts/discover_drone.py radio://0/80/2M
    
    # Save results
    python scripts/discover_drone.py udp://192.168.4.1:2390 --save-json capabilities.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add crazyflie-clients-python to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "crazyflie-clients-python" / "src"))

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


def discover_capabilities(scf: SyncCrazyflie, uri: str):
    """Discover and return drone capabilities."""
    cf = scf.cf
    capabilities = {
        "connected": True,
        "uri": uri,
        "parameters": {},
        "log_variables": {},
        "key_capabilities": {},
        "firmware_info": {}
    }
    
    print("=" * 60)
    print("DISCOVERING DRONE CAPABILITIES")
    print("=" * 60)
    
    # Get firmware info
    print("\n[1/4] Firmware Information")
    try:
        # Note: These may not be available on all firmwares
        if hasattr(cf, 'platform'):
            fw_version = getattr(cf.platform, 'get_protocol_version', lambda: "unknown")()
            capabilities["firmware_info"]["protocol_version"] = str(fw_version)
            print(f"  Protocol version: {fw_version}")
    except Exception as e:
        print(f"  Warning: Could not get firmware info: {e}")
    
    # Read parameter TOC
    print("\n[2/4] Available Parameters")
    param_toc = cf.param.toc.toc
    param_groups = {}
    key_params = [
        "commander.enHighLevel",
        "kalman.resetEstimation",
        "stabilizer.controller",
        "motorPowerSet.enable"
    ]
    
    for group in sorted(param_toc.keys()):
        param_groups[group] = []
        for param_name in sorted(param_toc[group].keys()):
            element = param_toc[group][param_name]
            access = "RW" if element.access == 0 else "RO"
            full_name = f"{group}.{param_name}"
            
            param_info = {
                "name": param_name,
                "type": element.ctype,
                "access": access
            }
            
            # Try to get current value
            try:
                value = cf.param.values.get(group, {}).get(param_name, None)
                param_info["value"] = str(value) if value is not None else None
            except:
                param_info["value"] = None
            
            param_groups[group].append(param_info)
            
            # Print key parameters
            if full_name in key_params:
                print(f"  ✓ {full_name} = {param_info['value']} ({access})")
    
    capabilities["parameters"] = param_groups
    
    # Read log TOC
    print("\n[3/4] Available Log Variables")
    log_toc = cf.log.toc.toc
    log_groups = {}
    key_logs = [
        "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
        "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz",
        "stateEstimate.roll", "stateEstimate.pitch", "stateEstimate.yaw",
        "range.zrange", "range.front", "range.back", "range.left", "range.right",
        "pm.vbat", "pm.state"
    ]
    
    for group in sorted(log_toc.keys()):
        log_groups[group] = []
        for log_name in sorted(log_toc[group].keys()):
            element = log_toc[group][log_name]
            full_name = f"{group}.{log_name}"
            
            log_info = {
                "name": log_name,
                "type": element.ctype
            }
            log_groups[group].append(log_info)
            
            # Print key log variables
            if full_name in key_logs:
                print(f"  ✓ {full_name} ({element.ctype})")
    
    capabilities["log_variables"] = log_groups
    
    # Test command support
    print("\n[4/4] Testing Command Support")
    test_results = {}
    
    # Test velocity world setpoint
    try:
        cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
        test_results["velocity_world_setpoint"] = True
        print("  ✓ send_velocity_world_setpoint() - SUPPORTED")
    except Exception as e:
        test_results["velocity_world_setpoint"] = False
        print(f"  ✗ send_velocity_world_setpoint() - FAILED: {e}")
    
    # Test hover setpoint
    try:
        cf.commander.send_hover_setpoint(0, 0, 0, 0)
        test_results["hover_setpoint"] = True
        print("  ✓ send_hover_setpoint() - SUPPORTED")
    except Exception as e:
        test_results["hover_setpoint"] = False
        print(f"  ✗ send_hover_setpoint() - FAILED: {e}")
    
    # Test position setpoint
    try:
        cf.commander.send_position_setpoint(0, 0, 0, 0)
        test_results["position_setpoint"] = True
        print("  ✓ send_position_setpoint() - SUPPORTED")
    except Exception as e:
        test_results["position_setpoint"] = False
        print(f"  ✗ send_position_setpoint() - FAILED: {e}")
    
    # Test basic setpoint (attitude + thrust)
    try:
        cf.commander.send_setpoint(0, 0, 0, 0)
        test_results["attitude_setpoint"] = True
        print("  ✓ send_setpoint() (attitude) - SUPPORTED")
    except Exception as e:
        test_results["attitude_setpoint"] = False
        print(f"  ✗ send_setpoint() (attitude) - FAILED: {e}")
    
    capabilities["key_capabilities"]["command_support"] = test_results
    
    # Analyze capabilities
    print("\n" + "=" * 60)
    print("CAPABILITY ANALYSIS")
    print("=" * 60)
    
    has_position = any(
        "stateEstimate" in group and any(
            log["name"] in ["x", "y", "z", "vx", "vy", "vz"]
            for log in capabilities["log_variables"].get(group, [])
        )
        for group in capabilities["log_variables"]
    )
    
    has_range = any(
        "range" in group
        for group in capabilities["log_variables"]
    )
    
    capabilities["key_capabilities"]["has_state_estimate"] = has_position
    capabilities["key_capabilities"]["has_range_sensors"] = has_range
    
    print(f"\n  Position/Velocity Estimation: {'✓ YES' if has_position else '✗ NO'}")
    print(f"  Range Sensors: {'✓ YES' if has_range else '✗ NO'}")
    print(f"  Velocity Commands: {'✓ YES' if test_results.get('velocity_world_setpoint') else '✗ NO'}")
    print(f"  Hover Commands: {'✓ YES' if test_results.get('hover_setpoint') else '✗ NO'}")
    print(f"  Position Commands: {'✓ YES' if test_results.get('position_setpoint') else '✗ NO'}")
    
    # Recommend control approach
    print("\n" + "=" * 60)
    print("RECOMMENDED CONTROL APPROACH")
    print("=" * 60)
    
    if has_position and test_results.get("velocity_world_setpoint"):
        print("\n  ✓ APPROACH A: Velocity Control (RECOMMENDED)")
        print("    - Drone has position/velocity estimation")
        print("    - Velocity commands are supported")
        print("    - Can directly integrate policy accelerations to velocities")
        print("    - Requires external positioning system (markers/beacons)")
        capabilities["key_capabilities"]["recommended_approach"] = "velocity_control"
        
    elif test_results.get("hover_setpoint"):
        print("\n  ⚠ APPROACH B: Hover Control (FALLBACK)")
        print("    - Use body-frame hover setpoints")
        print("    - Works with altitude + IMU only")
        print("    - Requires coordinate transformation")
        print("    - Less accurate, needs careful tuning")
        capabilities["key_capabilities"]["recommended_approach"] = "hover_control"
        
    elif test_results.get("attitude_setpoint"):
        print("\n  ⚠ APPROACH C: Attitude Control (MANUAL)")
        print("    - Only basic attitude control available")
        print("    - Need to convert accelerations to roll/pitch/thrust")
        print("    - Most complex, requires careful tuning")
        print("    - Consider retraining policy with attitude actions")
        capabilities["key_capabilities"]["recommended_approach"] = "attitude_control"
        
    else:
        print("\n  ✗ ERROR: No suitable control method found")
        print("    - Check firmware compatibility")
        print("    - Verify drone connection")
        capabilities["key_capabilities"]["recommended_approach"] = "unknown"
    
    return capabilities


def main():
    parser = argparse.ArgumentParser(
        description="Discover BCube drone capabilities",
        epilog="""
Connection URIs:
  WiFi/UDP (BCube ESP32-S2): udp://192.168.4.1:2390
  Radio (with dongle):       radio://0/80/2M
  
For WiFi: Connect to drone's WiFi network first (e.g., ESP_DRONE_XXXX)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("uri", nargs='?', default="udp://192.168.4.1:2390",
                       help="Drone URI (default: udp://192.168.4.1:2390 for WiFi)")
    parser.add_argument("--save-json", type=str, help="Save capabilities to JSON file")
    parser.add_argument("--cache", type=str, default="./cache", help="Cache directory for parameters")
    
    args = parser.parse_args()
    
    # Provide WiFi connection instructions if using UDP
    if args.uri.startswith('udp://'):
        print("\n" + "=" * 60)
        print("WiFi CONNECTION MODE")
        print("=" * 60)
        print("Before connecting, make sure:")
        print("  1. Drone is powered ON")
        print("  2. Your computer is connected to the drone's WiFi network")
        print("     (Look for: ESP_DRONE_XXXX or similar)")
        print("  3. If you can't find the WiFi network, check your drone manual")
        print("=" * 60 + "\n")
    
    print(f"Connecting to {args.uri}...")
    print("(This may take a few seconds...)\n")
    
    # Initialize drivers
    cflib.crtp.init_drivers()
    
    try:
        # Connect with cache
        with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache=args.cache)) as scf:
            print("✓ Connected!\n")
            time.sleep(1)  # Let connection stabilize
            
            # Discover capabilities
            capabilities = discover_capabilities(scf, args.uri)
            
            # Save to JSON if requested
            if args.save_json:
                output_path = Path(args.save_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(capabilities, f, indent=2)
                
                print(f"\n✓ Capabilities saved to: {output_path}")
            
            print("\n" + "=" * 60)
            print("DISCOVERY COMPLETE")
            print("=" * 60)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        
        if args.uri.startswith('udp://'):
            print("  WiFi/UDP Connection Issues:")
            print("  1. Is your computer connected to the drone's WiFi network?")
            print("     - Check WiFi settings for ESP_DRONE_XXXX or BCube network")
            print("  2. Is the drone powered on? (LEDs should be blinking)")
            print("  3. Try pinging the drone: ping 192.168.4.1")
            print("  4. Check if IP/port is correct (default: 192.168.4.1:2390)")
            print("  5. Some drones use different IPs like 192.168.43.1")
        else:
            print("  Radio Connection Issues:")
            print("  1. Check drone is powered on")
            print("  2. Verify USB radio dongle is connected")
            print("  3. Install libusb: pip install pyusb; then download libusb DLL")
            print("  4. Try scanning: cfclient")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
