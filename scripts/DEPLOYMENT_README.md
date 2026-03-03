# Real Drone Deployment Guide

Deployment system for running trained RL policies on BCube ESP32-S2 drone v2.0.

## Quick Start

### 1. Discover Drone Capabilities
```bash
cd simulator
python scripts/discover_drone.py radio://0/80/2M --save-json capabilities.json
```

This will:
- Connect to your drone
- Check available sensors (position, velocity, IMU, etc.)
- Test which command types are supported
- Recommend the best control approach

### 2. Test Basic Control
```bash
python scripts/test_drone_control.py radio://0/80/2M --test all
```

Validates that you can send commands and read state.

### 3. Deploy Policy
```bash
python scripts/deploy_policy.py radio://0/80/2M \
  --preset hover \
  --approach velocity_control \
  --safety-level conservative
```

## Control Approaches

### Approach A: Velocity Control (Recommended)
- **Requirements**: Position/velocity estimation (Lighthouse, motion capture, or GPS)
- **How it works**: Integrates policy accelerations → velocity commands
- **Pros**: Most direct, matches training
- **Cons**: Requires external positioning system

```bash
python scripts/deploy_policy.py radio://0/80/2M \
  --approach velocity_control \
  --goal 1.0 0.0 0.5
```

### Approach B: Hover Control
- **Requirements**: Altitude + IMU (barometer or range finder)
- **How it works**: Converts to body-frame hover commands
- **Pros**: Works without full positioning
- **Cons**: Less accurate, needs coordinate transforms

```bash
python scripts/deploy_policy.py radio://0/80/2M \
  --approach hover_control \
  --goal 1.0 0.0 0.5
```

### Approach C: Attitude Control
- **Requirements**: IMU only
- **How it works**: Converts accelerations → roll/pitch/thrust
- **Pros**: Minimal sensor requirements
- **Cons**: Most complex, needs careful tuning

```bash
python scripts/deploy_policy.py radio://0/80/2M \
  --approach attitude_control \
  --goal 1.0 0.0 0.5
```

## Safety Levels

- **conservative**: 30% action scaling, 0.5 m/s velocity limit (start here!)
- **normal**: 70% action scaling, 1.0 m/s velocity limit
- **full**: 100% action scaling, 2.0 m/s velocity limit

## Goal Presets

- `hover`: Stay at (0, 0, 0.5)
- `forward_1m`: Move 1m forward
- `square_1m`: Navigate 1m square
- `figure_eight`: Continuous figure-eight pattern

Or specify custom goal:
```bash
python scripts/deploy_policy.py radio://0/80/2M --goal 2.0 1.5 0.8
```

## Using Configuration File

Create `experiments/drone_deployment/config.yml` and run:
```bash
python scripts/deploy_policy.py radio://0/80/2M \
  --config experiments/drone_deployment/config.yml
```

## Files

- **`discover_drone.py`**: Query drone capabilities
- **`test_drone_control.py`**: Test basic control commands
- **`deploy_policy.py`**: Main deployment script
- **`policy_bridge.py`**: Converts policy actions to drone commands
- **`goal_handler.py`**: Manages goals and waypoints
- **`experiments/drone_deployment/config.yml`**: Configuration template

## Troubleshooting

### Connection Issues
1. Check drone is powered on and within range
2. Verify USB radio dongle is connected
3. Try scanning for drones: `cfclient`
4. Check antenna orientation

### No Position/Velocity Data
- Your drone may lack positioning system
- Use `--approach hover_control` or `--approach attitude_control`
- Or add positioning deck (Lighthouse, flow deck)

### Drone Not Responding to Commands
1. Run `test_drone_control.py` to verify basic control
2. Check firmware version compatibility
3. Verify battery is sufficiently charged (>3.5V)
4. Try lower safety level (conservative)

### Policy Doesn't Work Well
1. Start with `--safety-level conservative`
2. Check coordinate frame alignment (your positioning system may use different axes)
3. Calibrate coordinate transform in config
4. Consider retraining with hardware-compatible actions

## Flight Logs

Logs are saved to `flight_logs/flight_TIMESTAMP.json` and include:
- Timestamped position/velocity/attitude
- Policy observations and actions
- Drone commands sent
- Goal progress

Visualize using:
```python
# TODO: Add visualization script
```

## Safety Notes

⚠️ **Always:**
- Test in open space with safety net/cage
- Start with `conservative` safety level
- Have kill switch ready (Ctrl+C)
- Monitor battery voltage
- Set appropriate boundary limits
- Keep drone in line of sight

⚠️ **Never:**
- Fly near people
- Exceed tested velocity limits
- Ignore boundary violations
- Deploy without testing first

## Example Workflow

```bash
# 1. Discover capabilities
python scripts/discover_drone.py radio://0/80/2M --save-json my_drone.json

# 2. Check results, note recommended approach
cat my_drone.json

# 3. Test basic control
python scripts/test_drone_control.py radio://0/80/2M --test hover

# 4. Deploy conservatively
python scripts/deploy_policy.py radio://0/80/2M \
  --preset hover \
  --approach velocity_control \
  --safety-level conservative

# 5. If successful, try moving goal
python scripts/deploy_policy.py radio://0/80/2M \
  --goal 0.5 0.0 0.5 \
  --approach velocity_control \
  --safety-level conservative

# 6. Gradually increase difficulty
python scripts/deploy_policy.py radio://0/80/2M \
  --goal 1.0 0.0 0.5 \
  --approach velocity_control \
  --safety-level normal
```

## Next Steps

1. Test connection with `discover_drone.py`
2. Verify control with `test_drone_control.py`
3. Start with hover test at conservative level
4. Gradually increase goal distance
5. Tune parameters in config file if needed
6. Consider retraining policy if sim-to-real gap is large
