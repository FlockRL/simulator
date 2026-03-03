# WiFi Connection Guide for BCube ESP32-S2 Drone

Your BCube ESP32-S2 drone uses **WiFi** instead of a radio dongle. Here's how to connect:

## Quick Setup

### Step 1: Power On the Drone

1. Make sure battery is connected and charged (>3.5V)
2. Flip the power switch ON
3. Wait for LEDs to start blinking (usually blue/green)
4. **Wait 5-10 seconds** for the WiFi network to appear

### Step 2: Connect to Drone's WiFi Network

**On Windows:**

1. Click WiFi icon in taskbar (bottom right)
2. Look for a network named like:
   - `ESP_DRONE_XXXX`
   - `BCUBE_XXXX` 
   - `Crazyflie_XXXX`
   - Or check your drone's manual/label for the exact name

3. Click the network and connect
4. Enter password if required (check manual, common defaults: `12345678` or `password`)

**Using PowerShell to find it:**
```powershell
# List all WiFi networks
netsh wlan show networks

# Look for ESP_DRONE or BCUBE in the list
```

### Step 3: Note the IP Address

**Most common default:** `192.168.4.1:2390`

If that doesn't work, try:
- `192.168.43.1:2390`
- `192.168.4.1:8888`

Check your drone's documentation for the exact IP/port.

### Step 4: Test Connection

```powershell
# Test if drone is reachable
ping 192.168.4.1

# Should see replies like:
# Reply from 192.168.4.1: bytes=32 time=5ms TTL=64
```

## Using the Scripts with WiFi

Now use the WiFi URI instead of radio:

### Discover Drone (Step 1)
```powershell
cd simulator
python scripts/discover_drone.py udp://192.168.4.1:2390 --save-json capabilities.json
```

### Test Control (Step 2)
```powershell
python scripts/test_drone_control.py udp://192.168.4.1:2390 --test all
```

### Deploy Policy (Step 3)
```powershell
python scripts/deploy_policy.py udp://192.168.4.1:2390 --preset hover --safety-level conservative
```

## Troubleshooting WiFi Connection

### Can't Find Drone's WiFi Network

**Check:**
1. Drone is powered ON (LEDs blinking)
2. Wait 10-15 seconds after power on
3. Check if drone needs initial configuration
4. Try restarting the drone

**Find the network name:**
- Check label on drone
- Check manual/documentation
- Try putting drone in AP (Access Point) mode if it has a mode button

### Connected but Can't Ping

**Try different IP addresses:**
```powershell
ping 192.168.4.1
ping 192.168.43.1
ping 192.168.1.1
```

**Check connection:**
```powershell
# See your connected WiFi network
netsh wlan show interfaces

# Should show the drone's network as connected
```

### Connection Keeps Dropping

**Improve WiFi signal:**
- Keep computer within 5 meters of drone
- Minimize obstacles between computer and drone
- Ensure drone antenna is not blocked
- Disable other WiFi networks if possible

**Windows may try to disconnect (no internet):**
- Right-click drone's WiFi network
- Properties → Uncheck "Connect automatically when in range"
- Or disable "Set this network as metered connection"

## Common ESP-Drone WiFi Configurations

Your BCube is based on ESP-Drone. Common settings:

| Setting | Value |
|---------|-------|
| **Default SSID** | ESP_DRONE_XXXX or BCUBE_XXXX |
| **Default Password** | 12345678 |
| **Default IP** | 192.168.4.1 |
| **Control Port** | 2390 (UDP) |
| **Mode** | Access Point (AP) |

> **Note:** XXXX is usually based on the drone's MAC address

## Alternative: Configure Drone to Join Your WiFi

Some BCube drones can be configured to join your existing WiFi network instead of creating their own. Check your drone's manual for:
- Station mode (STA) vs Access Point mode (AP)
- WiFi configuration via mobile app or web interface
- Config file on the drone

## Next Steps

Once connected to WiFi and can ping the drone:

```powershell
# 1. Discover capabilities  
python scripts/discover_drone.py

# 2. Test control
python scripts/test_drone_control.py

# 3. Deploy policy
python scripts/deploy_policy.py --preset hover --safety-level conservative
```

**All scripts now default to WiFi connection** (`udp://192.168.4.1:2390`) so you don't need to specify the URI every time!

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│        BCube WiFi Quick Start           │
├─────────────────────────────────────────┤
│ 1. Power ON drone                       │
│ 2. Connect to: ESP_DRONE_XXXX           │
│ 3. Test: ping 192.168.4.1               │
│ 4. Run: python scripts/discover_drone.py│
└─────────────────────────────────────────┘

Default URI: udp://192.168.4.1:2390
Common password: 12345678
```
