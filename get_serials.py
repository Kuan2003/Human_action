import pyrealsense2 as rs

# Create context to list devices
context = rs.context()
devices = context.query_devices()

# Print serial numbers
for dev in devices:
    print(f"Device: {dev.get_info(rs.camera_info.name)}, Serial: {dev.get_info(rs.camera_info.serial_number)}")