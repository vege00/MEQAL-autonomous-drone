import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/drone008/ws_ros2_mavsdk/src/telemetry_publisher/install/telemetry_publisher'
