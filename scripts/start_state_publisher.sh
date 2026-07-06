#!/bin/bash
export HOME=/home/drone008
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

source /opt/ros/humble/setup.bash
source /home/drone008/ws_auto_drone_008/install/setup.bash
source /home/drone008/ws_livox_drivers/install/setup.bash

# Read the file into a variable here in the script - no shell quoting issues
URDF=$(cat /home/drone008/ws_auto_drone_008/drone_tf/drone-008.xml)

exec ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$URDF"

exec ros2 run telemetry_publisher telemetry_node 

