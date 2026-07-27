from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='telemetry_publisher', executable='flight_manager', output='screen'),
        Node(package='telemetry_publisher', executable='mission_sequencer', output='screen'),
    ])