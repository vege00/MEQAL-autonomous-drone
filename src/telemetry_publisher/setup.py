from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'telemetry_publisher'

setup(
    name=package_name,
    version='0.0.3',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='drone008',
    maintainer_email='veetio@hotmail.fi',
    description='ROS2 package for the necessary nodes for communication between the other ros2 packages and offboard mission control in the drone ws.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		'telemetry_node = telemetry_publisher.telemetry_node:main',
        'odom_relay = telemetry_publisher.odom_relay:main',
        'fastlio_bridge_node = telemetry_publisher.fastlio_bridge_node:main',
        'cloud_registered_windowed = telemetry_publisher.cloud_registered_windowed:main',
        'mission_sequencer = telemetry_publisher.mission_sequencer:main',
        'flight_manager = telemetry_publisher.flight_manager:main',
        'single_goal_test = telemetry_publisher.single_goal_test:main',
        'simple_flight_test = telemetry_publisher.simple_flight_test:main',
	    'odometry_transformer = telemetry_publisher.odometry_transformer:main',
        'square_flight_test.py = telemetry_publisher.square_flight_test:main',
        ],
    },
)
