Project repository for my bachelor's thesis.
The end goal of this project is to make an autonomous drone platform for use by the [MEQAL](https://www.frostbit.fi/en/portfolio/meqal/) project and related parties.
This project builds upon [Samy Warnant's drone 008 project](https://github.com/SamyWarnants/Samy-Warnants-Drone-Project-008) by utilising its frame and hardware stack.
<img width="5792" height="4344" alt="rooni" src="https://github.com/user-attachments/assets/51eaf94a-e130-4770-96e9-f60f53c9c9cb" />

The greater end goal is to design and program a drone capable of fully autonomous mission completion within operator line of sight.
The drone can be equipped with any necessary sensor payload for surveying and mapping methane emissions and other quantifiable factors.



## NOTICE
This project builds on several open-source ROS 2 packages located in /src. Each retains its own license. 
See "3rd party licenses.md" and each package's LICENSE files for more details.

## Project /SRC folder Tree view
src
├── ego-planner-swarm
│   └── src
│       ├── planner
│       └── uav_simulator
├── FAST_LIO_SLAM_ros2
│   ├── config
│   ├── doc
│   │   └── results
│   ├── include
│   │   ├── ikd-Tree
│   │   └── IKFoM_toolkit
│   ├── launch
│   ├── Log
│   ├── msg
│   ├── PCD
│   ├── rviz
│   ├── rviz_cfg
│   └── src
├── livox_ros_driver2
│   ├── 3rdparty
│   │   └── rapidjson
│   ├── cmake
│   ├── config
│   ├── launch_ROS1
│   ├── launch_ROS2
│   ├── msg
│   └── src
│       ├── call_back
│       ├── comm
│       ├── include
│       └── parse_cfg_file
├── mavlink
│   ├── component_metadata
│   ├── doc
│   ├── examples
│   │   └── c
│   ├── message_definitions
│   │   └── v1.0
│   ├── pymavlink
│   │   ├── APM_Mavtest
│   │   ├── dfindexer
│   │   ├── dialects
│   │   ├── examples
│   │   ├── generator
│   │   ├── __pycache__
│   │   ├── tests
│   │   └── tools
│   └── scripts
├── mavros
│   ├── docs
│   │   └── examples
│   ├── libmavconn
│   │   ├── cmake
│   │   ├── include
│   │   ├── src
│   │   └── test
│   ├── mavros
│   │   ├── include
│   │   ├── launch
│   │   ├── mavros
│   │   ├── scripts
│   │   ├── src
│   │   └── test
│   ├── mavros_examples
│   │   └── mavros_examples
│   ├── mavros_extras
│   │   ├── include
│   │   ├── launch
│   │   └── src
│   ├── mavros_msgs
│   │   ├── include
│   │   ├── msg
│   │   └── srv
│   ├── test_mavros
│   │   ├── include
│   │   ├── launch
│   │   ├── scripts
│   │   └── sitl_test
│   └── tools
└── telemetry_publisher
    ├── launch
    ├── resource
    ├── telemetry_publisher
    └── test

#This project is still under development!
There are persisting problems with this project.
Current mission handling is unfinished and the current missions are hard coded.
The launch file for this drone is unfinished and does not work.
