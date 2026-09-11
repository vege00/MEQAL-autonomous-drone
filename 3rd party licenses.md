# Third-Party Licenses

This project builds on several open-source ROS 2 packages located in `src/`.
Each retains its own license, listed below. Only `telemetry_publisher/` is
original code written for this project (see main repo LICENSE for its terms).

---

## FAST_LIO_SLAM_ros2
- **Source:** FAST-LIO / FAST-LIO-SLAM, ROS 2 port
- **License:** GNU General Public License v2.0 (GPL-2.0)
- **Copyright:** HKU-MARS Lab and contributors
- **Notes:** GPL-2.0 is copyleft. If this package is compiled into the same
  binary/executable as other project code and that binary is distributed,
  GPL-2.0 terms generally extend to the combined work.

## ego-planner-swarm
- **Source:** https://github.com/ZJU-FAST-Lab/ego-planner-swarm
- **License:** GNU General Public License v3.0 (GPL-3.0)
- **Copyright:** ZJU-FAST-Lab (Xin Zhou, Fei Gao, et al.)
- **Notes:** GPL-3.0 is copyleft, with terms similar to the note above.
  Commercial licensing inquiries are directed to the original maintainers.

## livox_ros_driver2
- **Source:** https://github.com/Livox-SDK/livox_ros_driver2
- **License:** MIT License (primary)
- **Copyright:** © 2022 Livox
- **Notes:** This package also bundles/uses, unmodified, third-party code
  under other permissive licenses:
  - RapidJSON — MIT License
  - ROS2-rclcpp interfaces — Apache License 2.0
  - ROS interfaces — BSD 3-Clause License

## MAVLink
- **Source:** https://github.com/mavlink/mavlink
- **License:** Dual — the code *generator* is licensed under LGPL-3.0;
  the *generated* protocol/message headers are distributed under the
  MIT License.
- **Notes:** As long as you only use the generated MAVLink headers/libraries
  (not modify and redistribute the generator itself), MIT terms apply to
  what you're actually shipping.

## MAVROS
- **Source:** https://github.com/mavlink/mavros
- **License:** Triple-licensed — BSD 3-Clause, GPL-3.0, or LGPL-3.0
  (your choice of any one, per the project's LICENSE.md)
- **Copyright:** Vladimir Ermakov and contributors
- **Notes:** Since this is offered under multiple options, using it under
  the BSD-3-Clause term is the least restrictive choice if you want to
  avoid copyleft obligations elsewhere in the project.

## telemetry_publisher
- **Source:** Original code written for this project.
- **License:** Lisenced under MIT

---

### A note on combining GPL and non-GPL code

Two dependencies above (FAST_LIO_SLAM_ros2, ego-planner-swarm) are GPL
licensed. ROS 2 nodes that communicate only via topics/services/actions
(separate processes, not linked into one binary) are commonly treated as
separate works, which avoids triggering GPL's copyleft on your other code.
Directly linking GPL code into `telemetry_publisher` or another binary is
different — that combined binary would generally need to be GPL-compatible
if you distribute it. This is general information, not legal advice; consult
a lawyer if licensing terms matter for how you plan to release/sell this
project.