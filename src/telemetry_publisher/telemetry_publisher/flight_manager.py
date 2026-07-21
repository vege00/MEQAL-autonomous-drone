#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from mavros_msgs.msg import State, PositionTarget
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import Bool

handoff_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class FlightManager(Node):
    def __init__(self):
        super().__init__('flight_manager')

        self.declare_parameter('takeoff_alt', 5.0)
        self.declare_parameter('takeoff_tolerance', 0.3)
        self.takeoff_alt = self.get_parameter('takeoff_alt').value
        self.takeoff_tolerance = self.get_parameter('takeoff_tolerance').value

        self.phase = 'INIT'  # INIT -> ARMED_OFFBOARD -> TAKEOFF -> MISSION -> LANDING -> DONE
        self.launch_xy = None
        self.local_pos = None
        self.current_state = State()
        self.latest_pos_cmd = None

        # --- clients ---
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')

        # --- subscriptions ---
        self.sub_state = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, '/Odometry', self.odom_cb, 10)
        self.sub_pos_cmd = self.create_subscription(
            PositionCommand, '/drone_0_planning/pos_cmd', self.pos_cmd_cb, 10)
        self.sub_land_reached = self.create_subscription(
            Bool, '/mission/land_reached', self.land_reached_cb, handoff_qos)

        # --- publishers ---
        self.setpoint_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.pub_mission_enable = self.create_publisher(Bool, '/mission/enable', handoff_qos)

        # --- timers ---
        self.create_timer(0.05, self.setpoint_timer_cb)  # 20 Hz, required to keep OFFBOARD alive
        self.create_timer(0.5, self.state_machine_cb)

    # ---------- callbacks ----------

    def state_cb(self, msg: State):
        self.current_state = msg

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.local_pos = (p.x, p.y, p.z)
        if self.launch_xy is None:
            self.launch_xy = (p.x, p.y)

    def pos_cmd_cb(self, msg: PositionCommand):
        bad_status = (
            PositionCommand.TRAJECTROY_STATUS_ABORT,  # typo preserved from actual message def
            PositionCommand.TRAJECTORY_STATUS_ILLEGAL_START,
            PositionCommand.TRAJECTORY_STATUS_ILLEGAL_FINAL,
            PositionCommand.TRAJECTORY_STATUS_IMPOSSIBLE,
        )
        if msg.trajectory_flag in bad_status:
            self.get_logger().warn(
                f'PositionCommand reports bad trajectory status: {msg.trajectory_flag}. Holding position.')
            self.latest_pos_cmd = None
            return
        self.latest_pos_cmd = msg

    def land_reached_cb(self, msg: Bool):
        if msg.data and self.phase == 'MISSION':
            self.get_logger().info('Mission sequencer signaled land point reached.')
            self.begin_landing()

    # ---------- setpoint construction ----------

    def make_position_setpoint(self, x, y, z, yaw=0.0):
        pt = PositionTarget()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        pt.type_mask = (
            PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )
        pt.position.x, pt.position.y, pt.position.z = x, y, z
        pt.yaw = yaw
        return pt

    def pos_cmd_to_setpoint(self, msg: PositionCommand) -> PositionTarget:
        pt = PositionTarget()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        pt.type_mask = PositionTarget.IGNORE_YAW_RATE

        pt.position.x = msg.position.x
        pt.position.y = msg.position.y
        pt.position.z = msg.position.z

        pt.velocity.x = msg.velocity.x
        pt.velocity.y = msg.velocity.y
        pt.velocity.z = msg.velocity.z

        pt.acceleration_or_force.x = msg.acceleration.x
        pt.acceleration_or_force.y = msg.acceleration.y
        pt.acceleration_or_force.z = msg.acceleration.z

        pt.yaw = msg.yaw
        return pt

    # ---------- main loop ----------

    def setpoint_timer_cb(self):
        if self.phase in ('INIT', 'ARMED_OFFBOARD', 'TAKEOFF'):
            if self.launch_xy is None:
                return
            x, y = self.launch_xy
            self.setpoint_pub.publish(self.make_position_setpoint(x, y, self.takeoff_alt))

        elif self.phase == 'MISSION':
            if self.latest_pos_cmd is not None:
                self.setpoint_pub.publish(self.pos_cmd_to_setpoint(self.latest_pos_cmd))
            elif self.local_pos is not None:
                x, y, z = self.local_pos
                self.setpoint_pub.publish(self.make_position_setpoint(x, y, z))

        elif self.phase == 'LANDING':
            if self.local_pos is None or self.launch_xy is None:
                return
            x, y = self.launch_xy
            target_z = max(self.local_pos[2] - 0.1, 0.3)
            self.setpoint_pub.publish(self.make_position_setpoint(x, y, target_z))

    def state_machine_cb(self):
        if self.phase == 'INIT':
            if self.local_pos is None:
                return
            self.try_arm_and_offboard()

        elif self.phase == 'ARMED_OFFBOARD':
            if self.current_state.armed and self.current_state.mode == 'OFFBOARD':
                self.get_logger().info('Armed and in OFFBOARD — beginning takeoff climb.')
                self.phase = 'TAKEOFF'

        elif self.phase == 'TAKEOFF':
            if self.local_pos and abs(self.local_pos[2] - self.takeoff_alt) < self.takeoff_tolerance:
                self.get_logger().info('Takeoff altitude reached — enabling mission sequencer.')
                self.phase = 'MISSION'
                self.pub_mission_enable.publish(Bool(data=True))

        elif self.phase == 'LANDING':
            if self.local_pos and self.local_pos[2] < 0.4:
                self.get_logger().info('Near ground — switching to AUTO.LAND for touchdown + disarm.')
                self.call_land()
                self.phase = 'DONE'

    # ---------- service calls ----------

    def try_arm_and_offboard(self):
        if not self.arming_client.wait_for_service(timeout_sec=1.0):
            return
        if not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            return

        arm_req = CommandBool.Request()
        arm_req.value = True
        self.arming_client.call_async(arm_req)

        mode_req = SetMode.Request()
        mode_req.custom_mode = 'OFFBOARD'
        self.set_mode_client.call_async(mode_req)

        self.phase = 'ARMED_OFFBOARD'

    def call_land(self):
        if not self.land_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Land service unavailable')
            return
        req = CommandTOL.Request()
        req.yaw = 0.0
        req.latitude = 0.0
        req.longitude = 0.0
        req.altitude = 0.0
        self.land_client.call_async(req)

    def begin_landing(self):
        self.phase = 'LANDING'


def main():
    rclpy.init()
    node = FlightManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
