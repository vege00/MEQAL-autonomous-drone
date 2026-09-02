#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from mavros_msgs.msg import State, PositionTarget
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import PositionCommand

goal_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


class SimpleFlightTest(Node):
    def __init__(self):
        super().__init__('simple_flight_test')

        self.declare_parameter('takeoff_alt', 1.5)
        self.declare_parameter('takeoff_tolerance', 0.3)
        self.declare_parameter('fly_distance_m', 5.0)
        self.declare_parameter('goal_tolerance_m', 0.5)
        self.declare_parameter('goal_topic', '/move_base_simple/goal')
        self.declare_parameter('goal_frame_id', 'world')
        self.declare_parameter('use_body_forward', True)  # False = straight world +X

        self.takeoff_alt = self.get_parameter('takeoff_alt').value
        self.takeoff_tolerance = self.get_parameter('takeoff_tolerance').value
        self.fly_distance = self.get_parameter('fly_distance_m').value
        self.goal_tolerance = self.get_parameter('goal_tolerance_m').value
        self.goal_topic = self.get_parameter('goal_topic').value
        self.goal_frame_id = self.get_parameter('goal_frame_id').value
        self.use_body_forward = self.get_parameter('use_body_forward').value

        # phase: INIT -> ARMED_OFFBOARD -> TAKEOFF -> FLY_OUT -> LANDING -> DONE
        self.phase = 'INIT'
        self.launch_xy = None
        self.launch_yaw = None
        self.local_pos = None
        self.current_state = State()
        self.latest_pos_cmd = None
        self.test_goal = None
        self.goal_sent = False

        
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')

        self.sub_state = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, '/Odometry', self.odom_cb, 10)
        self.sub_pos_cmd = self.create_subscription(
            PositionCommand, '/drone_0_planning/pos_cmd', self.pos_cmd_cb, 10)

        self.setpoint_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.pub_goal = self.create_publisher(PoseStamped, self.goal_topic, goal_qos)

        self.create_timer(0.05, self.setpoint_timer_cb)  # 20 Hz, keeps OFFBOARD alive
        self.create_timer(0.5, self.state_machine_cb)

        self.get_logger().info(
            f'Simple flight test ready: takeoff_alt={self.takeoff_alt}m, '
            f'fly_distance={self.fly_distance}m. Waiting for odometry.')

    # ---------- callbacks ----------

    def state_cb(self, msg: State):
        self.current_state = msg

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.local_pos = (p.x, p.y, p.z)
        if self.launch_xy is None:
            self.launch_xy = (p.x, p.y)
            q = msg.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.launch_yaw = math.atan2(siny_cosp, cosy_cosp)
            self.get_logger().info(
                f'Launch position captured: {self.launch_xy}, yaw={math.degrees(self.launch_yaw):.1f}°')

    def pos_cmd_cb(self, msg: PositionCommand):
        bad_status = (
            PositionCommand.TRAJECTROY_STATUS_ABORT,
            PositionCommand.TRAJECTORY_STATUS_ILLEGAL_START,
            PositionCommand.TRAJECTORY_STATUS_ILLEGAL_FINAL,
            PositionCommand.TRAJECTORY_STATUS_IMPOSSIBLE,
        )
        if msg.trajectory_flag in bad_status:
            self.get_logger().warn(f'Bad trajectory status: {msg.trajectory_flag}. Holding.')
            self.latest_pos_cmd = None
            return
        self.latest_pos_cmd = msg

##Superfluous. not needed or called.
    def call_takeoff(self):
        if not self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Takeoff service unavailable')
            return
        req = CommandTOL.Request()
        req.yaw = 0.0
        req.latitude = 0.0
        req.longitude = 0.0
        req.altitude = self.takeoff_alt
        future = self.takeoff_client.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info(f'Takeoff result: {f.result().success}'))

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
        pt.type_mask = PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE #yaw masked off for testing
        pt.position.x, pt.position.y, pt.position.z = msg.position.x, msg.position.y, msg.position.z
        pt.velocity.x, pt.velocity.y, pt.velocity.z = msg.velocity.x, msg.velocity.y, msg.velocity.z
        pt.acceleration_or_force.x = msg.acceleration.x
        pt.acceleration_or_force.y = msg.acceleration.y
        pt.acceleration_or_force.z = msg.acceleration.z
        pt.yaw = msg.yaw
        return pt

    # ---------- main loop ----------

    

    def setpoint_timer_cb(self):
        if self.phase in ('INIT', 'ARMED_OFFBOARD', 'TAKEOFF'):
            if self.launch_xy is None or self.local_pos is None:
                return
            x, y = self.launch_xy
            climb_rate = 0.3  # m/s, tune conservatively
            dt = 0.05  # matches this timer's period
            current_z = self.local_pos[2]
            target_z = min(current_z + climb_rate * dt, self.takeoff_alt)
            self.setpoint_pub.publish(self.make_position_setpoint(x, y, target_z))

        elif self.phase == 'FLY_OUT':
            if self.latest_pos_cmd is not None:
                self.setpoint_pub.publish(self.pos_cmd_to_setpoint(self.latest_pos_cmd))
            elif self.local_pos is not None:
                x, y, z = self.local_pos
                self.setpoint_pub.publish(self.make_position_setpoint(x, y, z))

        elif self.phase == 'LANDING':
            if self.local_pos is None or self.launch_xy is None:
                return
            x, y = self.launch_xy  # return toward launch xy, descending
            target_z = max(self.local_pos[2] - 0.1, 0.3)
            self.setpoint_pub.publish(self.make_position_setpoint(x, y, target_z))

        elif self.phase == 'ARMED_OFFBOARD':
            if self.current_state.armed and self.current_state.mode == 'OFFBOARD':
                self.get_logger().info('Armed and in OFFBOARD — issuing takeoff command.')
                self.call_takeoff()
                self.phase = 'TAKEOFF'

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
                self.get_logger().info('Takeoff altitude reached — computing test goal.')
                self.compute_and_send_test_goal()
                self.phase = 'FLY_OUT'

        elif self.phase == 'FLY_OUT':
            if self.test_goal is None or self.local_pos is None:
                return
            dx = self.local_pos[0] - self.test_goal[0]
            dy = self.local_pos[1] - self.test_goal[1]
            dz = self.local_pos[2] - self.test_goal[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < self.goal_tolerance:
                self.get_logger().info(f'Test goal reached (dist={dist:.2f}m). Beginning landing.')
                self.phase = 'LANDING'

        elif self.phase == 'LANDING':
            if self.local_pos and self.local_pos[2] < 0.4:
                self.get_logger().info('Near ground — switching to AUTO.LAND for touchdown + disarm.')
                self.call_land()
                self.phase = 'DONE'

    def compute_and_send_test_goal(self):
        x0, y0 = self.launch_xy
        z = self.takeoff_alt
        if self.use_body_forward and self.launch_yaw is not None:
            gx = x0 + self.fly_distance * math.cos(self.launch_yaw)
            gy = y0 + self.fly_distance * math.sin(self.launch_yaw)
        else:
            gx = x0 + self.fly_distance
            gy = y0

        self.test_goal = (gx, gy, z)

        goal = PoseStamped()
        goal.header.frame_id = self.goal_frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = gx, gy, z
        goal.pose.orientation.w = 1.0
        self.pub_goal.publish(goal)
        self.goal_sent = True
        self.get_logger().info(f'Sent test goal: ({gx:.2f}, {gy:.2f}, {z:.2f})')

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
        req.yaw, req.latitude, req.longitude, req.altitude = 0.0, 0.0, 0.0, 0.0
        self.land_client.call_async(req)


def main():
    rclpy.init()
    node = SimpleFlightTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
