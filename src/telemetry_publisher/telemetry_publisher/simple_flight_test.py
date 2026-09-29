#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from mavros_msgs.msg import State, PositionTarget
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import PositionCommand

goal_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_from_quat(q) -> float:
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SimpleFlightTest(Node):
    """
    Phase machine: INIT -> WARMUP -> ARMED_OFFBOARD -> TAKEOFF -> HOVER -> FLY_OUT -> LANDING -> DONE
    Terminal failure phase: ABORTED

    Changes vs previous version
    ---------------------------
    - TAKEOFF is now a single, constant position setpoint at takeoff_alt.
      The liftoff velocity burst and the position ramp are gone. The old
      logic switched control mode whenever measured height crossed
      liftoff_height_m (0.4 m) and RESET the ramp every time the vehicle
      dipped back below it, which produces a limit cycle right at 0.4 m.
      Climb speed is now limited by the FC (PX4 MPC_Z_VEL_MAX_UP), not here.
    - Every setpoint carries an explicit yaw. Hold phases use the launch
      heading; LANDING captures the heading at phase entry. Previously all
      hold/landing setpoints used yaw=0.0 (world +X), which commanded a
      rotation at arming and a snap back to 0 when LANDING began.
    - FLY_OUT goal-reached check is HORIZONTAL only. ego-planner's manual
      goal callback ignores the goal z (see compute_and_send_test_goal), so a
      3D check could never succeed when takeoff_alt differs from the
      planner's fixed flight height.

    Retained design notes
    ---------------------
    - WARMUP streams hold setpoints BEFORE requesting arm/OFFBOARD; PX4
      silently refuses OFFBOARD without an existing setpoint stream.
    - Only OFFBOARD setpoints get the vehicle off the ground; the FC-native
      takeoff service is deliberately not used.
    - Phase transitions happen ONLY in state_machine_cb; the 20 Hz setpoint
      timer only reads self.phase.
    - Every phase has a watchdog timeout. Loss of armed+OFFBOARD mid-flight
      aborts rather than fighting whoever took over.
    - The 20 Hz timer has its own callback group under a MultiThreadedExecutor
      so it cannot be starved; service calls are non-blocking.
    """

    def __init__(self):
        super().__init__('simple_flight_test')

        self.declare_parameter('takeoff_alt', 1.5)
        self.declare_parameter('takeoff_tolerance', 0.3)
        self.declare_parameter('fly_distance_m', 5.0)
        self.declare_parameter('goal_tolerance_m', 0.5)
        self.declare_parameter('goal_topic', '/move_base_simple/goal')
        self.declare_parameter('goal_frame_id', 'world')
        self.declare_parameter('use_body_forward', True)  # False = straight world +X

        self.declare_parameter('descend_rate_mps', 0.25)
        self.declare_parameter('landing_slow_alt_m', 1.0)   # below this, descend slower
        self.declare_parameter('landing_slow_rate_mps', 0.1)
        self.declare_parameter('land_trigger_alt_m', 0.4)   # switch to FC land below this

        # Hold at takeoff altitude before handing control to the planner.
        self.declare_parameter('hover_settle_time_s', 2.0)
        # Stream hold setpoints this long BEFORE requesting arm + OFFBOARD.
        self.declare_parameter('setpoint_warmup_time_s', 1.5)

        self.declare_parameter('odom_stale_timeout_s', 0.5)
        self.declare_parameter('armed_offboard_timeout_s', 10.0)
        self.declare_parameter('takeoff_timeout_s', 20.0)
        self.declare_parameter('hover_timeout_s', 15.0)
        self.declare_parameter('fly_out_timeout_s', 60.0)
        self.declare_parameter('landing_timeout_s', 30.0)
        self.declare_parameter('service_retry_interval_s', 1.0)

        self.takeoff_alt = self.get_parameter('takeoff_alt').value
        self.takeoff_tolerance = self.get_parameter('takeoff_tolerance').value
        self.fly_distance = self.get_parameter('fly_distance_m').value
        self.goal_tolerance = self.get_parameter('goal_tolerance_m').value
        self.goal_topic = self.get_parameter('goal_topic').value
        self.goal_frame_id = self.get_parameter('goal_frame_id').value
        self.use_body_forward = self.get_parameter('use_body_forward').value

        self.descend_rate = self.get_parameter('descend_rate_mps').value
        self.landing_slow_alt = self.get_parameter('landing_slow_alt_m').value
        self.landing_slow_rate = self.get_parameter('landing_slow_rate_mps').value
        self.land_trigger_alt = self.get_parameter('land_trigger_alt_m').value

        self.hover_settle_time = self.get_parameter('hover_settle_time_s').value
        self.setpoint_warmup_time = self.get_parameter('setpoint_warmup_time_s').value
        self.odom_stale_timeout = self.get_parameter('odom_stale_timeout_s').value
        self.phase_timeouts = {
            'ARMED_OFFBOARD': self.get_parameter('armed_offboard_timeout_s').value,
            'TAKEOFF': self.get_parameter('takeoff_timeout_s').value,
            'HOVER': self.get_parameter('hover_timeout_s').value,
            'FLY_OUT': self.get_parameter('fly_out_timeout_s').value,
            'LANDING': self.get_parameter('landing_timeout_s').value,
        }
        self.service_retry_interval = self.get_parameter('service_retry_interval_s').value

        self.phase = 'INIT'
        self.phase_entered_at = time.monotonic()
        self.launch_xy = None
        self.launch_z = None
        self.launch_yaw = None
        self.local_pos = None
        self.local_yaw = None
        self.landing_xy = None       # captured once at LANDING entry
        self.landing_yaw = None      # captured once at LANDING entry
        self.landing_ramp_z = None   # descent ramp target, seeded at LANDING entry
        self.last_odom_time = None
        self.current_state = State()
        self.latest_pos_cmd = None
        self.test_goal = None
        self.goal_sent = False
        self.arm_offboard_requested = False
        self.last_service_retry_log = 0.0

        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')

        # Confirm services exist ONCE, up front. wait_for_service() blocks, so
        # it must never be called from a timer callback.
        for client, name in (
            (self.arming_client, 'arming'),
            (self.set_mode_client, 'set_mode'),
            (self.land_client, 'land'),
        ):
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error(
                    f'{name} service not available at startup - it may come up '
                    f'later; try_arm_and_offboard will keep checking non-blockingly.')

        self.sub_state = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, '/Odometry_corrected', self.odom_cb, 10)
        self.sub_pos_cmd = self.create_subscription(
            PositionCommand, '/drone_0_planning/pos_cmd', self.pos_cmd_cb, 10)

        self.setpoint_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.pub_goal = self.create_publisher(PoseStamped, self.goal_topic, goal_qos)

        self.setpoint_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(0.05, self.setpoint_timer_cb,
                          callback_group=self.setpoint_cb_group)  # 20 Hz, keeps OFFBOARD alive
        self.create_timer(0.5, self.state_machine_cb)

        self.get_logger().info(
            f'Simple flight test ready: takeoff_alt={self.takeoff_alt}m, '
            f'fly_distance={self.fly_distance}m, hover_settle_time={self.hover_settle_time}s. '
            f'Waiting for odometry.')

    # ---------- callbacks ----------

    def state_cb(self, msg: State):
        self.current_state = msg

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.local_pos = (p.x, p.y, p.z)
        self.local_yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.last_odom_time = time.monotonic()
        if self.launch_xy is None:
            self.launch_xy = (p.x, p.y)
            self.launch_z = p.z
            self.launch_yaw = self.local_yaw
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

    # ---------- helpers ----------

    def odom_is_stale(self) -> bool:
        if self.last_odom_time is None:
            return True
        return (time.monotonic() - self.last_odom_time) > self.odom_stale_timeout

    def set_phase(self, new_phase: str):
        if new_phase != self.phase:
            self.get_logger().info(f'Phase: {self.phase} -> {new_phase}')
            if new_phase == 'LANDING' and self.local_pos is not None:
                # Capture x, y, z-ramp seed and HEADING once, at entry, so
                # LANDING is a clean vertical drop with no rotation and no
                # per-tick re-tracking of the measurement.
                self.landing_xy = (self.local_pos[0], self.local_pos[1])
                self.landing_yaw = self.local_yaw
                self.landing_ramp_z = self.local_pos[2]
            self.phase = new_phase
            self.phase_entered_at = time.monotonic()

    def time_in_phase(self) -> float:
        return time.monotonic() - self.phase_entered_at

    def phase_timed_out(self) -> bool:
        limit = self.phase_timeouts.get(self.phase)
        return limit is not None and self.time_in_phase() > limit

    def offboard_lost(self) -> bool:
        return not (self.current_state.armed and self.current_state.mode == 'OFFBOARD')

    def abort(self, reason: str):
        self.get_logger().error(f'ABORT ({self.phase}): {reason}')
        self.set_phase('ABORTED')

    # ---------- setpoint construction ----------

    def make_position_setpoint(self, x, y, z, yaw):
        """Position + yaw setpoint (velocity/accel/yaw-rate ignored).
        yaw is required on purpose: a silent default of 0.0 (world +X) is
        what caused the unintended 90 deg swings."""
        pt = PositionTarget()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = 'map'
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
        pt.header.frame_id = 'map'
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        pt.type_mask = PositionTarget.IGNORE_YAW_RATE  # pos + vel + accel + yaw all used
        pt.position.x, pt.position.y, pt.position.z = msg.position.x, msg.position.y, msg.position.z
        pt.velocity.x, pt.velocity.y, pt.velocity.z = msg.velocity.x, msg.velocity.y, msg.velocity.z
        pt.acceleration_or_force.x = msg.acceleration.x
        pt.acceleration_or_force.y = msg.acceleration.y
        pt.acceleration_or_force.z = msg.acceleration.z
        pt.yaw = msg.yaw
        return pt

    # ---------- fast loop: publish only, never change phase, never block ----------

    def setpoint_timer_cb(self):
        if self.phase in ('INIT', 'DONE', 'ABORTED'):
            return

        if self.odom_is_stale():
            # Don't publish setpoints derived from dead data.
            return

        if self.phase in ('WARMUP', 'ARMED_OFFBOARD'):
            # Hold launch xy / launch heading, at the current altitude (no climb).
            if self.local_pos is not None and self.launch_xy is not None:
                x, y = self.launch_xy
                self.setpoint_pub.publish(
                    self.make_position_setpoint(x, y, self.local_pos[2], self.launch_yaw))

        elif self.phase in ('TAKEOFF', 'HOVER'):
            # Constant target: launch xy at takeoff_alt. Speed of the climb is
            # governed by the FC's position controller limits.
            x, y = self.launch_xy
            self.setpoint_pub.publish(
                self.make_position_setpoint(x, y, self.takeoff_alt, self.launch_yaw))

        elif self.phase == 'FLY_OUT':
            if self.latest_pos_cmd is not None:
                self.setpoint_pub.publish(self.pos_cmd_to_setpoint(self.latest_pos_cmd))
            else:
                # No (valid) planner command: hold where we are, heading included.
                x, y, z = self.local_pos
                self.setpoint_pub.publish(self.make_position_setpoint(x, y, z, self.local_yaw))

        elif self.phase == 'LANDING':
            # Descend straight down at the xy and heading captured at entry.
            x, y = self.landing_xy if self.landing_xy is not None else (self.local_pos[0], self.local_pos[1])
            yaw = self.landing_yaw if self.landing_yaw is not None else self.local_yaw
            dt = 0.05
            current_z = self.local_pos[2]
            rate = self.descend_rate if current_z > self.landing_slow_alt else self.landing_slow_rate
            if self.landing_ramp_z is None:
                self.landing_ramp_z = current_z
            self.landing_ramp_z = max(self.landing_ramp_z - rate * dt, 0.0)
            self.setpoint_pub.publish(self.make_position_setpoint(x, y, self.landing_ramp_z, yaw))

    # ---------- slow loop: the only place phase transitions happen ----------

    def state_machine_cb(self):
        if self.phase == 'INIT':
            if self.local_pos is None or self.odom_is_stale():
                return
            self.get_logger().info(
                f'Odometry valid - streaming setpoints for {self.setpoint_warmup_time}s before arming.')
            self.set_phase('WARMUP')
            return

        if self.phase == 'ABORTED' or self.phase == 'DONE':
            return

        if self.phase in ('WARMUP', 'ARMED_OFFBOARD', 'TAKEOFF', 'HOVER', 'FLY_OUT', 'LANDING') and self.odom_is_stale():
            self.abort('Odometry feed went stale mid-phase.')
            return

        if self.phase == 'WARMUP':
            if self.time_in_phase() > self.setpoint_warmup_time:
                self.try_arm_and_offboard()
            return

        if self.phase in ('ARMED_OFFBOARD', 'TAKEOFF', 'HOVER', 'FLY_OUT') and self.offboard_lost():
            if self.phase != 'ARMED_OFFBOARD' or self.time_in_phase() > 2.0:
                self.abort('Lost armed+OFFBOARD state unexpectedly.')
                return

        if self.phase_timed_out():
            if self.phase == 'LANDING':
                self.abort('Landing did not complete before timeout - check vehicle immediately.')
            else:
                self.get_logger().warn(f'{self.phase} timed out - forcing LANDING.')
                self.set_phase('LANDING')
            return

        if self.phase == 'ARMED_OFFBOARD':
            if self.current_state.armed and self.current_state.mode == 'OFFBOARD':
                self.get_logger().info(f'Armed and in OFFBOARD - climbing to {self.takeoff_alt}m.')
                self.set_phase('TAKEOFF')
            elif not self.arm_offboard_requested:
                self.try_arm_and_offboard()

        elif self.phase == 'TAKEOFF':
            if self.local_pos and abs(self.local_pos[2] - self.takeoff_alt) < self.takeoff_tolerance:
                self.get_logger().info(
                    f'Takeoff altitude reached - holding for {self.hover_settle_time}s to settle.')
                self.set_phase('HOVER')

        elif self.phase == 'HOVER':
            if self.time_in_phase() > self.hover_settle_time:
                self.get_logger().info('Settle time elapsed - computing test goal.')
                self.compute_and_send_test_goal()
                self.set_phase('FLY_OUT')

        elif self.phase == 'FLY_OUT':
            if self.test_goal is None or self.local_pos is None:
                return
            # Horizontal distance only - the planner does not honor goal z.
            dx = self.local_pos[0] - self.test_goal[0]
            dy = self.local_pos[1] - self.test_goal[1]
            dist = math.hypot(dx, dy)
            if dist < self.goal_tolerance:
                self.get_logger().info(f'Test goal reached (horizontal dist={dist:.2f}m). Beginning landing.')
                self.set_phase('LANDING')

        elif self.phase == 'LANDING':
            if self.local_pos and self.local_pos[2] < self.land_trigger_alt:
                self.get_logger().info('Near ground - switching to AUTO.LAND for touchdown + disarm.')
                self.call_land()
                self.set_phase('DONE')

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

        # NOTE: upstream ego-planner's waypointCallback builds the target as
        # (goal.x, goal.y, 1.0) - the z below is ignored there. Set takeoff_alt
        # to the planner's flight height (1.0 m) to avoid a descent at handoff.
        goal = PoseStamped()
        goal.header.frame_id = self.goal_frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = gx, gy, z
        goal.pose.orientation.w = 1.0
        self.pub_goal.publish(goal)
        self.goal_sent = True
        self.get_logger().info(f'Sent test goal: ({gx:.2f}, {gy:.2f}, {z:.2f})')

    # ---------- service calls (all non-blocking) ----------

    def try_arm_and_offboard(self):
        if self.arm_offboard_requested:
            return

        if not self.arming_client.service_is_ready() or not self.set_mode_client.service_is_ready():
            now = time.monotonic()
            if now - self.last_service_retry_log > self.service_retry_interval:
                self.get_logger().warn(
                    'Arm/set_mode service not ready yet - will keep retrying non-blockingly.')
                self.last_service_retry_log = now
            return

        self.arm_offboard_requested = True

        arm_req = CommandBool.Request()
        arm_req.value = True
        arm_future = self.arming_client.call_async(arm_req)
        arm_future.add_done_callback(self._on_arm_result)

        mode_req = SetMode.Request()
        mode_req.custom_mode = 'OFFBOARD'
        mode_future = self.set_mode_client.call_async(mode_req)
        mode_future.add_done_callback(self._on_mode_result)

        self.set_phase('ARMED_OFFBOARD')

    def _on_arm_result(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('Arming request accepted.')
            else:
                self.get_logger().error('Arming request rejected by FCU.')
                self.arm_offboard_requested = False
        except Exception as e:
            self.get_logger().error(f'Arming service call failed: {e}')
            self.arm_offboard_requested = False

    def _on_mode_result(self, future):
        try:
            result = future.result()
            if result.mode_sent:
                self.get_logger().info('OFFBOARD mode request sent.')
            else:
                self.get_logger().error('OFFBOARD mode request rejected by FCU.')
                self.arm_offboard_requested = False
        except Exception as e:
            self.get_logger().error(f'Set-mode service call failed: {e}')
            self.arm_offboard_requested = False

    def call_land(self):
        if not self.land_client.service_is_ready():
            self.get_logger().error(
                'Land service not ready - vehicle will remain in LANDING setpoint mode.')
            self.set_phase('LANDING')
            return
        req = CommandTOL.Request()
        req.yaw, req.latitude, req.longitude, req.altitude = 0.0, 0.0, 0.0, 0.0
        future = self.land_client.call_async(req)
        future.add_done_callback(self._on_land_result)

    def _on_land_result(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('Land command accepted.')
            else:
                self.get_logger().error('Land command rejected by FCU - monitor vehicle manually.')
        except Exception as e:
            self.get_logger().error(f'Land service call failed: {e}')


def main():
    rclpy.init()
    node = SimpleFlightTest()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()