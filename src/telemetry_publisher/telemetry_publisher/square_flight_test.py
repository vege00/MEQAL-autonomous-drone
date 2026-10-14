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


class SquareFlightTest(Node):
    """
    Phase machine:
        INIT -> WARMUP -> ARMED_OFFBOARD -> TAKEOFF -> HOVER
             -> FLY_OUT <-> WP_HOLD   (one FLY_OUT per square corner)
             -> LANDING -> DONE
    Terminal failure phase: ABORTED

    Mission
    -------
    A square of side `square_side_m`, flown in the launch-heading frame:
    forward, then left (or right, `square_turn`), back, and return to the
    launch point, then land there. Each corner is sent to ego-planner as a
    separate goal; WP_HOLD pauses `waypoint_dwell_s` at each corner.

    Stale planner commands (the bug behind the crash)
    -------------------------------------------------
    ego-planner's traj_server keeps republishing the LAST point of a finished
    trajectory (velocity 0, same trajectory_id) at ~100 Hz forever. Those
    messages are fresh by timestamp, so a freshness-by-age check cannot see
    them. Before this version FLY_OUT forwarded that point to PX4, which
    flew in a straight line at it.

    Now a command is only used if
      1. its trajectory_id differs from the id that was being published at
         the moment the current goal was sent (stale_traj_id), and
      2. its position is within `cmd_max_jump_m` of the vehicle (a real
         trajectory starts where the drone is).
    If no valid command arrives, the setpoint is frozen at ONE point (so PX4
    brakes instead of coasting), the goal is re-sent a couple of times, and
    after `plan_start_timeout_s` (no trajectory yet) or `cmd_lost_timeout_s`
    (trajectory lost mid-leg) the script lands instead of waiting.
    This is what lets you re-run without restarting the planner stack: a
    leftover trajectory from the previous run can no longer be flown.

    Retained design notes
    ---------------------
    - WARMUP streams hold setpoints BEFORE requesting arm/OFFBOARD; PX4
      silently refuses OFFBOARD without an existing setpoint stream.
    - Only OFFBOARD setpoints get the vehicle off the ground; the FC-native
      takeoff service is deliberately not used.
    - Phase transitions happen ONLY in state_machine_cb; the 20 Hz setpoint
      timer only reads self.phase.
    - Every phase has a watchdog timeout (FLY_OUT's is PER LEG, because
      set_phase resets the phase clock). Loss of armed+OFFBOARD aborts.
    - The 20 Hz timer has its own callback group under a MultiThreadedExecutor
      so it cannot be starved; service calls are non-blocking.
    - ego-planner's goal callback ignores goal z and flies at its own fixed
      height (1.0 m in the pos_cmd you echoed), so takeoff_alt defaults to
      1.0 to avoid a climb/descent at handoff. Goal reached is HORIZONTAL only.
    """

    FLIGHT_PHASES = ('WARMUP', 'ARMED_OFFBOARD', 'TAKEOFF', 'HOVER',
                     'FLY_OUT', 'WP_HOLD', 'LANDING')
    OFFBOARD_PHASES = ('ARMED_OFFBOARD', 'TAKEOFF', 'HOVER', 'FLY_OUT', 'WP_HOLD')

    def __init__(self):
        super().__init__('square_flight_test')

        # --- mission ---
        self.declare_parameter('takeoff_alt', 2)       # = planner flight height
        self.declare_parameter('takeoff_tolerance', 0.3)
        self.declare_parameter('square_side_m', 5.0)
        self.declare_parameter('square_turn', 'left')    # 'left' or 'right'
        self.declare_parameter('waypoint_dwell_s', 1.0)
        self.declare_parameter('goal_tolerance_m', 0.5)
        self.declare_parameter('goal_topic', '/move_base_simple/goal')
        self.declare_parameter('goal_frame_id', 'world')
        self.declare_parameter('use_body_forward', True)  # False = square aligned to world +X

        # --- landing ---
        self.declare_parameter('descend_rate_mps', 0.25)
        self.declare_parameter('landing_slow_alt_m', 1.0)
        self.declare_parameter('landing_slow_rate_mps', 0.1)
        self.declare_parameter('land_trigger_alt_m', 0.4)

        # --- start-up ---
        self.declare_parameter('hover_settle_time_s', 2.0)
        self.declare_parameter('setpoint_warmup_time_s', 1.5)

        # --- stale-command / planner watchdogs ---
        self.declare_parameter('cmd_max_jump_m', 1.5)
        self.declare_parameter('plan_start_timeout_s', 5.0)
        self.declare_parameter('cmd_lost_timeout_s', 2.0)
        self.declare_parameter('goal_resend_interval_s', 2.0)
        self.declare_parameter('max_goal_resends', 2)

        # --- phase timeouts ---
        self.declare_parameter('odom_stale_timeout_s', 0.5)
        self.declare_parameter('armed_offboard_timeout_s', 10.0)
        self.declare_parameter('takeoff_timeout_s', 20.0)
        self.declare_parameter('hover_timeout_s', 15.0)
        self.declare_parameter('fly_out_timeout_s', 45.0)    # PER LEG
        self.declare_parameter('wp_hold_timeout_s', 15.0)
        self.declare_parameter('landing_timeout_s', 30.0)
        self.declare_parameter('service_retry_interval_s', 1.0)

        gp = lambda n: self.get_parameter(n).value
        self.takeoff_alt = gp('takeoff_alt')
        self.takeoff_tolerance = gp('takeoff_tolerance')
        self.square_side = gp('square_side_m')
        self.square_turn = gp('square_turn')
        self.waypoint_dwell = gp('waypoint_dwell_s')
        self.goal_tolerance = gp('goal_tolerance_m')
        self.goal_topic = gp('goal_topic')
        self.goal_frame_id = gp('goal_frame_id')
        self.use_body_forward = gp('use_body_forward')

        self.descend_rate = gp('descend_rate_mps')
        self.landing_slow_alt = gp('landing_slow_alt_m')
        self.landing_slow_rate = gp('landing_slow_rate_mps')
        self.land_trigger_alt = gp('land_trigger_alt_m')

        self.hover_settle_time = gp('hover_settle_time_s')
        self.setpoint_warmup_time = gp('setpoint_warmup_time_s')

        self.cmd_max_jump = gp('cmd_max_jump_m')
        self.plan_start_timeout = gp('plan_start_timeout_s')
        self.cmd_lost_timeout = gp('cmd_lost_timeout_s')
        self.goal_resend_interval = gp('goal_resend_interval_s')
        self.max_goal_resends = gp('max_goal_resends')

        self.odom_stale_timeout = gp('odom_stale_timeout_s')
        self.phase_timeouts = {
            'ARMED_OFFBOARD': gp('armed_offboard_timeout_s'),
            'TAKEOFF': gp('takeoff_timeout_s'),
            'HOVER': gp('hover_timeout_s'),
            'FLY_OUT': gp('fly_out_timeout_s'),
            'WP_HOLD': gp('wp_hold_timeout_s'),
            'LANDING': gp('landing_timeout_s'),
        }
        self.service_retry_interval = gp('service_retry_interval_s')

        if self.square_turn not in ('left', 'right'):
            self.get_logger().warn(f"square_turn='{self.square_turn}' invalid, using 'left'.")
            self.square_turn = 'left'

        self.phase = 'INIT'
        self.phase_entered_at = time.monotonic()
        self.launch_xy = None
        self.launch_z = None
        self.launch_yaw = None
        self.local_pos = None
        self.local_yaw = None
        self.landing_xy = None
        self.landing_yaw = None
        self.landing_ramp_z = None
        self.last_odom_time = None
        self.current_state = State()
        self.arm_offboard_requested = False
        self.last_service_retry_log = 0.0

        # planner interface state
        self.latest_pos_cmd = None
        self.last_seen_traj_id = None   # newest trajectory_id seen on pos_cmd (any content)
        self.stale_traj_id = None       # id being published when the current goal was sent
        self.hold_sp = None             # frozen (x, y, z, yaw) hold target
        self.waypoints = None
        self.wp_idx = -1
        self.test_goal = None
        self.leg_got_fresh = False
        self.last_fresh_cmd_t = 0.0
        self.last_goal_pub_t = 0.0
        self.goal_resends = 0
        self.last_sub_warn = 0.0

        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')

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
            f'Square flight test ready: side={self.square_side}m turn={self.square_turn} '
            f'takeoff_alt={self.takeoff_alt}m. Waiting for odometry.')

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
        # Track the id BEFORE any filtering: send_goal snapshots this as
        # "the trajectory that existed before the new goal".
        self.last_seen_traj_id = msg.trajectory_id
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
        if new_phase == self.phase:
            return
        self.get_logger().info(f'Phase: {self.phase} -> {new_phase}')
        self.hold_sp = None  # every phase starts with a fresh hold target
        pos, yaw = self.local_pos, self.local_yaw
        if new_phase == 'LANDING' and pos is not None:
            # Capture x, y, z-ramp seed and HEADING once, at entry.
            self.landing_xy = (pos[0], pos[1])
            self.landing_yaw = yaw
            self.landing_ramp_z = pos[2]
        elif new_phase == 'WP_HOLD' and pos is not None and self.test_goal is not None:
            # Dwell exactly at the corner (xy), at the current height/heading.
            self.hold_sp = (self.test_goal[0], self.test_goal[1], pos[2], yaw)
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

    # ---------- planner command validation ----------

    def fresh_pos_cmd(self):
        """Return the latest planner command if it belongs to a trajectory
        planned for the CURRENT goal and starts near the vehicle, else None.
        Reads latest_pos_cmd once: pos_cmd_cb runs on another thread."""
        msg = self.latest_pos_cmd
        pos = self.local_pos
        if msg is None or pos is None:
            return None
        if msg.trajectory_id == self.stale_traj_id:
            return None   # traj_server republishing the previous trajectory's end
        p = (msg.position.x, msg.position.y, msg.position.z)
        if math.dist(p, pos) > self.cmd_max_jump:
            return None   # a real trajectory starts where the drone is
        return msg

    # ---------- setpoint construction ----------

    def make_position_setpoint(self, x, y, z, yaw):
        """Position + yaw setpoint (velocity/accel/yaw-rate ignored)."""
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

    def publish_hold(self):
        """Hold ONE frozen point (captured on first use), so PX4 brakes to a
        stop. A hold that re-reads the measurement every tick just follows
        the drone's drift."""
        hold = self.hold_sp
        if hold is None:
            pos, yaw = self.local_pos, self.local_yaw
            hold = (pos[0], pos[1], pos[2], yaw)
            self.hold_sp = hold
        self.setpoint_pub.publish(self.make_position_setpoint(*hold))

    # ---------- fast loop: publish only, never change phase, never block ----------

    def setpoint_timer_cb(self):
        if self.phase in ('INIT', 'DONE', 'ABORTED'):
            return

        if self.odom_is_stale():
            return  # don't publish setpoints derived from dead data

        if self.phase in ('WARMUP', 'ARMED_OFFBOARD'):
            if self.local_pos is not None and self.launch_xy is not None:
                x, y = self.launch_xy
                self.setpoint_pub.publish(
                    self.make_position_setpoint(x, y, self.local_pos[2], self.launch_yaw))

        elif self.phase in ('TAKEOFF', 'HOVER'):
            x, y = self.launch_xy
            self.setpoint_pub.publish(
                self.make_position_setpoint(x, y, self.takeoff_alt, self.launch_yaw))

        elif self.phase == 'FLY_OUT':
            cmd = self.fresh_pos_cmd()
            if cmd is not None:
                self.hold_sp = None
                self.setpoint_pub.publish(self.pos_cmd_to_setpoint(cmd))
            else:
                self.publish_hold()

        elif self.phase == 'WP_HOLD':
            self.publish_hold()

        elif self.phase == 'LANDING':
            pos = self.local_pos
            x, y = self.landing_xy if self.landing_xy is not None else (pos[0], pos[1])
            yaw = self.landing_yaw if self.landing_yaw is not None else self.local_yaw
            dt = 0.05
            current_z = pos[2]
            rate = self.descend_rate if current_z > self.landing_slow_alt else self.landing_slow_rate
            if self.landing_ramp_z is None:
                self.landing_ramp_z = current_z
            self.landing_ramp_z = max(self.landing_ramp_z - rate * dt, 0.0)
            self.setpoint_pub.publish(self.make_position_setpoint(x, y, self.landing_ramp_z, yaw))

    # ---------- mission ----------

    def build_waypoints(self):
        """Square corners in the launch-heading frame -> odometry frame.
        Order: forward, forward+side, side, back to launch."""
        L = self.square_side
        s = 1.0 if self.square_turn == 'left' else -1.0
        corners = [(L, 0.0), (L, s * L), (0.0, s * L), (0.0, 0.0)]
        x0, y0 = self.launch_xy
        yaw = self.launch_yaw if (self.use_body_forward and self.launch_yaw is not None) else 0.0
        c, sn = math.cos(yaw), math.sin(yaw)
        wps = [(x0 + fx * c - fy * sn, y0 + fx * sn + fy * c, self.takeoff_alt)
               for fx, fy in corners]
        self.get_logger().info(
            'Square waypoints: ' + ', '.join(f'({x:.2f}, {y:.2f})' for x, y, _ in wps))
        return wps

    def publish_goal(self, wp):
        goal = PoseStamped()
        goal.header.frame_id = self.goal_frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = wp
        goal.pose.orientation.w = 1.0
        self.pub_goal.publish(goal)
        self.last_goal_pub_t = time.monotonic()

    def start_leg(self, idx: int) -> bool:
        """Send the goal for waypoint idx. Returns False (and sends nothing)
        if the planner is not listening yet; callers retry on the next tick."""
        if self.pub_goal.get_subscription_count() == 0:
            now = time.monotonic()
            if now - self.last_sub_warn > 2.0:
                self.get_logger().warn('No subscriber on the goal topic - planner not running? Waiting.')
                self.last_sub_warn = now
            return False
        if self.waypoints is None:
            self.waypoints = self.build_waypoints()
        self.wp_idx = idx
        self.test_goal = self.waypoints[idx]
        # ORDER MATTERS: snapshot the stale id, clear the cached command, THEN
        # publish, so anything that arrives afterwards is judged against it.
        self.stale_traj_id = self.last_seen_traj_id
        self.latest_pos_cmd = None
        self.leg_got_fresh = False
        self.goal_resends = 0
        self.last_fresh_cmd_t = time.monotonic()
        self.publish_goal(self.test_goal)
        gx, gy, gz = self.test_goal
        self.get_logger().info(
            f'Leg {idx + 1}/{len(self.waypoints)}: goal ({gx:.2f}, {gy:.2f}, {gz:.2f}), '
            f'ignoring trajectory_id={self.stale_traj_id} as stale.')
        return True

    # ---------- slow loop: the only place phase transitions happen ----------

    def state_machine_cb(self):
        if self.phase == 'INIT':
            if self.local_pos is None or self.odom_is_stale():
                return
            self.get_logger().info(
                f'Odometry valid - streaming setpoints for {self.setpoint_warmup_time}s before arming.')
            self.set_phase('WARMUP')
            return

        if self.phase in ('ABORTED', 'DONE'):
            return

        if self.phase in self.FLIGHT_PHASES and self.odom_is_stale():
            self.abort('Odometry feed went stale mid-phase.')
            return

        if self.phase == 'WARMUP':
            if self.time_in_phase() > self.setpoint_warmup_time:
                self.try_arm_and_offboard()
            return

        if self.phase in self.OFFBOARD_PHASES and self.offboard_lost():
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
                self.get_logger().info('Settle time elapsed - starting square.')
                if self.start_leg(0):
                    self.set_phase('FLY_OUT')
                # else: retry next tick; HOVER's own timeout lands us eventually

        elif self.phase == 'FLY_OUT':
            if self.test_goal is None or self.local_pos is None:
                return
            now = time.monotonic()

            # 1) Corner reached? (horizontal distance only)
            dx = self.local_pos[0] - self.test_goal[0]
            dy = self.local_pos[1] - self.test_goal[1]
            dist = math.hypot(dx, dy)
            if dist < self.goal_tolerance:
                self.get_logger().info(
                    f'Corner {self.wp_idx + 1}/{len(self.waypoints)} reached (dist={dist:.2f}m).')
                self.set_phase('WP_HOLD')
                return

            # 2) Is the planner actually giving us a trajectory for this goal?
            cmd = self.fresh_pos_cmd()
            if cmd is not None:
                if not self.leg_got_fresh:
                    self.get_logger().info(
                        f'Planner trajectory received (id={cmd.trajectory_id}).')
                self.leg_got_fresh = True
                self.last_fresh_cmd_t = now
            else:
                waited = now - self.last_fresh_cmd_t
                if not self.leg_got_fresh:
                    if (self.goal_resends < self.max_goal_resends and
                            now - self.last_goal_pub_t > self.goal_resend_interval):
                        self.goal_resends += 1
                        self.get_logger().warn(
                            f'No new trajectory {waited:.1f}s after goal - re-sending '
                            f'({self.goal_resends}/{self.max_goal_resends}).')
                        self.publish_goal(self.test_goal)
                    if waited > self.plan_start_timeout:
                        self.get_logger().error(
                            'Planner produced no new trajectory for this goal - holding and LANDING '
                            '(refusing to fly the previous trajectory).')
                        self.set_phase('LANDING')
                        return
                elif waited > self.cmd_lost_timeout:
                    self.get_logger().error(
                        'Planner command lost mid-leg - LANDING.')
                    self.set_phase('LANDING')
                    return

        elif self.phase == 'WP_HOLD':
            if self.time_in_phase() >= self.waypoint_dwell:
                if self.wp_idx + 1 >= len(self.waypoints):
                    self.get_logger().info('Square complete - landing at launch point.')
                    self.set_phase('LANDING')
                elif self.start_leg(self.wp_idx + 1):
                    self.set_phase('FLY_OUT')
                # else: planner not listening; WP_HOLD timeout lands us

        elif self.phase == 'LANDING':
            if self.local_pos and self.local_pos[2] < self.land_trigger_alt:
                self.get_logger().info('Near ground - switching to AUTO.LAND for touchdown + disarm.')
                self.call_land()
                self.set_phase('DONE')

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
    node = SquareFlightTest()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()