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


class SimpleFlightTest(Node):
    """
    Phase machine: INIT -> WARMUP -> ARMED_OFFBOARD -> TAKEOFF -> HOVER -> FLY_OUT -> LANDING -> DONE
    Terminal failure phase: ABORTED

    Design notes:
    - WARMUP exists because PX4 will refuse (silently, from our point of view -
      the mavros service call still reports mode_sent=true) an OFFBOARD mode
      switch that arrives before it has already been receiving a setpoint
      stream. We must publish hold-position setpoints for a bit BEFORE asking
      to arm/switch to OFFBOARD, not just after.
    - There is exactly ONE mechanism responsible for getting the vehicle off the
      ground: OFFBOARD position setpoints, published continuously at 20 Hz.
      We deliberately do NOT call the FC-native /mavros/cmd/takeoff service,
      because having two independent authorities (our own climb setpoints AND
      the FC's internal takeoff routine) fighting for control of the trajectory
      is exactly what caused ambiguous behavior in the previous version.
    - HOVER exists so the vehicle settles at takeoff altitude (damps out any
      residual climb velocity / oscillation) before authority is handed to the
      external trajectory planner. Handing off while still moving vertically is
      what causes the planner to start from a stale state estimate and drift.
      The goal is only computed and sent once HOVER's settle time has elapsed.
    - Phase transitions happen ONLY inside state_machine_cb. The fast setpoint
      timer never mutates self.phase; it only reads it.
    - Every phase has a watchdog timeout. If we get stuck, we abort toward
      LANDING (or straight to ABORTED if landing itself is unsafe/stale).
    - If OFFBOARD or arming drops unexpectedly during flight (RC override,
      safety trip, FC-side rejection, etc.), we stop publishing setpoints and
      go to ABORTED rather than continuing to fight whatever took control.

    Executor note:
    - The node runs under a MultiThreadedExecutor with the 20 Hz setpoint timer
      on its own callback group. This guarantees the setpoint stream can never
      be starved by a slow/blocking callback elsewhere in the node (service
      waits, future callbacks, future bugs). Service readiness is also checked
      ONCE at startup (wait_for_service) rather than inside the timing-critical
      WARMUP -> ARMED_OFFBOARD transition; try_arm_and_offboard() only ever
      does a non-blocking service_is_ready() check, so it can never stall the
      executor even on the single-threaded fallback.
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

        self.declare_parameter('climb_rate_mps', 0.3)

        # Liftoff burst: a firmer, VELOCITY-commanded climb used only for the
        # first liftoff_height_m of altitude gain. Commanding position (as the
        # rest of TAKEOFF does) makes PX4 derive climb velocity from a tiny,
        # self-shrinking position error each tick - fine once airborne, but it
        # gives too little thrust authority to reliably break ground contact,
        # which is what causes an uneven, one-side-first liftoff that tips
        # over instead of climbing cleanly. Commanding velocity directly here
        # avoids that self-limiting effect for exactly the window where it
        # matters, then TAKEOFF falls back to the gentler position-based climb
        # once clear of the ground.
        self.declare_parameter('liftoff_speed_mps', 0.8)
        self.declare_parameter('liftoff_height_m', 0.4)

        self.declare_parameter('descend_rate_mps', 0.25)
        self.declare_parameter('landing_slow_alt_m', 1.0)   # below this, descend slower
        self.declare_parameter('landing_slow_rate_mps', 0.1)
        self.declare_parameter('land_trigger_alt_m', 0.4)   # switch to FC land below this

        # How long to hold position at takeoff altitude before handing control
        # to the planner. Tune this up if you still see drift at FLY_OUT entry.
        self.declare_parameter('hover_settle_time_s', 2.0)

        # How long to stream hold-position setpoints BEFORE requesting arm and
        # OFFBOARD. PX4 requires an existing setpoint stream to accept the
        # OFFBOARD switch; without this, the switch is silently refused.
        self.declare_parameter('setpoint_warmup_time_s', 1.5)

        self.declare_parameter('odom_stale_timeout_s', 0.5)
        self.declare_parameter('armed_offboard_timeout_s', 10.0)
        self.declare_parameter('takeoff_timeout_s', 20.0)
        self.declare_parameter('hover_timeout_s', 15.0)
        self.declare_parameter('fly_out_timeout_s', 60.0)
        self.declare_parameter('landing_timeout_s', 30.0)

        # How often to retry starting arm+OFFBOARD if the services weren't
        # ready yet the first time WARMUP elapsed (non-blocking retry).
        self.declare_parameter('service_retry_interval_s', 1.0)

        self.takeoff_alt = self.get_parameter('takeoff_alt').value
        self.takeoff_tolerance = self.get_parameter('takeoff_tolerance').value
        self.fly_distance = self.get_parameter('fly_distance_m').value
        self.goal_tolerance = self.get_parameter('goal_tolerance_m').value
        self.goal_topic = self.get_parameter('goal_topic').value
        self.goal_frame_id = self.get_parameter('goal_frame_id').value
        self.use_body_forward = self.get_parameter('use_body_forward').value

        self.climb_rate = self.get_parameter('climb_rate_mps').value
        self.liftoff_speed = self.get_parameter('liftoff_speed_mps').value
        self.liftoff_height = self.get_parameter('liftoff_height_m').value
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
        self.landing_xy = None  # captured once, at LANDING entry - see set_phase-adjacent logic below
        self.local_pos = None
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

        # Confirm services exist ONCE, up front, well before the timing-critical
        # WARMUP -> ARMED_OFFBOARD transition. wait_for_service() blocks, so it
        # must never be called from inside a callback that runs while the
        # setpoint stream needs to keep flowing (i.e. never from
        # try_arm_and_offboard or any timer callback).
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

        # The setpoint timer gets its own mutually-exclusive callback group so
        # a MultiThreadedExecutor can always run it on time, even if the state
        # machine callback (or a service future callback) is mid-execution on
        # another thread. This is the belt-and-suspenders fix: even if a future
        # change to this file accidentally reintroduces a blocking call
        # somewhere else, the 20 Hz OFFBOARD stream still can't be starved.
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
        self.last_odom_time = time.monotonic()
        if self.launch_xy is None:
            self.launch_xy = (p.x, p.y)
            self.launch_z = p.z
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

    # ---------- helpers ----------

    def odom_is_stale(self) -> bool:
        if self.last_odom_time is None:
            return True
        return (time.monotonic() - self.last_odom_time) > self.odom_stale_timeout

    def set_phase(self, new_phase: str):
        if new_phase != self.phase:
            self.get_logger().info(f'Phase: {self.phase} -> {new_phase}')
            if new_phase == 'LANDING' and self.local_pos is not None:
                # Capture horizontal position ONCE, right here at entry, so
                # LANDING descends straight down from wherever the vehicle
                # actually is (goal, timeout point, wherever) rather than
                # continuously re-tracking it every tick (which would cause
                # horizontal drift during descent instead of a clean drop).
                self.landing_xy = (self.local_pos[0], self.local_pos[1])
            self.phase = new_phase
            self.phase_entered_at = time.monotonic()

    def time_in_phase(self) -> float:
        return time.monotonic() - self.phase_entered_at

    def phase_timed_out(self) -> bool:
        limit = self.phase_timeouts.get(self.phase)
        return limit is not None and self.time_in_phase() > limit

    def offboard_lost(self) -> bool:
        """True if the FC is no longer in a state we expect during an
        active OFFBOARD phase (armed + OFFBOARD mode)."""
        return not (self.current_state.armed and self.current_state.mode == 'OFFBOARD')

    def abort(self, reason: str):
        self.get_logger().error(f'ABORT ({self.phase}): {reason}')
        self.set_phase('ABORTED')

    # ---------- setpoint construction ----------

    def make_position_setpoint(self, x, y, z, yaw=0.0):
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

    def make_velocity_setpoint(self, vx, vy, vz, yaw=0.0):
        """Velocity-only setpoint (position ignored). Used for the liftoff
        burst: commanding velocity directly gives PX4 a firm, non-self-limiting
        climb target, unlike deriving velocity implicitly from a tiny position
        error - see liftoff_speed_mps/liftoff_height_m for why this matters."""
        pt = PositionTarget()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = 'map'
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        pt.type_mask = (
            PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )
        pt.velocity.x, pt.velocity.y, pt.velocity.z = vx, vy, vz
        pt.yaw = yaw
        return pt

    def pos_cmd_to_setpoint(self, msg: PositionCommand) -> PositionTarget:
        pt = PositionTarget()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = 'map'
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        # Yaw is no longer masked off - msg.yaw is actually honored now.
        pt.type_mask = PositionTarget.IGNORE_YAW_RATE
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
            # Don't blindly publish setpoints derived from dead data.
            return

        if self.phase in ('WARMUP', 'ARMED_OFFBOARD'):
            # Hold current position. During WARMUP this establishes the
            # setpoint stream PX4 requires before it will accept an OFFBOARD
            # switch; during ARMED_OFFBOARD it keeps holding while waiting for
            # arm+OFFBOARD confirmation. This must keep running uninterrupted
            # through that transition, which is why try_arm_and_offboard()
            # below is non-blocking.
            if self.local_pos is not None and self.launch_xy is not None:
                x, y = self.launch_xy
                self.setpoint_pub.publish(self.make_position_setpoint(x, y, self.local_pos[2]))

        elif self.phase == 'TAKEOFF':
            x, y = self.launch_xy
            current_z = self.local_pos[2]
            height_gained = current_z - self.launch_z if self.launch_z is not None else 0.0

            if height_gained < self.liftoff_height:
                # Liftoff burst: firm velocity command to punch cleanly through
                # the ground-effect / tip-over-risk zone, rather than the
                # self-limiting tiny position-error ramp used once airborne.
                self.setpoint_pub.publish(self.make_velocity_setpoint(0.0, 0.0, self.liftoff_speed))
            else:
                # Clear of the ground - fall back to the gentler position-based
                # climb the rest of the way to takeoff_alt to avoid overshoot.
                dt = 0.05  # matches this timer's period
                target_z = min(current_z + self.climb_rate * dt, self.takeoff_alt)
                self.setpoint_pub.publish(self.make_position_setpoint(x, y, target_z))

        elif self.phase == 'HOVER':
            # Hold steady at takeoff altitude while the vehicle settles, before
            # handing control to the planner in FLY_OUT.
            x, y = self.launch_xy
            self.setpoint_pub.publish(self.make_position_setpoint(x, y, self.takeoff_alt))

        elif self.phase == 'FLY_OUT':
            if self.latest_pos_cmd is not None:
                self.setpoint_pub.publish(self.pos_cmd_to_setpoint(self.latest_pos_cmd))
            else:
                x, y, z = self.local_pos
                self.setpoint_pub.publish(self.make_position_setpoint(x, y, z))

        elif self.phase == 'LANDING':
            # Hold horizontal position where the vehicle actually is at the
            # moment LANDING begins (captured once, at phase entry, as
            # self.landing_xy) and only descend from there. This lands at the
            # goal/wherever FLY_OUT ended, NOT at launch_xy - using launch_xy
            # here would fly the vehicle all the way back before descending.
            # Fallback to current position if landing_xy somehow wasn't
            # captured (shouldn't happen in practice - set_phase always
            # captures it when local_pos is available).
            x, y = self.landing_xy if self.landing_xy is not None else (self.local_pos[0], self.local_pos[1])
            dt = 0.05
            current_z = self.local_pos[2]
            rate = self.descend_rate if current_z > self.landing_slow_alt else self.landing_slow_rate
            target_z = max(current_z - rate * dt, 0.0)
            self.setpoint_pub.publish(self.make_position_setpoint(x, y, target_z))

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

        # Odometry staleness is fatal in any active flight phase.
        if self.phase in ('WARMUP', 'ARMED_OFFBOARD', 'TAKEOFF', 'HOVER', 'FLY_OUT', 'LANDING') and self.odom_is_stale():
            self.abort('Odometry feed went stale mid-phase.')
            return

        if self.phase == 'WARMUP':
            if self.time_in_phase() > self.setpoint_warmup_time:
                self.try_arm_and_offboard()
            return

        # Unexpected loss of arm/OFFBOARD during an active flight phase means
        # something else (safety pilot, FC rejection, RC override) has taken
        # control. Don't fight it - stop and hand back control cleanly.
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
                self.get_logger().info('Armed and in OFFBOARD - beginning takeoff climb.')
                self.set_phase('TAKEOFF')
            elif not self.arm_offboard_requested:
                # A previous attempt bailed out (services not ready yet) -
                # this is a non-blocking retry, safe to call from here.
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
            dx = self.local_pos[0] - self.test_goal[0]
            dy = self.local_pos[1] - self.test_goal[1]
            dz = self.local_pos[2] - self.test_goal[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < self.goal_tolerance:
                self.get_logger().info(f'Test goal reached (dist={dist:.2f}m). Beginning landing.')
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

        # Non-blocking readiness check only. Never call wait_for_service()
        # here - this method runs from state_machine_cb while the setpoint
        # timer must keep publishing at 20 Hz without interruption.
        if not self.arming_client.service_is_ready() or not self.set_mode_client.service_is_ready():
            now = time.monotonic()
            if now - self.last_service_retry_log > self.service_retry_interval:
                self.get_logger().warn(
                    'Arm/set_mode service not ready yet - will keep retrying non-blockingly.')
                self.last_service_retry_log = now
            return  # setpoint stream is untouched; we'll just try again next tick

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
    # MultiThreadedExecutor + the setpoint timer's own callback group (set up
    # in __init__) guarantees the 20 Hz OFFBOARD setpoint stream can never be
    # starved by a slow or blocking callback elsewhere in the node.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()