#!/usr/bin/env python3
"""
Centralized odometry frame corrector with automatic startup leveling.

Subscribes to raw /Odometry (FAST-LIO output) AND a raw IMU topic.

FAST-LIO's camera_init frame can retain an initial tilt because the LIO
system initializes from the physical IMU/LiDAR orientation at boot.

STARTUP LEVELING METHOD (gravity-based):

Instead of assuming the drone is physically level and treating any
measured startup roll/pitch as pure error, this node measures the
TRUE vertical directly from the accelerometer (gravity), and computes
the rotation needed to make FAST-LIO's world frame Z-axis line up
with that measured vertical.

This means the drone only needs to be STATIONARY during
initialization -- it does NOT need to be sitting perfectly level.
If it happens to be resting on a slight incline, that incline is
correctly preserved rather than zeroed out.

Processing chain:

    FAST-LIO raw odometry  +  raw accelerometer samples
            |
            v
    gravity-based startup roll/pitch correction
            |
            v
    body -> base_link mount correction
            |
            v
    corrected map/base_link odometry


IMPORTANT ASSUMPTIONS:

1. The drone is STATIONARY during initialization (no requirement that
   it be physically level -- gravity tells us the true vertical
   regardless of resting angle).

2. The IMU topic subscribed here is the SAME physical sensor whose
   orientation FAST-LIO publishes as /Odometry's orientation (i.e.
   the accelerometer samples and R_raw_body refer to the same body
   frame with no unmodeled extrinsic rotation between them). If your
   platform uses a separate IMU for this topic, you must rotate the
   accelerometer readings into the FAST-LIO body frame yourself
   before they reach this node.

3. sensor_msgs/Imu.linear_acceleration follows the "specific force"
   convention: a stationary, level sensor reads approximately
   +9.81 m/s^2 on its "up" axis (gravity is NOT pre-subtracted by the
   driver). If your driver publishes gravity-compensated
   acceleration, this method will not work as written.

4. R_MOUNT represents the physical body/IMU -> base_link mount
   rotation and is unrelated to the startup leveling correction.

5. FAST-LIO twist is expressed in the child/body frame.

6. Yaw is NOT corrected during startup. The gravity-alignment
   rotation used here is the minimal-angle rotation between the
   measured "up" vector and canonical vertical, which is
   mathematically guaranteed to contain zero yaw component (its
   rotation axis is always horizontal), so this invariant holds
   automatically.

7. The startup correction is a fixed rotation. It is NOT recalculated
   while flying.

8. The existing R_MOUNT correction and startup leveling correction are
   conceptually separate. Do not assume they are the same rotation.


Startup procedure:

    t = 0
        |
        | FAST-LIO initializes
        |
        | collect orientation samples (for frame reference)
        | collect accelerometer samples (for true vertical)
        |
    t = initialization_time
        |
        | average accelerometer -> measured gravity direction (body)
        | express it in FAST-LIO's (possibly tilted) world frame
        | compute minimal rotation -> measured vertical == true up
        |
        v
    lock correction
        |
        v
    publish corrected odometry


The raw /Odometry topic is never modified.
The corrected result is published to /Odometry_corrected.
"""

import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped

import tf2_ros
import numpy as np


# =====================================================================
# PHYSICAL MOUNT ROTATION
# =====================================================================

# Mount rotation: body (Livox/IMU) <-> base_link (FC).
#
# This is your existing 45-degree/upside-down mount correction.
#
# The matrix is self-inverse (approximately symmetric + orthogonal),
# so the same matrix can be used in the sandwich transform below.
#
# IMPORTANT:
# This is kept separate from the automatically measured startup
# leveling correction.
#
R_MOUNT = np.array([
    [ 0.70710681,  0.0,         0.70710676],
    [ 0.0,        -1.0,         0.0       ],
    [ 0.70710676,  0.0,        -0.70710681],
])


# =====================================================================
# LEVER ARM
# =====================================================================

# Base_link origin position, expressed in body frame.
#
# This is your existing approximation from the URDF mount offset.
#
T_BODY_TO_BASELINK = np.array([
    0.10,
    0.0,
   -0.12
])


# =====================================================================
# ROTATION UTILITIES
# =====================================================================

def quat_to_matrix(x, y, z, w):
    """Quaternion -> 3x3 rotation matrix."""

    n = x * x + y * y + z * z + w * w

    if n < 1e-12:
        return np.eye(3)

    s = 2.0 / n

    xx = x * x * s
    yy = y * y * s
    zz = z * z * s

    xy = x * y * s
    xz = x * z * s
    yz = y * z * s

    wx = w * x * s
    wy = w * y * s
    wz = w * z * s

    return np.array([
        [1 - (yy + zz), xy - wz,       xz + wy],
        [xy + wz,       1 - (xx + zz), yz - wx],
        [xz - wy,       yz + wx,       1 - (xx + yy)],
    ])


def matrix_to_quat(R):
    """3x3 rotation matrix -> quaternion (x, y, z, w)."""

    tr = np.trace(R)

    if tr > 0:

        S = np.sqrt(tr + 1.0) * 2

        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:

        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2

        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S

    elif R[1, 1] > R[2, 2]:

        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2

        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S

    else:

        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2

        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return x, y, z, w


def skew(v):
    """3-vector -> skew-symmetric matrix."""

    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def rotation_to_rpy(R):
    """
    Rotation matrix -> roll, pitch, yaw.

    Returns radians.

    Uses the standard ROS-style convention.
    """

    pitch = math.asin(
        np.clip(-R[2, 0], -1.0, 1.0)
    )

    roll = math.atan2(
        R[2, 1],
        R[2, 2]
    )

    yaw = math.atan2(
        R[1, 0],
        R[0, 0]
    )

    return roll, pitch, yaw


def rotation_from_two_vectors(v_from, v_to):
    """
    Return the minimal-angle rotation matrix R such that

        R @ (v_from / |v_from|) == (v_to / |v_to|)

    This is the shortest-arc rotation between the two directions.
    Its rotation axis is cross(v_from, v_to), which -- whenever v_to
    is the canonical vertical (0, 0, 1) -- is guaranteed to be
    horizontal (zero z-component). That means this rotation can never
    introduce a yaw component: it only ever expresses roll/pitch.

    Handles the near-identical and near-antiparallel edge cases
    explicitly, since the general formula is singular at 0 and 180
    degrees.
    """

    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)

    cos_angle = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    sin_angle = float(np.linalg.norm(axis))

    if sin_angle < 1e-9:

        if cos_angle > 0.0:
            # Already aligned -- no correction needed.
            return np.eye(3)

        # Antiparallel (180 degrees). The rotation axis is undefined
        # by the cross product here, so pick any axis perpendicular
        # to v_from.
        fallback = np.array([1.0, 0.0, 0.0])

        if abs(a[0]) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])

        perp_axis = np.cross(a, fallback)
        perp_axis = perp_axis / np.linalg.norm(perp_axis)

        K = skew(perp_axis)

        # Rodrigues formula at theta = pi: sin(pi)=0, 1-cos(pi)=2.
        return np.eye(3) + 2.0 * (K @ K)

    axis_normalized = axis / sin_angle
    K = skew(axis_normalized)

    # Rodrigues' rotation formula.
    R = (
        np.eye(3)
        + sin_angle * K
        + (1.0 - cos_angle) * (K @ K)
    )

    return R


# =====================================================================
# NODE
# =====================================================================

class OdomFrameCorrector(Node):

    def __init__(self):

        super().__init__('odom_frame_corrector')

        # -------------------------------------------------------------
        # Parameters
        # -------------------------------------------------------------

        self.declare_parameter(
            'input_topic',
            '/Odometry'
        )

        self.declare_parameter(
            'output_topic',
            '/Odometry_corrected'
        )

        self.declare_parameter(
            'output_frame_id',
            'map'
        )

        self.declare_parameter(
            'output_child_frame_id',
            'base_link'
        )

        self.declare_parameter(
            'broadcast_tf',
            True
        )

        self.declare_parameter(
            'initialization_time',
            2.0
        )

        self.declare_parameter(
            'min_samples',
            50
        )

        # NEW:
        #
        # Raw IMU topic used to measure true gravity direction during
        # startup. MUST be the same physical sensor whose orientation
        # FAST-LIO publishes on input_topic (see module docstring,
        # assumption 2).
        #
        self.declare_parameter(
            'imu_topic',
            '/livox/imu'
        )

        # NEW:
        #
        # Expected gravity magnitude (m/s^2), used only as a sanity
        # check on the averaged accelerometer reading.
        #
        self.declare_parameter(
            'expected_gravity',
            9.80665
        )

        # NEW:
        #
        # Fractional tolerance on the gravity magnitude sanity check
        # (0.05 = warn if measured norm is off by more than 5%).
        #
        self.declare_parameter(
            'gravity_tolerance',
            0.05
        )

        # NEW:
        #
        # Peak angular rate (rad/s) allowed during initialization
        # before warning that the drone may not have been stationary.
        # This is advisory only; it does not block initialization.
        #
        self.declare_parameter(
            'max_stationary_angular_rate',
            0.05
        )

        input_topic = self.get_parameter(
            'input_topic'
        ).value

        output_topic = self.get_parameter(
            'output_topic'
        ).value

        self.output_frame_id = self.get_parameter(
            'output_frame_id'
        ).value

        self.output_child_frame_id = self.get_parameter(
            'output_child_frame_id'
        ).value

        self.broadcast_tf = self.get_parameter(
            'broadcast_tf'
        ).value

        self.initialization_time = float(
            self.get_parameter(
                'initialization_time'
            ).value
        )

        self.min_samples = int(
            self.get_parameter(
                'min_samples'
            ).value
        )

        imu_topic = self.get_parameter(
            'imu_topic'
        ).value

        self.expected_gravity = float(
            self.get_parameter(
                'expected_gravity'
            ).value
        )

        self.gravity_tolerance = float(
            self.get_parameter(
                'gravity_tolerance'
            ).value
        )

        self.max_stationary_angular_rate = float(
            self.get_parameter(
                'max_stationary_angular_rate'
            ).value
        )

        # -------------------------------------------------------------
        # TF
        # -------------------------------------------------------------

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(
            self
        )

        # -------------------------------------------------------------
        # Twist transformation
        # -------------------------------------------------------------

        arm_matrix = skew(
            T_BODY_TO_BASELINK
        )

        angular_to_linear = (
            arm_matrix @ R_MOUNT
        )

        self._twist_tf_matrix = np.block([
            [
                R_MOUNT,
                angular_to_linear
            ],
            [
                np.zeros((3, 3)),
                R_MOUNT
            ]
        ])

        # -------------------------------------------------------------
        # ROS publishers/subscribers
        # -------------------------------------------------------------

        self.sub = self.create_subscription(
            Odometry,
            input_topic,
            self.cb,
            10
        )

        # NEW: raw IMU subscription, used only during startup.
        self.imu_sub = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_cb,
            50
        )

        self.pub = self.create_publisher(
            Odometry,
            output_topic,
            10
        )

        # -------------------------------------------------------------
        # Startup initialization state
        # -------------------------------------------------------------

        self.initialization_start = self.get_clock().now()

        self.initialized = False

        # Store raw startup rotations.
        #
        # We deliberately store rotation matrices instead of Euler
        # angles to avoid averaging-angle problems. These are still
        # needed to express the measured gravity direction (which
        # comes in the body frame) inside FAST-LIO's world frame.
        #
        self.startup_rotations = []

        # NEW: raw accelerometer and gyroscope samples collected
        # during the same startup window.
        self.accel_samples = []
        self.angular_rate_samples = []

        # Fixed correction calculated after initialization.
        #
        self.R_LEVEL = np.eye(3)

        # -------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------

        self.get_logger().info(
            '\n'
            '====================================================\n'
            ' Fast-LIO Odometry Frame Corrector (gravity leveling)\n'
            '====================================================\n'
            f' Input topic          : {input_topic}\n'
            f' IMU topic            : {imu_topic}\n'
            f' Output topic         : {output_topic}\n'
            f' Output frame         : {self.output_frame_id}\n'
            f' Output child frame   : {self.output_child_frame_id}\n'
            f' Startup time         : {self.initialization_time:.2f} s\n'
            f' Minimum samples      : {self.min_samples}\n'
            f' TF broadcasting      : {self.broadcast_tf}\n'
            '\n'
            ' IMPORTANT:\n'
            ' Keep the drone STATIONARY during the initialization\n'
            ' period. It no longer needs to be perfectly level --\n'
            ' the true vertical is measured from the accelerometer.\n'
            '===================================================='
        )

    # =================================================================
    # CALLBACKS
    # =================================================================

    def imu_cb(self, msg: Imu):
        """Collect accelerometer/gyro samples during startup only."""

        if self.initialized:
            return

        a = msg.linear_acceleration
        self.accel_samples.append(
            np.array([a.x, a.y, a.z])
        )

        w = msg.angular_velocity
        self.angular_rate_samples.append(
            np.array([w.x, w.y, w.z])
        )

        self.maybe_finish_initialization()

    def cb(self, msg: Odometry):

        # -------------------------------------------------------------
        # Extract raw orientation
        # -------------------------------------------------------------

        q_in = msg.pose.pose.orientation

        R_raw_body = quat_to_matrix(
            q_in.x,
            q_in.y,
            q_in.z,
            q_in.w
        )

        # -------------------------------------------------------------
        # STARTUP INITIALIZATION
        # -------------------------------------------------------------

        if not self.initialized:

            self.startup_rotations.append(
                R_raw_body.copy()
            )

            self.maybe_finish_initialization()

            # IMPORTANT:
            #
            # Do not publish anything while Fast-LIO is settling.
            #
            return

        # -------------------------------------------------------------
        # STEP 1:
        #
        # Remove the startup roll/pitch error.
        #
        # -------------------------------------------------------------

        R_levelled_body = (
            self.R_LEVEL @ R_raw_body
        )

        # -------------------------------------------------------------
        # STEP 2:
        #
        # Apply the existing physical mount correction.
        #
        # This converts body/Livox orientation into base_link.
        #
        # -------------------------------------------------------------

        R_world_baselink = (
            R_MOUNT
            @ R_levelled_body
            @ R_MOUNT
        )

        # -------------------------------------------------------------
        # Create output message
        # -------------------------------------------------------------

        out = Odometry()

        out.header.stamp = msg.header.stamp

        out.header.frame_id = (
            self.output_frame_id
        )

        out.child_frame_id = (
            self.output_child_frame_id
        )

        # -------------------------------------------------------------
        # ORIENTATION
        # -------------------------------------------------------------

        qx, qy, qz, qw = matrix_to_quat(
            R_world_baselink
        )

        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        # -------------------------------------------------------------
        # POSITION
        # -------------------------------------------------------------

        p_in = msg.pose.pose.position

        p_camera_init_body = np.array([
            p_in.x,
            p_in.y,
            p_in.z
        ])

        # Lever arm in raw body frame.
        p_camera_init_baselink = (
            p_camera_init_body
            + R_raw_body @ T_BODY_TO_BASELINK
        )

        # Apply startup leveling to position as well.
        #
        # This is important:
        #
        # If camera_init itself is tilted, positions must be rotated
        # into the corrected world frame too.
        #
        p_levelled_baselink = (
            self.R_LEVEL
            @ p_camera_init_baselink
        )

        # Apply mount correction.
        p_world_baselink = (
            R_MOUNT
            @ p_levelled_baselink
        )

        out.pose.pose.position.x = (
            p_world_baselink[0]
        )

        out.pose.pose.position.y = (
            p_world_baselink[1]
        )

        out.pose.pose.position.z = (
            p_world_baselink[2]
        )

        # -------------------------------------------------------------
        # POSE COVARIANCE
        # -------------------------------------------------------------

        cov_in = np.array(
            msg.pose.covariance
        ).reshape(6, 6)

        R_block = np.block([
            [
                R_MOUNT,
                np.zeros((3, 3))
            ],
            [
                np.zeros((3, 3)),
                R_MOUNT
            ]
        ])

        cov_out = (
            R_block
            @ cov_in
            @ R_block.T
        )

        out.pose.covariance = (
            cov_out.flatten().tolist()
        )

        # -------------------------------------------------------------
        # TWIST
        # -------------------------------------------------------------

        tw_lin = msg.twist.twist.linear
        tw_ang = msg.twist.twist.angular

        tw_vec = np.array([
            tw_lin.x,
            tw_lin.y,
            tw_lin.z,
            tw_ang.x,
            tw_ang.y,
            tw_ang.z
        ])

        tw_out = (
            self._twist_tf_matrix
            @ tw_vec
        )

        out.twist.twist.linear.x = tw_out[0]
        out.twist.twist.linear.y = tw_out[1]
        out.twist.twist.linear.z = tw_out[2]

        out.twist.twist.angular.x = tw_out[3]
        out.twist.twist.angular.y = tw_out[4]
        out.twist.twist.angular.z = tw_out[5]

        # -------------------------------------------------------------
        # TWIST COVARIANCE
        # -------------------------------------------------------------

        tw_cov_in = np.array(
            msg.twist.covariance
        ).reshape(6, 6)

        tw_cov_out = (
            self._twist_tf_matrix
            @ tw_cov_in
            @ self._twist_tf_matrix.T
        )

        out.twist.covariance = (
            tw_cov_out.flatten().tolist()
        )

        # -------------------------------------------------------------
        # PUBLISH
        # -------------------------------------------------------------

        self.pub.publish(out)

        # -------------------------------------------------------------
        # TF
        # -------------------------------------------------------------

        if self.broadcast_tf:

            t = TransformStamped()

            t.header.stamp = (
                out.header.stamp
            )

            t.header.frame_id = (
                self.output_frame_id
            )

            t.child_frame_id = (
                self.output_child_frame_id
            )

            t.transform.translation.x = (
                out.pose.pose.position.x
            )

            t.transform.translation.y = (
                out.pose.pose.position.y
            )

            t.transform.translation.z = (
                out.pose.pose.position.z
            )

            t.transform.rotation = (
                out.pose.pose.orientation
            )

            self.tf_broadcaster.sendTransform(t)

    # =================================================================
    # STARTUP GATING
    # =================================================================

    def maybe_finish_initialization(self):
        """
        Check whether enough time has passed and enough samples from
        BOTH the odometry topic and the IMU topic have been collected.
        Called from both callbacks since the two topics generally
        publish at different rates.
        """

        if self.initialized:
            return

        elapsed = (
            self.get_clock().now()
            - self.initialization_start
        ).nanoseconds / 1e9

        if elapsed < self.initialization_time:
            return

        if len(self.startup_rotations) < self.min_samples:
            return

        if len(self.accel_samples) < self.min_samples:
            return

        self.finish_initialization()

    # =================================================================
    # FINISH STARTUP INITIALIZATION
    # =================================================================

    def finish_initialization(self):

        # -------------------------------------------------------------
        # Average the startup rotation matrices.
        #
        # The arithmetic mean of rotation matrices is not itself
        # necessarily a valid rotation matrix, so we project it back
        # onto SO(3) using SVD.
        #
        # This is still needed even though leveling is now
        # gravity-based: it's what lets us express the measured
        # (body-frame) gravity direction inside FAST-LIO's world
        # frame.
        # -------------------------------------------------------------

        R_sum = np.zeros((3, 3))

        for R in self.startup_rotations:
            R_sum += R

        R_mean = (
            R_sum
            / len(self.startup_rotations)
        )

        U, _, Vt = np.linalg.svd(
            R_mean
        )

        R_initial = U @ Vt

        if np.linalg.det(R_initial) < 0:

            U[:, -1] *= -1

            R_initial = U @ Vt

        # -------------------------------------------------------------
        # NEW: average accelerometer samples -> measured gravity
        # direction in the body frame.
        # -------------------------------------------------------------

        accel_stack = np.array(self.accel_samples)
        accel_mean = accel_stack.mean(axis=0)
        accel_norm = np.linalg.norm(accel_mean)

        gravity_error = (
            abs(accel_norm - self.expected_gravity)
            / self.expected_gravity
        )

        if gravity_error > self.gravity_tolerance:
            self.get_logger().warn(
                f'Measured gravity magnitude {accel_norm:.3f} m/s^2 '
                f'differs from expected {self.expected_gravity:.3f} '
                f'm/s^2 by {gravity_error * 100:.1f}%. The drone may '
                f'not have been stationary during initialization, or '
                f'imu_topic may not match the IMU used by FAST-LIO. '
                f'The leveling result may be inaccurate.'
            )

        if len(self.angular_rate_samples) > 0:

            max_rate = float(np.max(
                np.linalg.norm(
                    np.array(self.angular_rate_samples),
                    axis=1
                )
            ))

            if max_rate > self.max_stationary_angular_rate:
                self.get_logger().warn(
                    f'Peak angular rate during initialization was '
                    f'{math.degrees(max_rate):.2f} deg/s, exceeding '
                    f'the stationary threshold of '
                    f'{math.degrees(self.max_stationary_angular_rate):.2f} '
                    f'deg/s. The drone may have been moved during '
                    f'initialization.'
                )

        gravity_dir_body = accel_mean / accel_norm

        # Express the measured gravity direction (body frame) inside
        # FAST-LIO's world frame, using the averaged raw orientation.
        gravity_dir_world = R_initial @ gravity_dir_body

        # A stationary, level accelerometer reads +g on its "up" axis
        # (specific force convention). We want base_link's up axis to
        # align with that once the full chain (R_LEVEL sandwiched by
        # R_MOUNT) is applied.
        #
        # IMPORTANT: R_LEVEL is composed with R_raw_body BEFORE
        # R_MOUNT is applied (see cb(): R_levelled_body = R_LEVEL @
        # R_raw_body, then R_MOUNT @ R_levelled_body @ R_MOUNT). So
        # the target vertical must be expressed in that same
        # pre-mount frame -- i.e. pulled back through R_MOUNT -- NOT
        # left as plain [0, 0, 1]. Using [0, 0, 1] directly here would
        # ask FAST-LIO's raw body frame to itself be vertical, which
        # is never true when there's a fixed non-trivial mount
        # rotation (e.g. a 45-degree mount), and produces a large
        # spurious "correction" that actually fights R_MOUNT instead
        # of complementing it.
        target_up = R_MOUNT @ np.array([0.0, 0.0, 1.0])

        # -------------------------------------------------------------
        # The correction that removes the measured initial tilt.
        #
        # This is the minimal rotation aligning the measured gravity
        # direction to true vertical. It is guaranteed to contain no
        # yaw component (see rotation_from_two_vectors docstring).
        # -------------------------------------------------------------

        self.R_LEVEL = rotation_from_two_vectors(
            gravity_dir_world,
            target_up
        )

        self.initialized = True

        # -------------------------------------------------------------
        # Diagnostics: compare FAST-LIO's own raw tilt estimate
        # against what the gravity measurement implies, and show the
        # corrected result.
        # -------------------------------------------------------------

        roll0, pitch0, yaw0 = rotation_to_rpy(R_initial)

        R_test = self.R_LEVEL @ R_initial

        roll_test, pitch_test, yaw_test = rotation_to_rpy(R_test)

        self.get_logger().info(
            '\n'
            '====================================================\n'
            ' FAST-LIO STARTUP INITIALIZATION COMPLETE (gravity)\n'
            '====================================================\n'
            f' Odometry samples     : {len(self.startup_rotations)}\n'
            f' Accelerometer samples: {len(self.accel_samples)}\n'
            f' Measured |g|         : {accel_norm:.4f} m/s^2\n'
            '\n'
            f' FAST-LIO raw roll  : '
            f'{math.degrees(roll0):+.3f} deg\n'
            f' FAST-LIO raw pitch : '
            f'{math.degrees(pitch0):+.3f} deg\n'
            f' FAST-LIO raw yaw   : '
            f'{math.degrees(yaw0):+.3f} deg\n'
            '\n'
            f' Corrected roll     : '
            f'{math.degrees(roll_test):+.3f} deg\n'
            f' Corrected pitch    : '
            f'{math.degrees(pitch_test):+.3f} deg\n'
            f' Corrected yaw      : '
            f'{math.degrees(yaw_test):+.3f} deg\n'
            '===================================================='
        )

        # Free the samples.
        self.startup_rotations.clear()
        self.accel_samples.clear()
        self.angular_rate_samples.clear()


# =====================================================================
# MAIN
# =====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = OdomFrameCorrector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
