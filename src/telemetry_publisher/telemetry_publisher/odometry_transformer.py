#!/usr/bin/env python3
"""
Centralized odometry frame corrector with automatic startup leveling.

Subscribes to raw /Odometry (FAST-LIO output).

FAST-LIO's camera_init frame can retain an initial tilt because the LIO
system initializes from the physical IMU/LiDAR orientation at boot.
If the drone is physically level and stationary during initialization,
this node measures the residual initial roll/pitch and removes it.

Processing chain:

    FAST-LIO raw odometry
            |
            v
    startup roll/pitch correction
            |
            v
    body -> base_link mount correction
            |
            v
    corrected map/base_link odometry


IMPORTANT ASSUMPTIONS:

1. The drone is physically LEVEL and STATIONARY during initialization.

2. R_MOUNT represents the physical body/IMU -> base_link mount rotation.

3. FAST-LIO twist is expressed in the child/body frame.

4. Yaw is NOT corrected during startup. Only roll and pitch are leveled.

5. The startup correction is a fixed rotation. It is NOT recalculated
   while flying.

6. The existing R_MOUNT correction and startup leveling correction are
   conceptually separate. Do not assume they are the same rotation.


Startup procedure:

    t = 0
        |
        | FAST-LIO initializes
        |
        | collect orientation samples
        |
    t = initialization_time
        |
        | calculate initial roll/pitch
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

        # NEW:
        #
        # How long to collect startup samples.
        #
        self.declare_parameter(
            'initialization_time',
            2.0
        )

        # NEW:
        #
        # Minimum number of samples required before initialization
        # can complete.
        #
        self.declare_parameter(
            'min_samples',
            50
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
        # angles to avoid averaging-angle problems.
        #
        self.startup_rotations = []

        # Fixed correction calculated after initialization.
        #
        self.R_LEVEL = np.eye(3)

        # -------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------

        self.get_logger().info(
            '\n'
            '====================================================\n'
            ' Fast-LIO Odometry Frame Corrector\n'
            '====================================================\n'
            f' Input topic          : {input_topic}\n'
            f' Output topic         : {output_topic}\n'
            f' Output frame         : {self.output_frame_id}\n'
            f' Output child frame   : {self.output_child_frame_id}\n'
            f' Startup time         : {self.initialization_time:.2f} s\n'
            f' Minimum samples      : {self.min_samples}\n'
            f' TF broadcasting      : {self.broadcast_tf}\n'
            '\n'
            ' IMPORTANT:\n'
            ' Keep the drone physically LEVEL and STATIONARY\n'
            ' during the initialization period.\n'
            '===================================================='
        )

    # =================================================================
    # CALLBACK
    # =================================================================

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

            elapsed = (
                self.get_clock().now()
                - self.initialization_start
            ).nanoseconds / 1e9

            if (
                elapsed >= self.initialization_time
                and len(self.startup_rotations)
                >= self.min_samples
            ):

                self.finish_initialization()

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
    # FINISH STARTUP INITIALIZATION
    # =================================================================

    def finish_initialization(self):

        # -------------------------------------------------------------
        # Average the startup rotation matrices.
        #
        # The arithmetic mean of rotation matrices is not itself
        # necessarily a valid rotation matrix, so we project it back
        # onto SO(3) using SVD.
        # -------------------------------------------------------------

        R_sum = np.zeros((3, 3))

        for R in self.startup_rotations:
            R_sum += R

        R_mean = (
            R_sum
            / len(self.startup_rotations)
        )

        # -------------------------------------------------------------
        # Project onto the nearest valid rotation matrix.
        # -------------------------------------------------------------

        U, _, Vt = np.linalg.svd(
            R_mean
        )

        R_initial = U @ Vt

        # Make sure determinant is +1.
        if np.linalg.det(R_initial) < 0:

            U[:, -1] *= -1

            R_initial = U @ Vt

        # -------------------------------------------------------------
        # Extract startup roll/pitch/yaw.
        # -------------------------------------------------------------

        roll0, pitch0, yaw0 = (
            rotation_to_rpy(R_initial)
        )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # We only want to remove roll and pitch.
        #
        # Yaw is deliberately preserved.
        #
        # Build a rotation representing ONLY the startup roll/pitch.
        # -------------------------------------------------------------

        cr = math.cos(roll0)
        sr = math.sin(roll0)

        cp = math.cos(pitch0)
        sp = math.sin(pitch0)

        R_roll = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ])

        R_pitch = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ])

        R_initial_tilt = (
            R_pitch @ R_roll
        )

        # -------------------------------------------------------------
        # The correction that removes this initial tilt.
        # -------------------------------------------------------------

        self.R_LEVEL = (
            R_initial_tilt.T
        )

        self.initialized = True

        # -------------------------------------------------------------
        # Calculate what the corrected startup orientation looks like.
        # -------------------------------------------------------------

        R_test = (
            self.R_LEVEL
            @ R_initial
        )

        roll_test, pitch_test, yaw_test = (
            rotation_to_rpy(R_test)
        )

        # -------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------

        self.get_logger().info(
            '\n'
            '====================================================\n'
            ' FAST-LIO STARTUP INITIALIZATION COMPLETE\n'
            '====================================================\n'
            f' Samples collected : {len(self.startup_rotations)}\n'
            '\n'
            f' Raw initial roll  : '
            f'{math.degrees(roll0):+.3f} deg\n'
            f' Raw initial pitch : '
            f'{math.degrees(pitch0):+.3f} deg\n'
            f' Raw initial yaw   : '
            f'{math.degrees(yaw0):+.3f} deg\n'
            '\n'
            f' Roll correction   : '
            f'{math.degrees(-roll0):+.3f} deg\n'
            f' Pitch correction  : '
            f'{math.degrees(-pitch0):+.3f} deg\n'
            ' Yaw correction    : 0.000 deg\n'
            '\n'
            f' Corrected roll    : '
            f'{math.degrees(roll_test):+.3f} deg\n'
            f' Corrected pitch   : '
            f'{math.degrees(pitch_test):+.3f} deg\n'
            f' Corrected yaw     : '
            f'{math.degrees(yaw_test):+.3f} deg\n'
            '===================================================='
        )

        # Free the samples.
        self.startup_rotations.clear()


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
