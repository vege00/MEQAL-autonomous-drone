#!/usr/bin/env python3
"""
Centralized odometry frame corrector.

Subscribes to raw /Odometry (FAST-LIO output: pose of `body` in `camera_init`,
where camera_init is NOT gravity-aligned -- it's defined as coincident with
body's orientation at boot, tilt included).

Publishes /Odometry_corrected: pose of `base_link` in a properly-leveled
world frame, ready for any downstream consumer (MAVROS bridge, point cloud
aggregator, ego-planner, RViz, etc) without each of them re-deriving the
correction independently.

ASSUMPTIONS -- verify these before trusting this in flight:
  1. The drone was physically level when FAST-LIO/the LIO node was
     initialized. If not, R_MOUNT is NOT the correct camera_init-tilt
     correction and this will produce a wrong constant offset.
  2. twist.twist in the incoming Odometry is expressed in the CHILD frame
     (body), per REP 103/105 convention. If FAST-LIO's ROS2 driver instead
     expresses twist in the world/camera_init frame, the twist transform
     below needs to change (drop the lever-arm/rotation logic and just
     apply R_MOUNT directly, matching the position correction instead).
     Verify by driving in a known direction and checking twist.linear.x
     lines up with body-forward, not world-forward.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
import numpy as np


# Mount rotation: body (Livox/IMU) <-> base_link (FC).
# Self-inverse (symmetric + orthogonal) -- same matrix serves as both the
# body->base_link mount AND (per the level-boot assumption) the camera_init
# ->true-world tilt correction. These are conceptually different physical
# quantities that happen to coincide numerically in this specific setup;
# don't assume this coincidence holds if you remount the sensor differently.
R_MOUNT = np.array([
    [ 0.70710681,  0.0,  0.70710676],
    [ 0.0,        -1.0,  0.0       ],
    [ 0.70710676,  0.0, -0.70710681],
])

# Lever arm: base_link origin position, expressed in the body frame.
# NOTE: approximation from URDF mount offset -- verify physically if you
# need tight lever-arm accuracy (matters most for tight self-exclusion
# radii or aggressive rotation).
T_BODY_TO_BASELINK = np.array([0.10, 0.0, -0.12])


def quat_to_matrix(x, y, z, w):
    """Quaternion -> 3x3 rotation matrix, no scipy dependency."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
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
    """3-vector -> skew-symmetric matrix (for cross-product-as-matmul)."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


class OdomFrameCorrector(Node):
    def __init__(self):
        super().__init__('odom_frame_corrector')

        self.declare_parameter('input_topic', '/Odometry')
        self.declare_parameter('output_topic', '/Odometry_corrected')
        self.declare_parameter('output_frame_id', 'map')
        self.declare_parameter('output_child_frame_id', 'base_link')
        self.declare_parameter('broadcast_tf', True)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.output_frame_id = self.get_parameter('output_frame_id').value
        self.output_child_frame_id = self.get_parameter('output_child_frame_id').value
        self.broadcast_tf = self.get_parameter('broadcast_tf').value

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Precompute the twist transform matrix (Karelics-style arm-lever block matrix).
        # Rotates+shifts a body-frame twist (linear, angular) into a base_link-frame twist,
        # accounting for the lever arm's contribution to linear velocity under rotation.
        arm_matrix = skew(T_BODY_TO_BASELINK)
        angular_to_linear = arm_matrix @ R_MOUNT
        self._twist_tf_matrix = np.block([
            [R_MOUNT, angular_to_linear],
            [np.zeros((3, 3)), R_MOUNT],
        ])

        self.sub = self.create_subscription(Odometry, input_topic, self.cb, 10)
        self.pub = self.create_publisher(Odometry, output_topic, 10)

        self.get_logger().info(
            f'odom_frame_corrector started\n'
            f'  subscribing : {input_topic}\n'
            f'  publishing  : {output_topic} (frame_id={self.output_frame_id}, '
            f'child_frame_id={self.output_child_frame_id})\n'
            f'  ASSUMES: level boot, twist expressed in child(body) frame -- verify both!'
        )

    def cb(self, msg: Odometry):
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.output_frame_id
        out.child_frame_id = self.output_child_frame_id

        # --- Orientation: sandwich correction (camera_init tilt + mount) ---
        q_in = msg.pose.pose.orientation
        R_raw_body = quat_to_matrix(q_in.x, q_in.y, q_in.z, q_in.w)
        R_world_baselink = R_MOUNT @ R_raw_body @ R_MOUNT
        qx, qy, qz, qw = matrix_to_quat(R_world_baselink)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        # --- Position ---
        p_in = msg.pose.pose.position
        p_camera_init_body = np.array([p_in.x, p_in.y, p_in.z])
        p_camera_init_baselink = p_camera_init_body + R_raw_body @ T_BODY_TO_BASELINK
        p_world_baselink = R_MOUNT @ p_camera_init_baselink
        out.pose.pose.position.x = p_world_baselink[0]
        out.pose.pose.position.y = p_world_baselink[1]
        out.pose.pose.position.z = p_world_baselink[2]

        # --- Pose covariance (approximate) ---
        # Rotates the position and orientation 3x3 covariance blocks by R_MOUNT.
        # This ignores the nonlinear coupling introduced by the lever arm rotating
        # with orientation -- a standard linearization also used by Karelics'
        # library. Fine for typical FAST-LIO covariance magnitudes; revisit if
        # you're doing tight EKF fusion downstream and see inconsistent covariance.
        cov_in = np.array(msg.pose.covariance).reshape(6, 6)
        R_block = np.block([
            [R_MOUNT, np.zeros((3, 3))],
            [np.zeros((3, 3)), R_MOUNT],
        ])
        cov_out = R_block @ cov_in @ R_block.T
        out.pose.covariance = cov_out.flatten().tolist()

        # --- Twist ---
        # ASSUMPTION: msg.twist.twist is expressed in the child (body) frame.
        # If FAST-LIO instead publishes twist in camera_init/world frame,
        # this needs to change -- see module docstring.
        tw_lin = msg.twist.twist.linear
        tw_ang = msg.twist.twist.angular
        tw_vec = np.array([tw_lin.x, tw_lin.y, tw_lin.z, tw_ang.x, tw_ang.y, tw_ang.z])
        tw_out = self._twist_tf_matrix @ tw_vec
        out.twist.twist.linear.x = tw_out[0]
        out.twist.twist.linear.y = tw_out[1]
        out.twist.twist.linear.z = tw_out[2]
        out.twist.twist.angular.x = tw_out[3]
        out.twist.twist.angular.y = tw_out[4]
        out.twist.twist.angular.z = tw_out[5]

        tw_cov_in = np.array(msg.twist.covariance).reshape(6, 6)
        tw_cov_out = self._twist_tf_matrix @ tw_cov_in @ self._twist_tf_matrix.T
        out.twist.covariance = tw_cov_out.flatten().tolist()

        self.pub.publish(out)

        # --- Live TF broadcast, so RViz/robot_state_publisher's children
        # (propellers, livox_frame) and any TF-based ego-planner lookup see
        # a correctly-oriented base_link. This REPLACES the old static
        # body->base_link transform -- that one must be stopped, it now
        # conflicts with this (it's uncorrected: single mount rotation
        # only, no camera_init-tilt correction, no lever arm).
        if self.broadcast_tf:
            t = TransformStamped()
            t.header.stamp = out.header.stamp
            t.header.frame_id = self.output_frame_id
            t.child_frame_id = self.output_child_frame_id
            t.transform.translation.x = out.pose.pose.position.x
            t.transform.translation.y = out.pose.pose.position.y
            t.transform.translation.z = out.pose.pose.position.z
            t.transform.rotation = out.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)


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