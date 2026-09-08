#!/usr/bin/env python3
"""
Minimal relay: fast_lio /Odometry -> /mavros/odometry/out
Fixes: frame IDs, zero twist covariance, broadcasts required static TFs
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros

import numpy as np
from scipy.spatial.transform import Rotation as R

TWIST_COV_DIAGONAL = 0.01  # m/s and rad/s uncertainty

# # Mount rotation: base_link -> livox_frame (from URDF: rpy = 0, 0.7854, 0)
# R_BL_TO_LIVOX = R.from_euler('xyz', [0, 0.7854, 0])
# R_LIVOX_TO_BL = R_BL_TO_LIVOX.inv()
# T_LIVOX_IN_BL = np.array([0.10, 0.0, -0.12])  # offset of livox_frame origin in base_link

# s = np.sqrt(0.5)

# rotation_matrix = np.array([
#     [s,   0.0,  s],
#     [0.0, -1.0, 0.0],
#     [s,   0.0, -s]
# ])

# q = R.from_matrix(rotation_matrix).as_quat()

# def diagonal_cov(val: float) -> list:
#     c = [0.0] * 36
#     for i in range(6):
#         c[i * 6 + i] = val
#     return c

class OdomRelay(Node):
    def __init__(self):
        super().__init__('fast_lio_to_mavros')
        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)
        self._broadcast_mavros_static_tfs()

        self.pub = self.create_publisher(
            Odometry, '/mavros/odometry/out', 10)
        self.sub = self.create_subscription(
            Odometry, '/Odometry', self.cb, 10)
        self.get_logger().info('fast_lio_to_mavros relay ready')

    def _broadcast_mavros_static_tfs(self):
        now = self.get_clock().now().to_msg()
        tfs = []

        # Connect fast_lio's world frame to MAVROS's expected world frame
        # camera_init IS odom — same frame, identity transform
        t0 = TransformStamped()
        t0.header.stamp = now
        t0.header.frame_id = 'odom'
        t0.child_frame_id = 'camera_init'
        t0.transform.rotation.w = 1.0  # identity
        tfs.append(t0)

        #Connect fast_lio's body frame to MAVROS's expected body frame
        #body IS base_link — identity
        # t1 = TransformStamped()
        # t1.header.stamp = now
        # t1.header.frame_id = 'body'
        # t1.child_frame_id = 'base_link'
        # t1.transform.rotation.w = 1.0  # identity
        # tfs.append(t1)

        # t1 = TransformStamped()
        # t1.header.stamp = now
        # t1.header.frame_id = 'body'
        # t1.child_frame_id = 'base_link'

        # t1.transform.rotation.x = q[0]
        # t1.transform.rotation.y = q[1]
        # t1.transform.rotation.z = q[2]
        # t1.transform.rotation.w = q[3]

        # tfs.append(t1)

        # ENU (odom) -> NED (odom_ned)
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = 'odom'
        t2.child_frame_id = 'odom_ned'
        t2.transform.rotation.x = 0.5
        t2.transform.rotation.y = 0.5
        t2.transform.rotation.z = 0.5
        t2.transform.rotation.w = -0.5
        tfs.append(t2)

        # FLU (base_link) -> FRD (base_link_frd)
        t3 = TransformStamped()
        t3.header.stamp = now
        t3.header.frame_id = 'base_link'
        t3.child_frame_id = 'base_link_frd'
        t3.transform.rotation.x = 1.0
        t3.transform.rotation.y = 0.0
        t3.transform.rotation.z = 0.0
        t3.transform.rotation.w = 0.0
        tfs.append(t3)

        self.tf_static.sendTransform(tfs)
        self.get_logger().info('Static TFs broadcast: odom->odom_ned, base_link->base_link_frd')

    def cb(self, msg: Odometry):
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'odom'       # MAVROS appends _ned
        out.child_frame_id = 'base_link'   # MAVROS appends _frd

        # Pass pose through unchanged — fast_lio covariance is good
        out.pose = msg.pose

        # Pass twist, but replace zero covariance with small diagonal
        out.twist.twist = msg.twist.twist
        out.twist.covariance = diagonal_cov(TWIST_COV_DIAGONAL)

        self.pub.publish(out)

    # def cb(self, msg: Odometry):
    #     out = Odometry()
    #     out.header.stamp = msg.header.stamp
    #     out.header.frame_id = 'odom'
    #     out.child_frame_id = 'base_link'

    #     # Orientation: rotate body(=livox) orientation into base_link orientation
    #     q_in = msg.pose.pose.orientation
    #     R_world_body = R.from_quat([q_in.x, q_in.y, q_in.z, q_in.w])
    #     R_world_baselink = R_world_body * R_LIVOX_TO_BL
    #     q_out = R_world_baselink.as_quat()
    #     out.pose.pose.orientation.x = q_out[0]
    #     out.pose.pose.orientation.y = q_out[1]
    #     out.pose.pose.orientation.z = q_out[2]
    #     out.pose.pose.orientation.w = q_out[3]

    #     # Position: account for the lever arm between livox_frame and base_link
    #     p_in = msg.pose.pose.position
    #     p_body = np.array([p_in.x, p_in.y, p_in.z])
    #     offset_world = R_world_body.apply(-T_LIVOX_IN_BL)  # rotate offset into world frame
    #     p_out = p_body + offset_world
    #     out.pose.pose.position.x = p_out[0]
    #     out.pose.pose.position.y = p_out[1]
    #     out.pose.pose.position.z = p_out[2]

    #     out.pose.covariance = msg.pose.covariance  # approx OK; small orientation error from lever arm rotation, usually negligible

    #     # Twist: linear velocity is expressed in child frame (body/livox) — rotate into base_link too
    #     v_in = msg.twist.twist.linear
    #     v_body = np.array([v_in.x, v_in.y, v_in.z])
    #     v_out = R_LIVOX_TO_BL.apply(v_body)
    #     out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = v_out
    #     out.twist.twist.angular = msg.twist.twist.angular  # angular velocity rotates the same way if needed — same R_LIVOX_TO_BL.apply()

    #     out.twist.covariance = diagonal_cov(TWIST_COV_DIAGONAL)
    #     self.pub.publish(out)

def main():
    rclpy.init()
    rclpy.spin(OdomRelay())

if __name__ == '__main__':
    main()