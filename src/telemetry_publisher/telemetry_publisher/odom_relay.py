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

TWIST_COV_DIAGONAL = 0.01  # m/s and rad/s uncertainty

def diagonal_cov(val: float) -> list:
    c = [0.0] * 36
    for i in range(6):
        c[i * 6 + i] = val
    return c

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

        # Connect fast_lio's body frame to MAVROS's expected body frame
        # body IS base_link — identity
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = 'body'
        t1.child_frame_id = 'base_link'
        t1.transform.rotation.w = 1.0  # identity
        tfs.append(t1)

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

def main():
    rclpy.init()
    rclpy.spin(OdomRelay())

if __name__ == '__main__':
    main()