#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class FastLioBridgeNode(Node):
    """
    Bridges corrected odometry to MAVROS vision_pose for PX4 EKF2.

    Input   : /Odometry_corrected (nav_msgs/Odometry) -- already gravity-
              aligned and expressed as base_link's pose, via odom_frame_corrector.
    Output  : /mavros/vision_pose/pose (geometry_msgs/PoseStamped)

    No rotation math here anymore -- odom_frame_corrector already handles
    the camera_init-tilt + body->base_link mount correction upstream.
    This node just repackages Odometry -> PoseStamped for MAVROS.

    PX4 EKF2 params (QGroundControl):
        EKF2_EV_CTRL  = 11  (ev_pos + ev_yaw + ev_hgt)
        EKF2_HGT_REF  = 3   (vision height)
        EKF2_EV_DELAY = 0   (tune if you see lag, in ms)
        EKF2_EV_NOISE_MD = 0  (use covariance from the message)
    """

    def __init__(self):
        super().__init__('fastlio_bridge_node')
        self.declare_parameter('input_topic', '/Odometry_corrected')
        self.declare_parameter('output_topic', '/mavros/vision_pose/pose')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.sub = self.create_subscription(
            Odometry, input_topic, self.odometry_callback, 10)
        self.pub = self.create_publisher(PoseStamped, output_topic, 10)

        self.get_logger().info(
            f'fastlio_bridge_node started\n'
            f'  subscribing : {input_topic}\n'
            f'  publishing  : {output_topic}'
        )

    def odometry_callback(self, msg: Odometry):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp
        # Reuse the corrected message's own frame_id rather than hardcoding,
        # so this stays in sync if odom_frame_corrector's output_frame_id changes.
        pose_msg.header.frame_id = msg.header.frame_id
        pose_msg.pose = msg.pose.pose
        self.pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FastLioBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()