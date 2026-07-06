import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class FastLioBridgeNode(Node):
    """
    Bridges Fast-LIO odometry to MAVROS vision_pose for PX4 EKF2.

    Fast-LIO publishes on  : /Odometry  (frame: camera_init, FLU convention)
    MAVROS expects on      : /mavros/vision_pose/pose  (frame: map, ENU)

    Fast-LIO ROS2 branch uses a right-handed FLU frame (x-forward, y-left,
    z-up) which matches ROS/ENU convention directly, so no axis flip is needed.
    The frame_id is remapped from 'camera_init' to 'map' so MAVROS accepts it.

    PX4 EKF2 params to set in QGroundControl:
        EKF2_EV_CTRL  = 11  (bits: enable ev_pos + ev_yaw + ev_hgt)
        EKF2_HGT_REF  = 3   (vision height)
        EKF2_EV_DELAY = 0   (tune this if you see lag, in ms)
        EKF2_EV_NOISE_MD = 0  (use covariance from the message)
    """

    def __init__(self):
        super().__init__('fastlio_bridge_node')

        # --- Parameters ---
        self.declare_parameter('input_topic',  '/Odometry')
        self.declare_parameter('output_topic', '/mavros/vision_pose/pose')
        self.declare_parameter('map_frame_id', 'map')

        # Position/yaw covariance to report to PX4 EKF2.
        # Fast-LIO is very accurate — these are conservative defaults.
        # Lower = more trust given to Fast-LIO vs FC internal EKF.
        self.declare_parameter('pos_covariance',  0.01)   # metres²
        self.declare_parameter('att_covariance',  0.001)  # rad²

        input_topic  = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        # --- Sub / Pub ---
        self.sub = self.create_subscription(
            Odometry,
            input_topic,
            self.odometry_callback,
            10
        )
        self.pub = self.create_publisher(PoseStamped, output_topic, 10)

        self.get_logger().info(
            f'fastlio_bridge_node started\n'
            f'  subscribing : {input_topic}\n'
            f'  publishing  : {output_topic}'
        )

    def odometry_callback(self, msg: Odometry):
        pose_msg = PoseStamped()

        # Use the incoming timestamp so PX4 EKF2 can account for delay
        pose_msg.header.stamp    = msg.header.stamp
        pose_msg.header.frame_id = self.get_parameter('map_frame_id').value

        # Pass position straight through — Fast-LIO ROS2 is already ENU/FLU
        pose_msg.pose.position.x = msg.pose.pose.position.x
        pose_msg.pose.position.y = msg.pose.pose.position.y
        pose_msg.pose.position.z = msg.pose.pose.position.z

        # Pass orientation straight through
        pose_msg.pose.orientation.x = msg.pose.pose.orientation.x
        pose_msg.pose.orientation.y = msg.pose.pose.orientation.y
        pose_msg.pose.orientation.z = msg.pose.pose.orientation.z
        pose_msg.pose.orientation.w = msg.pose.pose.orientation.w

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
