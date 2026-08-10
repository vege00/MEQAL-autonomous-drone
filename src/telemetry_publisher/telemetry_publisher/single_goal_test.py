#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class SingleGoalTest(Node):
    def __init__(self):
        super().__init__('single_goal_test')

        self.declare_parameter('distance_m', 10.0)
        self.declare_parameter('goal_topic', '/move_base_simple/goal')
        self.declare_parameter('frame_id', 'world')  # match grid_map/frame_id
        self.declare_parameter('use_body_forward', True)  # False = straight +X in world frame

        self.distance = self.get_parameter('distance_m').value
        self.goal_topic = self.get_parameter('goal_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.use_body_forward = self.get_parameter('use_body_forward').value

        goal_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.pub_goal = self.create_publisher(PoseStamped, self.goal_topic, goal_qos)
        self.sub_odom = self.create_subscription(Odometry, '/Odometry', self.odom_cb, 10)

        self.sent = False
        self.get_logger().info('Waiting for first /Odometry message...')

    def odom_cb(self, msg: Odometry):
        if self.sent:
            return  # one-shot: only ever act on the first message received

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        if self.use_body_forward:
            # Extract yaw from quaternion, project distance_m forward along current heading
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            goal_x = p.x + self.distance * math.cos(yaw)
            goal_y = p.y + self.distance * math.sin(yaw)
        else:
            # Simpler: straight along world-frame +X, ignoring current heading entirely
            goal_x = p.x + self.distance
            goal_y = p.y

        goal_z = p.z  # hold current altitude

        goal = PoseStamped()
        goal.header.frame_id = self.frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = goal_z
        goal.pose.orientation.w = 1.0

        self.pub_goal.publish(goal)
        self.sent = True
        self.get_logger().info(
            f'Sent single test goal: ({goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f}) '
            f'from current pos ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})')


def main():
    rclpy.init()
    node = SingleGoalTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
