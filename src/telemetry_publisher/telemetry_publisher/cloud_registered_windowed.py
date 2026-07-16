#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from collections import deque
import time
from numpy.lib.recfunctions import structured_to_unstructured

class CloudAggregator(Node):
    def __init__(self):
        super().__init__('cloud_aggregator')

        self.declare_parameter('window_seconds', 1.5)
        self.declare_parameter('voxel_size', 0.1)          # match grid_map/resolution
        self.declare_parameter('crop_radius_xy', 6.0)      # slightly > local_update_range_x/y
        self.declare_parameter('crop_radius_z', 5.0)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('cloud_topic', '/cloud_registered')
        self.declare_parameter('odom_topic', '/Odometry')
        self.declare_parameter('output_topic', '/cloud_registered_windowed')

        self.window_seconds = self.get_parameter('window_seconds').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.crop_xy = self.get_parameter('crop_radius_xy').value
        self.crop_z = self.get_parameter('crop_radius_z').value

        self.buffer = deque()  # list of (timestamp, Nx3 np.array)
        self.latest_pos = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)

        self.sub_cloud = self.create_subscription(
            PointCloud2, self.get_parameter('cloud_topic').value, self.cloud_cb, qos)
        self.sub_odom = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_cb, 10)
        self.pub = self.create_publisher(
            PointCloud2, self.get_parameter('output_topic').value, qos)

        rate = self.get_parameter('publish_rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self.publish_merged)
        self.last_header = None

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.latest_pos = np.array([p.x, p.y, p.z])

    def cloud_cb(self, msg: PointCloud2):
        pts = np.array(list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)))
        if pts.size == 0:
            return
        now = time.time()
        self.buffer.append((now, pts))
        self.last_header = msg.header

        cutoff = now - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def voxel_downsample(self, pts: np.ndarray, voxel: float) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        keys = np.floor(pts / voxel).astype(np.int64)
        # dedupe by voxel key, keep first occurrence
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        return pts[unique_idx]

    def publish_merged(self):
        if not self.buffer or self.last_header is None:
            return

        merged = np.concatenate([p for _, p in self.buffer], axis=0)

        if self.latest_pos is not None:
            d = merged - self.latest_pos
            mask = (np.abs(d[:, 0]) < self.crop_xy) & \
                   (np.abs(d[:, 1]) < self.crop_xy) & \
                   (np.abs(d[:, 2]) < self.crop_z)
            merged = merged[mask]

        merged = self.voxel_downsample(merged, self.voxel_size)

        header = self.last_header
        header.stamp = self.get_clock().now().to_msg()
        cloud_msg = pc2.create_cloud_xyz32(header, merged.tolist())
        self.pub.publish(cloud_msg)

    def cloud_cb(self, msg: PointCloud2):
        pts_struct = np.array(list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)))
        if pts_struct.size == 0:
            return
        pts = structured_to_unstructured(pts_struct).astype(np.float64)   # <-- add this line
        now = time.time()
        self.buffer.append((now, pts))
        self.last_header = msg.header

def main():
    rclpy.init()
    node = CloudAggregator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
