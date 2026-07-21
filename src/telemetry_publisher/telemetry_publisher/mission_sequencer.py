#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from mavros_msgs.srv import WaypointPull
from mavros_msgs.msg import WaypointList
import pymap3d as pm

MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22

handoff_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


gps_qos = QoSProfile(
    depth=10, 
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST)

class MissionSequencer(Node):
    def __init__(self):
        super().__init__('mission_sequencer')

        self.declare_parameter('goal_tolerance_m', 0.5)
        self.declare_parameter('goal_topic', '/move_base_simple/goal')  # CONFIRM against your FSM source
        self.declare_parameter('goal_frame_id', 'world')                # match grid_map/frame_id

        self.goal_tolerance = self.get_parameter('goal_tolerance_m').value
        goal_topic = self.get_parameter('goal_topic').value
        self.goal_frame_id = self.get_parameter('goal_frame_id').value

        # --- state ---
        self.anchor = None            # (lat, lon, alt) captured once
        self.current_pos = None       # (x, y, z) from /Odometry
        self.mission_items = None     # raw WaypointList from FC
        self.waypoints = []           # converted local-frame waypoints
        self.wp_index = 0
        self.mission_started = False
        self.enabled = False          # gated by /mission/enable from FlightManager

        # --- subscriptions ---
        self.sub_gps = self.create_subscription(
            NavSatFix, '/mavros/global_position/global', self.gps_cb, gps_qos)
        self.sub_odom = self.create_subscription(
            Odometry, '/Odometry', self.odom_cb, 10)
        self.sub_waypoints = self.create_subscription(
            WaypointList, '/mavros/mission/waypoints', self.waypoints_cb, 10)
        self.sub_enable = self.create_subscription(
            Bool, '/mission/enable', self.enable_cb, handoff_qos)

        # --- publishers ---
        self.pub_goal = self.create_publisher(PoseStamped, goal_topic, 10)
        self.pub_land_reached = self.create_publisher(Bool, '/mission/land_reached', handoff_qos)

        # --- service client ---
        self.wp_pull_client = self.create_client(WaypointPull, '/mavros/mission/pull')

        # --- timers ---
        self.create_timer(0.5, self.check_progress)
        self.create_timer(2.0, self.try_pull_once)  # retries the pull until it succeeds
        self._pull_requested = False

        self.get_logger().info('Waiting for GPS anchor, FC mission, and enable signal...')

    # ---------- callbacks ----------

    def gps_cb(self, msg: NavSatFix):
        if self.anchor is not None:
            return
        if msg.status.status < 0:
            return  # no fix yet
        self.anchor = (msg.latitude, msg.longitude, msg.altitude)
        self.get_logger().info(f'Anchor captured: {self.anchor}')
        self.try_load_mission()

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.current_pos = (p.x, p.y, p.z)

    def waypoints_cb(self, msg: WaypointList):
        self.mission_items = msg.waypoints
        self.get_logger().info(f'Received {len(self.mission_items)} waypoints from FC')
        self.try_load_mission()

    def enable_cb(self, msg: Bool):
        self.enabled = msg.data
        self.get_logger().info(f'Mission enable set to {self.enabled}')
        if self.enabled:
            self.try_load_mission()

    def try_pull_once(self):
        if self._pull_requested and self.mission_items is not None:
            return
        if not self.wp_pull_client.wait_for_service(timeout_sec=0.5):
            return
        req = WaypointPull.Request()
        self.wp_pull_client.call_async(req)
        self._pull_requested = True

    # ---------- mission loading ----------

    def try_load_mission(self):
        if self.mission_started:
            return
        if self.anchor is None or self.mission_items is None or not self.enabled:
            return

        self.waypoints = self.convert_waypoints(self.mission_items, self.anchor)
        if not self.waypoints:
            self.get_logger().error('No usable waypoints converted from FC mission.')
            return

        self.mission_started = True
        self.wp_index = 0
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints, starting sequence.')
        self.send_current_goal()

    def convert_waypoints(self, mission_items, anchor):
        home_lat, home_lon, home_alt = anchor
        waypoints = []
        for wp in mission_items:
            cmd = wp.command
            if cmd not in (MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_LAND, MAV_CMD_NAV_TAKEOFF):
                continue
            lat, lon, rel_alt = wp.x_lat, wp.y_long, wp.z_alt
            e, n, u = pm.geodetic2enu(lat, lon, home_alt + rel_alt,
                                       home_lat, home_lon, home_alt)
            waypoints.append({
                'command': cmd, 'x': e, 'y': n, 'z': u,
                'is_takeoff': cmd == MAV_CMD_NAV_TAKEOFF,
                'is_land': cmd == MAV_CMD_NAV_LAND,
            })
        return waypoints

    # ---------- sequencing ----------

    def send_current_goal(self):
        if self.wp_index >= len(self.waypoints):
            self.get_logger().info('Mission complete.')
            return

        wp = self.waypoints[self.wp_index]

        if wp['is_takeoff']:
            # Takeoff is handled by FlightManager before MISSION phase begins — skip it here.
            self.wp_index += 1
            self.send_current_goal()
            return

        if wp['is_land']:
            self.get_logger().info('Land item reached in sequence — notifying flight manager.')
            self.pub_land_reached.publish(Bool(data=True))
            return

        goal = PoseStamped()
        goal.header.frame_id = self.goal_frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = wp['x']
        goal.pose.position.y = wp['y']
        goal.pose.position.z = wp['z']
        goal.pose.orientation.w = 1.0

        self.pub_goal.publish(goal)
        self.get_logger().info(f'Sent goal {self.wp_index}: ({wp["x"]:.2f}, {wp["y"]:.2f}, {wp["z"]:.2f})')

    def check_progress(self):
        if not self.mission_started or not self.enabled or self.current_pos is None:
            return
        if self.wp_index >= len(self.waypoints):
            return

        wp = self.waypoints[self.wp_index]
        if wp['is_takeoff'] or wp['is_land']:
            return

        dx = self.current_pos[0] - wp['x']
        dy = self.current_pos[1] - wp['y']
        dz = self.current_pos[2] - wp['z']
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < self.goal_tolerance:
            self.get_logger().info(f'Reached waypoint {self.wp_index} (dist={dist:.2f}m). Advancing.')
            self.wp_index += 1
            self.send_current_goal()


def main():
    rclpy.init()
    node = MissionSequencer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
