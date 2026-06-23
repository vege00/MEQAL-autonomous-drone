import rclpy
from rclpy.node import Node
import asyncio
from mavsdk import System
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import numpy as np
import threading


class TelemetryNode(Node):
    def __init__(self):
        super().__init__('telemetry_node')

        # --- Parameters (override via ROS2 params or launch file) ---
        self.declare_parameter('connection_url', 'serial:///dev/ttyACM0:57600')
        self.declare_parameter('imu_rate_hz', 200.0)
        self.declare_parameter('odom_rate_hz', 33.0)   # ~30Hz publish rate
        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')

        # ICM-42688-P datasheet noise values — adjust if using a different IMU
        self.declare_parameter('gyro_noise', 1.0e-4)      # rad/s
        self.declare_parameter('accel_noise', 1.0e-3)     # m/s²
        self.declare_parameter('gyro_bias_noise', 1.0e-6)
        self.declare_parameter('accel_bias_noise', 1.0e-5)

        conn_url    = self.get_parameter('connection_url').value
        imu_rate    = self.get_parameter('imu_rate_hz').value
        odom_rate   = self.get_parameter('odom_rate_hz').value

        # --- Publishers ---
        self.odom_pub = self.create_publisher(Odometry, 'telemetry/odometry', 10)
        self.imu_pub  = self.create_publisher(Imu,      'telemetry/imu',      10)

        # --- Shared state ---
        self.odom_data = None
        self.imu_data  = None
        self._lock     = threading.Lock()

        # --- Async thread ---
        self.async_thread = threading.Thread(
            target=self._run_async_loop,
            args=(conn_url, imu_rate),
            daemon=True
        )
        self.async_thread.start()

        # --- Publish timers ---
        self.create_timer(1.0 / odom_rate, self.publish_odometry)
        self.create_timer(1.0 / imu_rate,  self.publish_imu)

        self.get_logger().info(
            f'TelemetryNode started — connecting to FC at {conn_url}'
        )

    # ------------------------------------------------------------------ #
    #  Async loop                                                          #
    # ------------------------------------------------------------------ #

    def _run_async_loop(self, conn_url: str, imu_rate: float):
        asyncio.run(self._connect_and_subscribe(conn_url, imu_rate))

    async def _connect_and_subscribe(self, conn_url: str, imu_rate: float):
        drone = System()
        await drone.connect(conn_url)

        self.get_logger().info('Waiting for FC connection...')
        async for state in drone.core.connection_state():
            if state.is_connected:
                self.get_logger().info('FC connected!')
                break

        # Request high-rate IMU stream from the FC
        try:
            await drone.telemetry.set_rate_imu(imu_rate)
            self.get_logger().info(f'IMU stream rate set to {imu_rate} Hz')
        except Exception as e:
            self.get_logger().warn(f'Could not set IMU rate: {e}')

        # Run both subscriptions concurrently
        await asyncio.gather(
            self._subscribe_odometry(drone),
            self._subscribe_imu(drone),
        )

    async def _subscribe_odometry(self, drone):
        async for odom in drone.telemetry.odometry():
            with self._lock:
                self.odom_data = odom

    async def _subscribe_imu(self, drone):
        async for imu in drone.telemetry.imu():
            with self._lock:
                self.imu_data = imu

    # ------------------------------------------------------------------ #
    #  Publish: Odometry                                                   #
    # ------------------------------------------------------------------ #

    def publish_odometry(self):
        with self._lock:
            odom = self.odom_data

        if odom is None:
            self.get_logger().warn('No odometry data yet...', throttle_duration_sec=5.0)
            return

        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('odom_frame_id').value
        msg.child_frame_id  = self.get_parameter('base_frame_id').value

        # Position (body frame from FC)
        msg.pose.pose.position.x = odom.position_body.x_m
        msg.pose.pose.position.y = odom.position_body.y_m
        msg.pose.pose.position.z = odom.position_body.z_m

        # Orientation quaternion
        msg.pose.pose.orientation.w = odom.q.w
        msg.pose.pose.orientation.x = odom.q.x
        msg.pose.pose.orientation.y = odom.q.y
        msg.pose.pose.orientation.z = odom.q.z

        # Linear + angular velocity
        msg.twist.twist.linear.x  = odom.velocity_body.x_m_s
        msg.twist.twist.linear.y  = odom.velocity_body.y_m_s
        msg.twist.twist.linear.z  = odom.velocity_body.z_m_s
        msg.twist.twist.angular.x = odom.angular_velocity_body.roll_rad_s
        msg.twist.twist.angular.y = odom.angular_velocity_body.pitch_rad_s
        msg.twist.twist.angular.z = odom.angular_velocity_body.yaw_rad_s

        # Covariances (upper-triangle → full 6×6)
        msg.pose.covariance  = self._upper_tri_to_full(
            odom.pose_covariance.covariance_matrix
        )
        msg.twist.covariance = self._upper_tri_to_full(
            odom.velocity_covariance.covariance_matrix
        )

        self.odom_pub.publish(msg)

    # ------------------------------------------------------------------ #
    #  Publish: Raw IMU  (for FAST_LIO)                                   #
    # ------------------------------------------------------------------ #

    def publish_imu(self):
        with self._lock:
            imu = self.imu_data

        if imu is None:
            self.get_logger().warn('No IMU data yet...', throttle_duration_sec=5.0)
            return

        gyro_n  = self.get_parameter('gyro_noise').value
        accel_n = self.get_parameter('accel_noise').value

        msg = Imu()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('imu_frame_id').value

        # --- Frame convention ---
        # MAVSDK uses FRD (Forward-Right-Down).
        # ROS uses FLU (Forward-Left-Up).
        # Conversion: flip sign on Y and Z axes.
        msg.linear_acceleration.x =  imu.acceleration_frd.forward_m_s2
        msg.linear_acceleration.y = -imu.acceleration_frd.right_m_s2
        msg.linear_acceleration.z = -imu.acceleration_frd.down_m_s2

        msg.angular_velocity.x =  imu.angular_velocity_frd.forward_rad_s
        msg.angular_velocity.y = -imu.angular_velocity_frd.right_rad_s
        msg.angular_velocity.z = -imu.angular_velocity_frd.down_rad_s

        # No orientation from raw IMU — signal this with covariance[0] = -1
        msg.orientation_covariance[0] = -1.0

        # Diagonal noise covariance (3×3 stored as flat 9-element array)
        msg.angular_velocity_covariance[0] = gyro_n
        msg.angular_velocity_covariance[4] = gyro_n
        msg.angular_velocity_covariance[8] = gyro_n

        msg.linear_acceleration_covariance[0] = accel_n
        msg.linear_acceleration_covariance[4] = accel_n
        msg.linear_acceleration_covariance[8] = accel_n

        self.imu_pub.publish(msg)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _upper_tri_to_full(upper_tri) -> list:
        """Convert an upper-triangle covariance list to a flat 6×6 matrix."""
        mat = np.zeros((6, 6))
        idx = 0
        for row in range(6):
            for col in range(row, 6):
                val = float(upper_tri[idx]) if idx < len(upper_tri) else 0.0
                val = 0.0 if np.isnan(val) else val
                mat[row, col] = val
                mat[col, row] = val
                idx += 1
        return mat.flatten().tolist()


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main(args=None):
    rclpy.init(args=args)
    node = TelemetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()