#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32MultiArray


# ============================================================
# Quaternion → Yaw (NumPy 2.0 SAFE, no tf_transformations)
# ============================================================
def quaternion_to_yaw(qx, qy, qz, qw):
    """
    Convert quaternion to yaw (Z-axis rotation)
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


# ============================================================
# Observation Vector Node
# ============================================================
class ObservationVector(Node):

    def __init__(self, node_name='observation_vector'):
        super().__init__(node_name)

        # ---------------- Parameters ----------------
        self.declare_parameter('lidar_normalization_range', [0.0, 3.5])
        self.declare_parameter('num_lidar_samples', 20)
        self.declare_parameter('use_goal_position', True)
        self.declare_parameter('publish_rate', 10.0)  # Hz

        # ---------------- Subscribers ----------------
        # Use best effort QoS for faster updates
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.create_subscription(
            LaserScan, '/scan', 
            self.lidar_callback, 
            qos_profile
        )
        self.create_subscription(
            Odometry, '/odom', 
            self.odom_callback, 
            qos_profile
        )
        self.create_subscription(
            PoseStamped, '/goal_pose', 
            self.goal_callback, 
            10
        )
        self.create_subscription(
            Twist, '/cmd_vel', 
            self.cmd_vel_callback, 
            10
        )

        # ---------------- Publisher ----------------
        self.obs_pub = self.create_publisher(
            Float32MultiArray, 
            '/normalized_observation', 
            10
        )
        
        print(f"✓ Created publisher for /normalized_observation")

        # ---------------- Buffers ----------------
        self.lidar_data = None
        self.odom_data = None
        self.goal_data = None
        self.cmd_vel_data = Twist()
        
        self.observation_ready = False
        self.last_obs_time = self.get_clock().now()

        # ---------------- Timer for publishing ----------------
        publish_rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0/publish_rate, self.publish_observation)
        
        print(f"Observation Vector node initialized - Publishing at {publish_rate}Hz")
        print("Waiting for sensor data...")

    # ===============================
    # Callbacks
    # ===============================
    def lidar_callback(self, msg):
        try:
            num_samples = self.get_parameter('num_lidar_samples').value
            step = max(1, len(msg.ranges) // num_samples)

            samples = []
            for i in range(0, len(msg.ranges), step):
                if len(samples) >= num_samples:
                    break
                r = msg.ranges[i]
                if np.isnan(r) or np.isinf(r):
                    r = msg.range_max
                samples.append(r)

            while len(samples) < num_samples:
                samples.append(msg.range_max)

            self.lidar_data = np.array(samples, dtype=np.float32)
            
            if not hasattr(self, 'lidar_received'):
                self.lidar_received = True
                print("✓ LiDAR data received")

        except Exception as e:
            print(f"LiDAR callback error: {e}")

    def odom_callback(self, msg):
        try:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation

            yaw = quaternion_to_yaw(ori.x, ori.y, ori.z, ori.w)

            self.odom_data = {
                'position': [pos.x, pos.y],
                'orientation': yaw,
                'linear_velocity': msg.twist.twist.linear.x,
                'angular_velocity': msg.twist.twist.angular.z
            }
            
            if not hasattr(self, 'odom_received'):
                self.odom_received = True
                print("✓ Odometry data received")

        except Exception as e:
            print(f"Odom callback error: {e}")

    def goal_callback(self, msg):
        self.goal_data = [msg.pose.position.x, msg.pose.position.y]
        
        if not hasattr(self, 'goal_received'):
            self.goal_received = True
            print(f"✓ Goal data received: {self.goal_data}")

    def cmd_vel_callback(self, msg):
        self.cmd_vel_data = msg

    # ===============================
    # Helpers
    # ===============================
    def normalize_lidar(self, lidar):
        min_v, max_v = self.get_parameter('lidar_normalization_range').value
        lidar = np.clip(lidar, min_v, max_v)
        return (lidar - min_v) / (max_v - min_v + 1e-8)

    def compute_relative_goal(self):
        if self.odom_data is None or self.goal_data is None:
            return [0.0, 0.0, 1.0]

        rx, ry = self.odom_data['position']
        yaw = self.odom_data['orientation']
        gx, gy = self.goal_data

        dx = gx - rx
        dy = gy - ry

        rel_x = dx * np.cos(yaw) + dy * np.sin(yaw)
        rel_y = -dx * np.sin(yaw) + dy * np.cos(yaw)

        dist = np.sqrt(rel_x**2 + rel_y**2)
        max_dist = 10.0

        return [
            rel_x / max_dist,
            rel_y / max_dist,
            np.clip(dist / max_dist, 0.0, 1.0)
        ]

    # ===============================
    # Create Observation
    # ===============================
    def create_observation(self):
        """Create observation vector from current data"""
        
        # Check if we have minimum required data
        if self.lidar_data is None or self.odom_data is None:
            # Return default observation if no data
            obs = np.zeros(29, dtype=np.float32)
            obs[0:20] = 1.0  # Open space
            return obs
        
        obs = []

        # 1️⃣ LiDAR (20 values)
        lidar_norm = self.normalize_lidar(self.lidar_data)
        obs.extend(lidar_norm.tolist())

        # 2️⃣ Position + Orientation (4 values)
        x, y = self.odom_data['position']
        yaw = self.odom_data['orientation']
        obs.extend([x / 5.0, y / 5.0, np.sin(yaw), np.cos(yaw)])

        # 3️⃣ Velocities (2 values)
        obs.append(self.odom_data['linear_velocity'] / 0.22)
        obs.append(self.odom_data['angular_velocity'] / 2.84)

        # 4️⃣ Goal (3 values)
        if self.get_parameter('use_goal_position').value:
            obs.extend(self.compute_relative_goal())
        else:
            obs.extend([0.0, 0.0, 1.0])  # Default goal info

        obs = np.array(obs, dtype=np.float32)
        self.observation_ready = True
        return obs

    # ===============================
    # Publish Observation
    # ===============================
    def publish_observation(self):
        """Timer callback to publish observation"""
        obs = self.create_observation()
        
        # Create and publish message
        msg = Float32MultiArray()
        msg.data = obs.tolist()
        
        try:
            self.obs_pub.publish(msg)
            self.last_obs_time = self.get_clock().now()
            
            # Print first few publishes
            if not hasattr(self, 'first_publish_done'):
                self.first_publish_done = True
                print(f"\n✓ First observation published!")
                print(f"  Shape: {len(obs)} values")
                print(f"  Lidar mean: {np.mean(obs[0:20]):.3f}")
                print(f"  Position: [{obs[20]:.3f}, {obs[21]:.3f}]")
                
        except Exception as e:
            print(f"Error publishing observation: {e}")

    def get_observation(self):
        """Get current observation (for external calls)"""
        return self.create_observation()


# ===============================
# Main
# ===============================
def main(args=None):
    rclpy.init(args=args)
    
    print("\n" + "="*60)
    print("OBSERVATION VECTOR NODE")
    print("="*60)
    
    node = ObservationVector()
    
    # Check initial state
    print("\nNode status:")
    print(f"  Subscribers created: ✓")
    print(f"  Publisher created: ✓")
    print(f"  Publishing to: /normalized_observation")
    
    try:
        print("\nStarting node...")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n\nNode stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("\nNode shutdown complete")


if __name__ == "__main__":
    main()