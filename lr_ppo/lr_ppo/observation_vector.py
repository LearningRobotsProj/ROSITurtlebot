# lr_ppo/lr_ppo/observation_vector.py
#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32MultiArray
import tf_transformations

class ObservationVector(Node):
    def __init__(self, node_name='observation_vector'):
        super().__init__(node_name)
        
        # Parameters
        self.declare_parameter('lidar_normalization_range', [0.0, 3.5])
        self.declare_parameter('num_lidar_samples', 20)
        self.declare_parameter('use_goal_position', True)
        
        # Subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        
        # Publisher for normalized observation
        self.obs_pub = self.create_publisher(Float32MultiArray, '/normalized_observation', 10)
        
        # Data buffers
        self.lidar_data = None
        self.odom_data = None
        self.goal_data = None
        self.cmd_vel_data = Twist()
        
        self.get_logger().info("Observation Vector node initialized")
    
    def lidar_callback(self, msg):
        try:
            num_samples = self.get_parameter('num_lidar_samples').value
            step = max(1, len(msg.ranges) // num_samples)
            
            sampled_ranges = []
            for i in range(0, len(msg.ranges), step):
                if len(sampled_ranges) >= num_samples:
                    break
                
                range_val = msg.ranges[i]
                if np.isinf(range_val) or np.isnan(range_val):
                    range_val = msg.range_max
                sampled_ranges.append(range_val)
            
            # Pad if needed
            while len(sampled_ranges) < num_samples:
                sampled_ranges.append(msg.range_max)
            
            self.lidar_data = np.array(sampled_ranges[:num_samples], dtype=np.float32)
            
        except Exception as e:
            self.get_logger().error(f"Error in lidar callback: {e}")
    
    def odom_callback(self, msg):
        try:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            
            # Convert quaternion to yaw
            euler = tf_transformations.euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
            yaw = euler[2]
            
            self.odom_data = {
                'position': [position.x, position.y],
                'orientation': yaw,
                'linear_velocity': msg.twist.twist.linear.x,
                'angular_velocity': msg.twist.twist.angular.z
            }
            
        except Exception as e:
            self.get_logger().error(f"Error in odom callback: {e}")
    
    def goal_callback(self, msg):
        self.goal_data = [msg.pose.position.x, msg.pose.position.y]
    
    def cmd_vel_callback(self, msg):
        self.cmd_vel_data = msg
    
    def normalize_lidar(self, lidar_values):
        min_val, max_val = self.get_parameter('lidar_normalization_range').value
        clipped = np.clip(lidar_values, min_val, max_val)
        normalized = (clipped - min_val) / (max_val - min_val)
        return normalized
    
    def compute_relative_goal(self):
        if self.odom_data is None or self.goal_data is None:
            return [0.0, 0.0, 1.0]
        
        robot_x, robot_y = self.odom_data['position']
        robot_yaw = self.odom_data['orientation']
        goal_x, goal_y = self.goal_data
        
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        
        rel_x = dx * np.cos(robot_yaw) + dy * np.sin(robot_yaw)
        rel_y = -dx * np.sin(robot_yaw) + dy * np.cos(robot_yaw)
        
        distance = np.sqrt(rel_x**2 + rel_y**2)
        max_distance = 10.0
        
        return [
            rel_x / max_distance,
            rel_y / max_distance,
            np.clip(distance / max_distance, 0.0, 1.0)
        ]
    
    def create_observation_vector(self):
        if self.lidar_data is None or self.odom_data is None:
            return None
        
        observation_parts = []
        
        # 1. Normalized LiDAR (20 values)
        normalized_lidar = self.normalize_lidar(self.lidar_data)
        observation_parts.extend(normalized_lidar.tolist())
        
        # 2. Odometry (4 values)
        pos_x, pos_y = self.odom_data['position']
        observation_parts.append(pos_x / 5.0)  # Normalize to [-1, 1]
        observation_parts.append(pos_y / 5.0)
        
        yaw = self.odom_data['orientation']
        observation_parts.append(np.sin(yaw))
        observation_parts.append(np.cos(yaw))
        
        # 3. Velocities (2 values)
        max_linear_vel = 0.22
        max_angular_vel = 2.84
        
        linear_vel = self.odom_data['linear_velocity']
        angular_vel = self.odom_data['angular_velocity']
        
        observation_parts.append(linear_vel / max_linear_vel)
        observation_parts.append(angular_vel / max_angular_vel)
        
        # 4. Relative goal (3 values)
        if self.get_parameter('use_goal_position').value:
            rel_goal = self.compute_relative_goal()
            observation_parts.extend(rel_goal)
        
        obs_array = np.array(observation_parts, dtype=np.float32)
        
        # Publish observation
        if self.obs_pub.get_subscription_count() > 0:
            msg = Float32MultiArray()
            msg.data = obs_array.tolist()
            self.obs_pub.publish(msg)
        
        return obs_array
    
    def get_observation(self):
        return self.create_observation_vector()

def main(args=None):
    rclpy.init(args=args)
    node = ObservationVector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def debug_main():
    """Debug function to test observation vector."""
    print("=== Testing Observation Vector ===")
    
    # Initialize ROS2
    rclpy.init()
    
    # Create dummy node
    dummy_node = ObservationVector('test_obs')
    
    # Simulate data
    dummy_node.lidar_data = np.random.rand(20) * 3.5
    dummy_node.odom_data = {
        'position': [1.0, 2.0],
        'orientation': 0.5,
        'linear_velocity': 0.1,
        'angular_velocity': 0.2
    }
    dummy_node.goal_data = [3.0, 4.0]
    
    # Create observation
    obs = dummy_node.create_observation_vector()
    
    if obs is not None:
        print(f"✓ Observation vector created successfully")
        print(f"✓ Shape: {obs.shape}")
        print(f"✓ Range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"✓ Contains NaN: {np.any(np.isnan(obs))}")
        print(f"✓ Contains Inf: {np.any(np.isinf(obs))}")
        
        # Check normalization
        if obs.min() >= -1.0 and obs.max() <= 1.0:
            print("✓ Normalization within expected range")
        else:
            print("✗ Normalization outside expected range")
    else:
        print("✗ Failed to create observation vector")
    
    # Clean up
    dummy_node.destroy_node()
    rclpy.shutdown()
    
    return obs is not None

if __name__ == "__main__":
    main()