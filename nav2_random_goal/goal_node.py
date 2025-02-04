import rclpy
from rclpy import qos
from rclpy.node import Node

import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped, Point
from action_msgs.msg import GoalStatusArray, GoalStatus

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_list

import random

class Nav2RandomGoal(Node):
    def __init__(self):
        super().__init__('nav2_random_goal')

        self.empty_cells: list[Point] = []
        latched_qos = qos.QoSProfile(
            reliability=qos.QoSReliabilityPolicy.RELIABLE,
            depth=1,
            durability=qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_frame: str = 'map'
        self.create_subscription(PointCloud2, 'goal_points', self.points_cb, latched_qos)

        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', qos.qos_profile_system_default)
        self.create_subscription(GoalStatusArray, 'goal_status', self.status_cb, latched_qos)

    def status_cb(self, data: GoalStatusArray):
        if len(data.status_list) == 0: return # nothing to do

        latest_status: GoalStatus = data.status_list[-1] # newest goal is last
        self.get_logger().info(f'latest goal status: {latest_status.status}')
        if latest_status.status >= 4: # 4 = succeeded, 5 = canceled, 6 = aborted
            if len(self.empty_cells) == 0: return # no cells
            yaw = (np.random.rand() * 2 - 1) * np.pi # [-pi, pi)
            goal = PoseStamped()
            goal.header.frame_id = self.map_frame
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position = random.choice(self.empty_cells)
            goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = Rotation.from_euler('z', yaw).as_quat()
            print(f'publishing goal (({goal.pose.position.x},{goal.pose.position.y},{goal.pose.position.z}),{yaw})')
            self.goal_pub.publish(goal)

    def points_cb(self, data: PointCloud2):
        points = read_points_list(data, ['x', 'y', 'z'])
        self.empty_cells = []
        for p in points:
            point = Point()
            point.x = p.x.item()
            point.y = p.y.item()
            point.z = p.z.item()
            self.empty_cells.append(point)

def main():
    rclpy.init()
    node = Nav2RandomGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
