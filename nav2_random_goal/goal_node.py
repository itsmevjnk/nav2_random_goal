import rclpy
from rclpy import qos
from rclpy.node import Node

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial.transform import Rotation
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from action_msgs.msg import GoalStatusArray, GoalStatus
from sensor_msgs.msg import PointCloud2, PointField

import random

class Nav2RandomGoal(Node):
    def __init__(self):
        super().__init__('nav2_random_goal')

        self.sel_min = self.declare_parameter('sel_min', -128).get_parameter_value().integer_value
        self.sel_max = self.declare_parameter('sel_max', 75).get_parameter_value().integer_value

        self.empty_cells: list[Point] = []
        latched_qos = qos.QoSProfile(
            reliability=qos.QoSReliabilityPolicy.RELIABLE,
            depth=1,
            durability=qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_frame: str = 'map'
        self.create_subscription(OccupancyGrid, 'map', self.map_cb, latched_qos)

        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', qos.qos_profile_system_default)
        self.create_subscription(GoalStatusArray, 'goal_status', self.status_cb, latched_qos)

        self.points_pub = self.create_publisher(PointCloud2, 'goal_points', latched_qos)

    def publish_points(self, points: np.ndarray, frame_id: str = 'map'):
        npoints = points.shape[0]
        itemsz = np.dtype(np.float32).itemsize

        pcl_msg = PointCloud2()
        pcl_msg.header.frame_id = frame_id
        pcl_msg.header.stamp = self.get_clock().now().to_msg()
        pcl_msg.height = 1; pcl_msg.width = npoints
        pcl_msg.is_dense = False; pcl_msg.is_bigendian = False
        pcl_msg.fields = [
            PointField(name=n, offset=i*itemsz, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate('xyz')
        ]
        pcl_msg.point_step = itemsz * 3; pcl_msg.row_step = itemsz * 3 * npoints
        pcl_msg.data = points.astype(np.float32).tobytes()

        self.points_pub.publish(pcl_msg)

    def status_cb(self, data: GoalStatusArray):
        if len(data.status_list) == 0: return # nothing to do

        latest_status: GoalStatus = data.status_list[-1] # newest goal is last
        self.get_logger().info(f'latest goal status: {latest_status.status}')
        if latest_status.status >= 4: # 4 = succeeded, 5 = canceled, 6 = aborted
            yaw = (np.random.rand() * 2 - 1) * np.pi # [-pi, pi)
            goal = PoseStamped()
            goal.header.frame_id = self.map_frame
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position = random.choice(self.empty_cells)
            goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = Rotation.from_euler('z', yaw).as_quat()
            print(f'publishing goal (({goal.pose.position.x},{goal.pose.position.y},{goal.pose.position.z}),{yaw})')
            self.goal_pub.publish(goal)

    def map_cb(self, data: OccupancyGrid):
        resolution = data.info.resolution
        width = data.info.width
        height = data.info.height

        # make tf matrix
        origin_x = data.info.origin.position.x
        origin_y = data.info.origin.position.y
        origin_z = data.info.origin.position.z
        origin_rotmat = Rotation.from_quat([data.info.origin.orientation.x, data.info.origin.orientation.y, data.info.origin.orientation.z, data.info.origin.orientation.w]).as_matrix()
        tf_matrix = np.vstack([
            np.hstack([
                origin_rotmat,
                [[origin_x], [origin_y], [origin_z]]
            ]),
            [0., 0., 0., 1.]
        ])

        grid = np.reshape(data.data, (height, width))

        # select interior region
        filled_grid = binary_fill_holes((grid < self.sel_min) | (grid > self.sel_max))
        interior_mask = filled_grid & ((grid >= self.sel_min) & (grid <= self.sel_max))
        selected_region = np.argwhere(interior_mask)

        # selected_region = np.argwhere((grid >= self.sel_min) & (grid <= self.sel_max))

        selected_region[:, [0, 1]] = selected_region[:, [1, 0]] # swap to (x, y) order
        selected_pts = selected_region.shape[0]

        # filter occupancy grid
        empty_cells = resolution * (selected_region + 0.5).T
        empty_cells = np.vstack([
            empty_cells,
            np.zeros(selected_pts), # Z
            np.ones(selected_pts), # homogeneous
        ])
        empty_cells = tf_matrix.dot(empty_cells) # transform to map frame

        self.map_frame = data.header.frame_id
        self.publish_points(empty_cells[:3,].T, self.map_frame)

        self.get_logger().info(f'found {selected_pts} point(s) in frame {self.map_frame}')

        self.empty_cells = []
        for i in range(selected_pts):
            point = Point()
            point.x, point.y, point.z = empty_cells[:3, i].tolist()
            self.empty_cells.append(point)

def main():
    rclpy.init()
    node = Nav2RandomGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
