import rclpy
from rclpy import qos
from rclpy.node import Node

import csv
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

class MapPointsServer(Node):
    def __init__(self):
        super().__init__('nav2_map_points_server')

        file = self.declare_parameter('points_file', 'points.csv').get_parameter_value().string_value
        points: list[tuple[float, float, float]] = []
        map_frame = 'map'
        with open(file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append((row['x'], row['y'], row['z']))
                map_frame = row['frame_id']

        latched_qos = qos.QoSProfile(
            reliability=qos.QoSReliabilityPolicy.RELIABLE,
            depth=1,
            durability=qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.points_pub = self.create_publisher(PointCloud2, 'points', latched_qos)

        self.get_logger().info(f'serving {len(points)} point(s) in frame {map_frame}')
        self.publish_points(np.array(points), map_frame)        

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

def main():
    rclpy.init()
    node = MapPointsServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
