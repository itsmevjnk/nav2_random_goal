import rclpy
from rclpy import qos
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_list

class MapPointsSaver(Node):
    def __init__(self):
        super().__init__('nav2_random_goal')

        self.output = self.declare_parameter('output', 'points.csv').get_parameter_value().string_value

        latched_qos = qos.QoSProfile(
            reliability=qos.QoSReliabilityPolicy.RELIABLE,
            depth=1,
            durability=qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(PointCloud2, 'points', self.points_cb, latched_qos)

        self.get_logger().info('waiting for message')

    def points_cb(self, data: PointCloud2):
        frame_id = data.header.frame_id
        points = read_points_list(data, ['x', 'y', 'z'])
        self.get_logger().info(f'received {len(points)} point(s) in frame {frame_id}')
        with open(self.output, 'w') as f:
            f.write('frame_id,x,y,z\n')
            for p in points:
                f.write(f'{frame_id},{p.x},{p.y},{p.z}\n')
        
        self.get_logger().info(f'writing finished, shutting down')
        raise SystemExit


def main():
    rclpy.init()
    node = MapPointsSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass

    rclpy.shutdown()
