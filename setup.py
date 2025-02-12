from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'nav2_random_goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='itsmevjnk',
    maintainer_email='ngtv0404@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goal_node = nav2_random_goal.goal_node:main',
            'points_node = nav2_random_goal.points_node:main',
            'points_saver = nav2_random_goal.points_saver:main',
            'points_server_node = nav2_random_goal.points_server_node:main',
        ],
    },
)
