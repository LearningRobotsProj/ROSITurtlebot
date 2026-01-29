from setuptools import setup
import os
from glob import glob

package_name = 'lr_turtlebot_sim'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'),
         glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ROSI Team',
    maintainer_email='team@lecnam.net',
    description='TurtleBot3 maze simulation for ROSI PPO project',
    license='Apache-2.0',

    # ❌ REMOVED tests_require (not supported by colcon)

    entry_points={
        'console_scripts': [
            'maze_simulator = lr_turtlebot_sim.maze_simulator:main',
            'spawn_robot = lr_turtlebot_sim.spawn_robot:main',
            'test_node = lr_turtlebot_sim.test_spawn:main',
        ],
    },
)
