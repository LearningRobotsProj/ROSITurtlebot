# lr_ppo/setup.py
from setuptools import setup
import os
from glob import glob

package_name = 'lr_ppo'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ROSI Team',
    maintainer_email='team@example.com',
    description='PPO-based maze navigation for TurtleBot3',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'observation_vector = lr_ppo.observation_vector:main',
            'reward_function = lr_ppo.reward_function:main',
            'collision_checker = lr_ppo.collision_checker:main',
            'ppo_trainer = lr_ppo.trainer:main',
            'rosi_evaluator = lr_ppo.evaluator:main',
            'test_observation = lr_ppo.observation_vector:debug_main',
            'test_reward = lr_ppo.reward_function:debug_main',
            'test_collision = lr_ppo.collision_checker:debug_main',
            'test_ppo = lr_ppo.ppo_agent:debug_main',
        ],
    },
)