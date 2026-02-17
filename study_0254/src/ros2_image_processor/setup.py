from setuptools import setup
import os
from glob import glob

package_name = 'ros2_image_processor'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ROS 2 Developer',
    maintainer_email='developer@example.com',
    description='ROS 2 node for image stitching and object tracking from dual camera streams',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_processor = ros2_image_processor.image_processor_node:main',
        ],
    },
) 