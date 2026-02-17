#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Declare launch arguments
    bag_file_arg = DeclareLaunchArgument(
        'bag_file',
        default_value='rosbag2_2025_06_16-15_16_29',
        description='Path to the ROS bag file'
    )
    
    # Get the package share directory
    pkg_share = FindPackageShare('ros2_image_processor')
    
    # Image processor node
    image_processor_node = Node(
        package='ros2_image_processor',
        executable='image_processor',
        name='image_processor_node',
        output='screen',
        parameters=[{
            'left_camera_topic': '/camera_left/image_raw',
            'right_camera_topic': '/camera_right/image_raw',
            'processed_image_topic': '/processed_image'
        }]
    )
    
    # ROS bag play command
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_file')],
        output='screen'
    )
    
    return LaunchDescription([
        bag_file_arg,
        image_processor_node,
        bag_play
    ]) 