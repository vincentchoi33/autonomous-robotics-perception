#!/usr/bin/env python3

"""
Script to inspect ROS bag file and list available topics
"""

import subprocess
import sys
import os

def check_bag_info(bag_path):
    """Check bag file information"""
    try:
        # Check if bag file exists
        if not os.path.exists(bag_path):
            print(f"✗ Bag file not found: {bag_path}")
            return False
        
        print(f"✓ Bag file found: {bag_path}")
        
        # Get bag info
        result = subprocess.run(['ros2', 'bag', 'info', bag_path], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\nBag file information:")
            print(result.stdout)
        else:
            print(f"Error getting bag info: {result.stderr}")
            return False
        
        return True
        
    except FileNotFoundError:
        print("✗ ros2 command not found. Make sure ROS 2 is installed and sourced.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def list_bag_topics(bag_path):
    """List topics in the bag file"""
    try:
        result = subprocess.run(['ros2', 'bag', 'list', bag_path], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\nTopics in bag file:")
            print(result.stdout)
        else:
            print(f"Error listing topics: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error listing topics: {e}")
        return False

def main():
    bag_path = "rosbag2_2025_06_16-15_16_29"
    
    print("ROS 2 Bag File Inspector")
    print("=" * 30)
    
    # Check bag info
    if not check_bag_info(bag_path):
        sys.exit(1)
    
    # List topics
    if not list_bag_topics(bag_path):
        sys.exit(1)
    
    print("\n" + "=" * 30)
    print("✓ Bag file inspection complete!")
    print("\nExpected topics for this project:")
    print("  - /camera_left/image_raw")
    print("  - /camera_right/image_raw")

if __name__ == '__main__':
    main() 