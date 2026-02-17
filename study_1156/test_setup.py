#!/usr/bin/env python3

"""
Test script to verify ROS 2 Image Processor setup
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'rclpy',
        'sensor_msgs',
        'cv_bridge',
        'cv2',
        'numpy'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_opencv_features():
    """Test OpenCV features used in the project"""
    try:
        import cv2
        
        print("\nTesting OpenCV features...")
        
        # Test HOG detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("✓ HOG people detector")
        
        # Test Haar cascade
        car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_cars.xml')
        if not car_cascade.empty():
            print("✓ Haar cascade car detector")
        else:
            print("⚠ Haar cascade car detector not available (will use fallback)")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenCV features: {e}")
        return False

def test_ros2_environment():
    """Test ROS 2 environment"""
    try:
        import rclpy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        
        print("\nTesting ROS 2 environment...")
        print("✓ rclpy")
        print("✓ sensor_msgs")
        print("✓ cv_bridge")
        
        # Test CV bridge
        bridge = CvBridge()
        print("✓ CV bridge initialization")
        
        return True
        
    except Exception as e:
        print(f"✗ ROS 2 environment: {e}")
        return False

def main():
    print("ROS 2 Image Processor - Setup Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test OpenCV features
    if not test_opencv_features():
        all_tests_passed = False
    
    # Test ROS 2 environment
    if not test_ros2_environment():
        all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("✓ All tests passed! Setup is ready.")
        print("\nYou can now run:")
        print("  docker-compose up")
        print("  or")
        print("  ros2 launch ros2_image_processor image_processor.launch.py")
    else:
        print("✗ Some tests failed. Please check the setup.")
        print("\nCommon solutions:")
        print("  1. Install missing packages: pip3 install <package_name>")
        print("  2. Source ROS 2: source /opt/ros/humble/setup.bash")
        print("  3. Use Docker: docker-compose build")
        sys.exit(1)

if __name__ == '__main__':
    main() 