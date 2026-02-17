#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import threading
import time

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Image storage
        self.left_image = None
        self.right_image = None
        self.left_image_lock = threading.Lock()
        self.right_image_lock = threading.Lock()
        
        # Object tracking variables
        self.tracked_objects = {}  # Dictionary to store tracked objects
        self.object_id_counter = 0
        self.tracking_history = {}  # Store motion paths for each object
        self.max_history_length = 30  # Number of points to keep in motion path
        
        # Initialize object detection
        # self.init_object_detection()
        
        # Create subscribers
        self.left_camera_sub = self.create_subscription(
            Image,
            '/realsense/right/color/image_raw_throttle',
            self.left_camera_callback,
            10
        )
        
        self.right_camera_sub = self.create_subscription(
            Image,
            '/realsense/left/color/image_raw_throttle',
            self.right_camera_callback,
            10
        )
        
        # Create publisher for processed image
        self.processed_image_pub = self.create_publisher(
            Image,
            '/processed_image',
            10
        )
        
        # Create timer for processing and publishing
        self.timer = self.create_timer(0.1, self.process_and_publish)  # 10 FPS
        
        # Stitching method selection (change this to test different methods)
        self.stitching_method = 'side_by_side'  # Options: 'side_by_side', 'feature_based', 'overlap'
        
        # Image rotation settings
        self.rotate_left = True   # Set to True if left camera image needs rotation
        self.rotate_right = True  # Set to True if right camera image needs rotation
        self.rotation_angle = cv2.ROTATE_90_COUNTERCLOCKWISE  # Options: ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
        
        self.get_logger().info('Image Processor Node started')
        self.get_logger().info(f'Using stitching method: {self.stitching_method}')
        self.get_logger().info(f'Image rotation: Left={self.rotate_left}, Right={self.rotate_right}')
        
    # def init_object_detection(self):
    #     """Initialize object detection models for humans and vehicles"""
    #     # Load pre-trained models for human and vehicle detection
    #     # Using HOG for human detection and Haar cascades for vehicle detection
        
    #     # Human detection using HOG
    #     self.hog = cv2.HOGDescriptor()
    #     self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
    #     # Vehicle detection using Haar cascade
    #     self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_cars.xml')
        
    #     # If car cascade is not available, we'll use a simpler approach
    #     if self.car_cascade.empty():
    #         self.get_logger().warn('Car cascade not found, using alternative detection method')
    #         self.car_cascade = None
    
    def left_camera_callback(self, msg):
        """Callback for left camera images"""
        try:
            with self.left_image_lock:
                # Convert ROS image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                # Apply rotation if needed
                if self.rotate_left:
                    self.left_image = cv2.rotate(cv_image, self.rotation_angle)
                else:
                    self.left_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing left camera image: {e}')
    
    def right_camera_callback(self, msg):
        """Callback for right camera images"""
        try:
            with self.right_image_lock:
                # Convert ROS image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                # Apply rotation if needed
                if self.rotate_right:
                    self.right_image = cv2.rotate(cv_image, self.rotation_angle)
                else:
                    self.right_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing right camera image: {e}')
    
    def stitch_images(self, left_img, right_img):
        """Stitch two images side by side"""
        if left_img is None or right_img is None:
            return None
            
        # Resize images to have the same height
        height = min(left_img.shape[0], right_img.shape[0])
        left_resized = cv2.resize(left_img, (int(left_img.shape[1] * height / left_img.shape[0]), height))
        right_resized = cv2.resize(right_img, (int(right_img.shape[1] * height / right_img.shape[0]), height))
        
        # Concatenate images horizontally
        stitched_image = np.hstack((left_resized, right_resized))
        return stitched_image
    
    def process_and_publish(self):
        """Main processing function that runs periodically"""
        # Get current images
        left_img = None
        right_img = None
        
        with self.left_image_lock:
            if self.left_image is not None:
                left_img = self.left_image.copy()
        
        with self.right_image_lock:
            if self.right_image is not None:
                right_img = self.right_image.copy()
        
        if left_img is None and right_img is None:
            return
        
        # Stitch images
        stitched_image = self.stitch_images(left_img, right_img)
        
        if stitched_image is None:
            return
        
        # ====== 아래 부분 전체 주석처리 ======
        '''
        # Detect objects
        objects = self.detect_objects(stitched_image)
        
        # Track objects
        tracked_objects = self.track_objects(objects, stitched_image.shape[1], stitched_image.shape[0])
        
        # Draw annotations
        annotated_image = self.draw_tracking_annotations(stitched_image, tracked_objects)
        
        # Publish processed image
        try:
            ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            self.processed_image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f'Error publishing processed image: {e}')
        '''
        # ====== stitch된 이미지만 publish ======
        try:
            ros_image = self.bridge.cv2_to_imgmsg(stitched_image, "bgr8")
            self.processed_image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f'Error publishing stitched image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 