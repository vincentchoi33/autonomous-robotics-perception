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
from stitching_utils import (
    load_homography_matrix, 
    stitch_image_pair_homography, 
    stitch_images_side_by_side,
    simple_crop
)
# from seg import Seg

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
        self.stitching_method = 'homography'  # Options: 'side_by_side', 'homography'
        
        # Load homography matrix
        self.homography_matrix = load_homography_matrix('/ros2_ws/homography_matrix.npy')
        
        # Cropping settings
        self.enable_crop = True  # Enable/disable cropping of black areas
        
        # Segmentation settings
        self.enable_segmentation = False  # Enable/disable segmentation (temporarily disabled)
        self.seg = None
        # if self.enable_segmentation:
        #     try:
        #         self.seg = Seg()
        #         self.get_logger().info('Seg initialized successfully')
        #     except Exception as e:
        #         self.get_logger().warn(f'Failed to initialize Seg: {e}')
        #         self.enable_segmentation = False
        
        # Image rotation settings
        self.rotate_left = True   # Set to True if left camera image needs rotation
        self.rotate_right = True  # Set to True if right camera image needs rotation
        self.rotation_angle = cv2.ROTATE_90_COUNTERCLOCKWISE  # Options: ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
        try:
            self.get_logger().info('Image Processor Node started')
            self.get_logger().info(f'Using stitching method: {self.stitching_method}')
            if self.homography_matrix is not None:
                self.get_logger().info('Homography matrix loaded successfully')
            else:
                self.get_logger().warn('Homography matrix not found, will use side-by-side stitching')
            self.get_logger().info(f'Image rotation: Left={self.rotate_left}, Right={self.rotate_right}')
            self.get_logger().info(f'Cropping enabled: {self.enable_crop}')
            self.get_logger().info(f'Segmentation enabled: {self.enable_segmentation}')
            self.get_logger().info('Node initialization completed successfully')
        except Exception as e:
            self.get_logger().error(f'Exception during node initialization: {e}')
            raise
    
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
        """Stitch two images using the selected method"""
        if left_img is None or right_img is None:
            return None
        
        if self.stitching_method == 'homography' and self.homography_matrix is not None:
            # Use homography-based stitching from utils
            return stitch_image_pair_homography(left_img, right_img, self.homography_matrix)
        else:
            # Fallback to side-by-side stitching from utils
            return stitch_images_side_by_side(left_img, right_img)
    
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
        
        # Apply cropping if enabled
        if self.enable_crop and stitched_image is not None:
            try:
                stitched_image = simple_crop(stitched_image)
                self.get_logger().debug('Applied simple crop to stitched image')
            except Exception as e:
                self.get_logger().warn(f'Failed to apply crop: {e}')
        
        # Apply segmentation if enabled
        final_image = stitched_image
        # if self.enable_segmentation and self.seg is not None:
        #     try:
        #         segmentation_map = self.seg.segment(stitched_image)
        #         final_image = self.seg.overlay_segmentation(stitched_image, segmentation_map)
        #         self.get_logger().debug('Applied segmentation to stitched image')
        #     except Exception as e:
        #         self.get_logger().warn(f'Failed to apply segmentation: {e}')
        #         final_image = stitched_image
        
        # ====== 최종 이미지 publish ======
        try:
            ros_image = self.bridge.cv2_to_imgmsg(final_image, "bgr8")
            self.processed_image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f'Error publishing final image: {e}')

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