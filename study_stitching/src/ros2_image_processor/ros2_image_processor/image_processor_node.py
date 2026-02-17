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
import os
from stitching_utils import (
    load_homography_matrix, 
    stitch_image_pair_homography, 
    stitch_images_side_by_side,
    simple_crop
)

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create output directory
        self.output_dir = '/ros2_ws/visualization_output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Image storage with timestamps
        self.left_images = {}  # timestamp -> image
        self.right_images = {}  # timestamp -> image
        self.left_image_lock = threading.Lock()
        self.right_image_lock = threading.Lock()
        
        # Processing statistics
        self.processed_count = 0
        self.total_left_received = 0
        self.total_right_received = 0
        self.matched_pairs = 0
        self.saved_count = 0
        
        # # Object tracking variables
        # self.tracked_objects = {}  # Dictionary to store tracked objects
        # self.object_id_counter = 0
        # self.tracking_history = {}  # Store motion paths for each object
        # self.max_history_length = 30  # Number of points to keep in motion path

        # Create subscribers with larger queue size for batch processing
        self.left_camera_sub = self.create_subscription(
            Image,
            '/realsense/right/color/image_raw_throttle',
            self.left_camera_callback,
            1000  # Increased queue size for better throughput
        )
        
        self.right_camera_sub = self.create_subscription(
            Image,
            '/realsense/left/color/image_raw_throttle',
            self.right_camera_callback,
            1000  # Increased queue size for better throughput
        )
        
        # Create publisher for processed image (optional, for real-time viewing)
        self.processed_image_pub = self.create_publisher(
            Image,
            '/processed_image',
            100  # Increased queue size
        )
        
        # Create timer for processing and publishing
        self.timer = self.create_timer(0.01, self.process_and_save)  # 100 FPS for faster processing
        
        # Stitching method selection (change this to test different methods)
        self.stitching_method = 'homography'  # Options: 'side_by_side', 'homography'
        
        # Load homography matrix
        self.homography_matrix = load_homography_matrix('/ros2_ws/homography_matrix.npy')
        
        # Cropping settings
        self.enable_crop = True  # Enable/disable cropping of black areas
        
        # Image rotation settings
        self.rotate_left = True   # Set to True if left camera image needs rotation
        self.rotate_right = True  # Set to True if right camera image needs rotation
        self.rotation_angle = cv2.ROTATE_90_COUNTERCLOCKWISE  # Options: ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
        
        # Timestamp matching tolerance (in seconds)
        self.timestamp_tolerance = 0.1  # 100ms tolerance for matching
        
        # Save settings
        self.save_every_n = 1  # Save every N processed images
        
        self.get_logger().info('Image Processor Node started')
        self.get_logger().info(f'Using stitching method: {self.stitching_method}')
        if self.homography_matrix is not None:
            self.get_logger().info('Homography matrix loaded successfully')
        else:
            self.get_logger().warn('Homography matrix not found, will use side-by-side stitching')
        self.get_logger().info(f'Image rotation: Left={self.rotate_left}, Right={self.rotate_right}')
        self.get_logger().info(f'Cropping enabled: {self.enable_crop}')
        self.get_logger().info(f'Timestamp tolerance: {self.timestamp_tolerance}s')
        self.get_logger().info(f'Images will be saved to: {self.output_dir}')
        
    def left_camera_callback(self, msg):
        """Callback for left camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Apply rotation if needed
            if self.rotate_left:
                cv_image = cv2.rotate(cv_image, self.rotation_angle)
            
            # Store with timestamp
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            with self.left_image_lock:
                self.left_images[timestamp] = cv_image
                self.total_left_received += 1
                
                # Keep only recent images to avoid memory issues
                if len(self.left_images) > 100:
                    oldest_timestamp = min(self.left_images.keys())
                    del self.left_images[oldest_timestamp]
                    
        except Exception as e:
            self.get_logger().error(f'Error processing left camera image: {e}')
    
    def right_camera_callback(self, msg):
        """Callback for right camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Apply rotation if needed
            if self.rotate_right:
                cv_image = cv2.rotate(cv_image, self.rotation_angle)
            
            # Store with timestamp
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            with self.right_image_lock:
                self.right_images[timestamp] = cv_image
                self.total_right_received += 1
                
                # Keep only recent images to avoid memory issues
                if len(self.right_images) > 100:
                    oldest_timestamp = min(self.right_images.keys())
                    del self.right_images[oldest_timestamp]
                    
        except Exception as e:
            self.get_logger().error(f'Error processing right camera image: {e}')
    
    def find_matching_timestamp(self, target_timestamp, timestamps, tolerance=None):
        """Find the closest timestamp within tolerance"""
        if tolerance is None:
            tolerance = self.timestamp_tolerance
            
        closest_timestamp = None
        min_diff = float('inf')
        
        for ts in timestamps:
            diff = abs(ts - target_timestamp)
            if diff <= tolerance and diff < min_diff:
                min_diff = diff
                closest_timestamp = ts
                
        return closest_timestamp
    
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
    
    def save_image(self, image, timestamp):
        """Save image directly to file"""
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"processed_image_{timestamp_str}_{self.saved_count:04d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, image)
            
            self.get_logger().info(f'💾 SAVED IMAGE #{self.saved_count}: {filename} (Size: {image.shape[1]}x{image.shape[0]})')
            self.saved_count += 1
            
        except Exception as e:
            self.get_logger().error(f'❌ Error saving image: {e}')
    
    def process_and_save(self):
        """Main processing function that runs periodically"""
        # Get current images with timestamps
        left_timestamps = []
        right_timestamps = []
        
        with self.left_image_lock:
            left_timestamps = list(self.left_images.keys())
        
        with self.right_image_lock:
            right_timestamps = list(self.right_images.keys())
        
        if not left_timestamps or not right_timestamps:
            return
        
        # Debug: Log buffer status
        if self.processed_count % 100 == 0:
            self.get_logger().info(f'📦 BUFFER STATUS: Left={len(left_timestamps)}, Right={len(right_timestamps)}')
        
        self.processed_count += 1
        
        # Find matching pairs
        processed_this_cycle = 0
        max_pairs_per_cycle = 10  # Process up to 10 pairs per cycle to avoid blocking
        
        for left_ts in left_timestamps[:max_pairs_per_cycle]:
            # Find matching right image
            right_ts = self.find_matching_timestamp(left_ts, right_timestamps)
            
            if right_ts is not None:
                # Get images with safety check
                left_img = None
                right_img = None
                
                with self.left_image_lock:
                    if left_ts in self.left_images:
                        left_img = self.left_images[left_ts]
                
                with self.right_image_lock:
                    if right_ts in self.right_images:
                        right_img = self.right_images[right_ts]
                
                # Skip if either image is missing
                if left_img is None or right_img is None:
                    continue
                
                # Stitch images
                stitched_image = self.stitch_images(left_img, right_img)
                
                if stitched_image is not None:
                    # Apply cropping if enabled
                    if self.enable_crop:
                        try:
                            stitched_image = simple_crop(stitched_image)
                        except Exception as e:
                            self.get_logger().warn(f'Failed to apply crop: {e}')
                    
                    # Save image directly
                    if self.matched_pairs % self.save_every_n == 0:
                        self.save_image(stitched_image, left_ts)
                    
                    # Publish processed image (optional)
                    try:
                        ros_image = self.bridge.cv2_to_imgmsg(stitched_image, "bgr8")
                        ros_image.header.stamp = self.get_clock().now().to_msg()
                        self.processed_image_pub.publish(ros_image)
                    except Exception as e:
                        self.get_logger().error(f'Error publishing stitched image: {e}')
                    
                    self.matched_pairs += 1
                    processed_this_cycle += 1
                    
                    # Log progress every 10 pairs for debugging
                    if self.matched_pairs % 10 == 0:
                        self.get_logger().info(f'📊 PROGRESS: Processed {self.matched_pairs} image pairs (Left: {self.total_left_received}, Right: {self.total_right_received}, Saved: {self.saved_count})')
                    
                    # Log every pair for debugging
                    self.get_logger().info(f'🔗 MATCHED PAIR #{self.matched_pairs}: Left TS={left_ts:.3f}, Right TS={right_ts:.3f}, Diff={abs(left_ts-right_ts):.3f}s')
                
                # Remove processed images to free memory (IMMEDIATELY)
                with self.left_image_lock:
                    if left_ts in self.left_images:
                        del self.left_images[left_ts]
                        self.get_logger().debug(f'Removed left image: {left_ts}')
                
                with self.right_image_lock:
                    if right_ts in self.right_images:
                        del self.right_images[right_ts]
                        self.get_logger().debug(f'Removed right image: {right_ts}')

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