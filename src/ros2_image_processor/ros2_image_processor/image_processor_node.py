#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time
import os
from stitching_utils import (
    load_homography_matrix, 
    stitch_image_pair_homography, 
    stitch_images_side_by_side,
    simple_crop
)
from seg import SemanticSegmentation
from runtime_config import build_runtime_config

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')

        runtime_defaults = build_runtime_config()
        self.declare_parameter('left_camera_topic', runtime_defaults["left_camera_topic"])
        self.declare_parameter('right_camera_topic', runtime_defaults["right_camera_topic"])
        self.declare_parameter('processed_image_topic', runtime_defaults["processed_image_topic"])
        self.declare_parameter('segmented_image_topic', runtime_defaults["segmented_image_topic"])
        self.declare_parameter('visualization_output_root', str(runtime_defaults["visualization_output_root"]))
        self.declare_parameter('homography_matrix_path', str(runtime_defaults["homography_matrix_path"]))
        self.declare_parameter('stitching_method', 'homography')
        self.declare_parameter('enable_segmentation', True)
        self.declare_parameter('enable_crop', True)
        self.declare_parameter('rotate_left', True)
        self.declare_parameter('rotate_right', True)
        self.declare_parameter('timestamp_tolerance', 0.1)
        self.declare_parameter('save_every_n', 1)

        config = build_runtime_config(
            overrides={
                'left_camera_topic': self.get_parameter('left_camera_topic').value,
                'right_camera_topic': self.get_parameter('right_camera_topic').value,
                'processed_image_topic': self.get_parameter('processed_image_topic').value,
                'segmented_image_topic': self.get_parameter('segmented_image_topic').value,
                'visualization_output_root': self.get_parameter('visualization_output_root').value,
                'homography_matrix_path': self.get_parameter('homography_matrix_path').value,
            }
        )
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create output directories
        self.output_dir = str(config["stitched_output_dir"])
        self.segmentation_dir = str(config["segmented_output_dir"])
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.segmentation_dir, exist_ok=True)
        
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
        
        
        # NOTE: Topic names are intentionally cross-mapped to match the physical
        # camera mounting configuration provided with the dataset.
        self.left_camera_sub = self.create_subscription(
            Image,
            config["left_camera_topic"],
            self.left_camera_callback,
            1000
        )

        self.right_camera_sub = self.create_subscription(
            Image,
            config["right_camera_topic"],
            self.right_camera_callback,
            1000
        )
        
        # Create publisher for processed image (optional, for real-time viewing)
        self.processed_image_pub = self.create_publisher(
            Image,
            config["processed_image_topic"],
            100  # Increased queue size
        )
        
        # Create publisher for segmented image (optional, for real-time viewing)
        self.segmented_image_pub = self.create_publisher(
            Image,
            config["segmented_image_topic"],
            100
        )
        
        # Create timer for processing and publishing
        self.timer = self.create_timer(0.03, self.process_and_save)  # ~33 Hz, sufficient for 10-15 FPS camera input
        
        # Stitching method selection (change this to test different methods)
        self.stitching_method = self.get_parameter('stitching_method').value  # Options: 'side_by_side', 'homography'
        
        # Load homography matrix
        self.homography_matrix = load_homography_matrix(str(config["homography_matrix_path"]))
        
        # Initialize semantic segmentation
        self.enable_segmentation = self.get_parameter('enable_segmentation').value  # Enable/disable semantic segmentation
        self.segmentation_alpha = 0.5  # Overlay transparency
        self.segmentation = None
        
        if self.enable_segmentation:
            try:
                self.segmentation = SemanticSegmentation()
                self.get_logger().info('Mask2Former semantic segmentation initialized successfully')
            except Exception as e:
                self.get_logger().warn(f'Failed to initialize semantic segmentation: {e}')
                self.enable_segmentation = False
        
        # Cropping settings
        self.enable_crop = self.get_parameter('enable_crop').value  # Enable/disable cropping of black areas
        
        # Image rotation settings
        self.rotate_left = self.get_parameter('rotate_left').value   # Set to True if left camera image needs rotation
        self.rotate_right = self.get_parameter('rotate_right').value  # Set to True if right camera image needs rotation
        self.rotation_angle = cv2.ROTATE_90_COUNTERCLOCKWISE  # Options: ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
        
        # Timestamp matching tolerance (in seconds)
        self.timestamp_tolerance = self.get_parameter('timestamp_tolerance').value  # 100ms tolerance for matching
        
        # Save settings
        self.save_every_n = self.get_parameter('save_every_n').value  # Save every processed image (10x faster)
        self.save_overlay = True  # Disable overlay generation for speed
        self.enable_analysis_log = False  # Disable detailed analysis logging
        
        self.get_logger().info('Image Processor Node started')
        self.get_logger().info(f'Using stitching method: {self.stitching_method}')
        if self.homography_matrix is not None:
            self.get_logger().info('Homography matrix loaded successfully')
        else:
            self.get_logger().warn('Homography matrix not found, will use side-by-side stitching')
        self.get_logger().info(f'Image rotation: Left={self.rotate_left}, Right={self.rotate_right}')
        self.get_logger().info(f'Cropping enabled: {self.enable_crop}')
        self.get_logger().info(f'semantic segmentation enabled: {self.enable_segmentation}')
        self.get_logger().info(f'Timestamp tolerance: {self.timestamp_tolerance}s')
        self.get_logger().info(f'Left topic: {config["left_camera_topic"]}')
        self.get_logger().info(f'Right topic: {config["right_camera_topic"]}')
        self.get_logger().info(f'Stitched images will be saved to: {self.output_dir}')
        self.get_logger().info(f'Segmented images will be saved to: {self.segmentation_dir}')
        
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
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")  # Wall clock for unique filenames
            filename = f"processed_image_{timestamp_str}_{self.saved_count:04d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, image)
            
            self.get_logger().info(f'SAVED IMAGE #{self.saved_count}: {filename} (Size: {image.shape[1]}x{image.shape[0]})')
            self.saved_count += 1
            
        except Exception as e:
            self.get_logger().error(f'Error saving image: {e}')
    
    def save_segmented_image(self, image, segmentation_map, timestamp):
        """Save segmented image and analysis"""
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            
            # Save segmented image
            filename = f"segmented_image_{timestamp_str}_{self.saved_count:04d}.jpg"
            filepath = os.path.join(self.segmentation_dir, filename)
            cv2.imwrite(filepath, image)
            
            # Analyze segmentation results
            if self.segmentation and segmentation_map is not None:
                # Extract dominant classes (top 5)
                dominant_classes = self.segmentation.get_dominant_classes(segmentation_map, top_k=5)
                dominant_str = ", ".join([f"{cls}: {info['percentage']}%" for cls, info in dominant_classes])
                self.get_logger().info(f'TOP 5 CLASSES: {dominant_str}')
                
                # Full analysis (for debugging)
                if self.matched_pairs % 10 == 0:  # Print full analysis every 10 pairs
                    analysis = self.segmentation.analyze_segmentation(segmentation_map)
                    analysis_str = ", ".join([f"{k}: {v['percentage']}%" for k, v in analysis.items()])
                    self.get_logger().debug(f'FULL ANALYSIS: {analysis_str}')
            
            self.get_logger().info(f'SAVED SEGMENTED IMAGE #{self.saved_count}: {filename}')
            
        except Exception as e:
            self.get_logger().error(f'Error saving segmented image: {e}')
    
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
            self.get_logger().info(f'BUFFER STATUS: Left={len(left_timestamps)}, Right={len(right_timestamps)}')
        
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
                    
                    # Apply semantic segmentation if enabled
                    segmentation_map = None
                    segmented_image = None
                    if self.enable_segmentation and self.segmentation:
                        try:
                            segmentation_map, color_mask = self.segmentation.segment_image(stitched_image)
                            if segmentation_map is not None:
                                # Only create overlay if needed for saving
                                if self.save_overlay and self.matched_pairs % self.save_every_n == 0:
                                    segmented_image = self.segmentation.overlay_segmentation(
                                        stitched_image, segmentation_map, self.segmentation_alpha
                                    )
                                self.get_logger().info(f'Segmentation applied successfully')
                            else:
                                self.get_logger().warn('Segmentation map is None')
                        except Exception as e:
                            self.get_logger().warn(f'Failed to apply segmentation: {e}')
                    
                    # Save images (less frequently for speed)
                    if self.matched_pairs % self.save_every_n == 0:
                        # Save stitched image
                        self.save_image(stitched_image, left_ts)
                        
                        # Save segmented image if available
                        if segmented_image is not None:
                            self.save_segmented_image(segmented_image, segmentation_map, left_ts)
                        elif segmentation_map is not None and self.enable_analysis_log:
                            # Just log analysis without saving overlay
                            dominant_classes = self.segmentation.get_dominant_classes(segmentation_map, top_k=3)
                            dominant_str = ", ".join([f"{cls}: {info['percentage']}%" for cls, info in dominant_classes])
                            self.get_logger().info(f'TOP 3 CLASSES: {dominant_str}')
                    
                    # Publish processed images
                    try:
                        # Publish stitched image
                        ros_image = self.bridge.cv2_to_imgmsg(stitched_image, "bgr8")
                        ros_image.header.stamp = self.get_clock().now().to_msg()
                        self.processed_image_pub.publish(ros_image)
                        
                        # Publish segmented image if available
                        if segmented_image is not None:
                            ros_segmented = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
                            ros_segmented.header.stamp = self.get_clock().now().to_msg()
                            self.segmented_image_pub.publish(ros_segmented)
                    except Exception as e:
                        self.get_logger().error(f'Error publishing images: {e}')
                    
                    self.matched_pairs += 1
                    processed_this_cycle += 1
                    
                    # Log progress every 50 pairs for debugging (reduced from 10)
                    if self.matched_pairs % 50 == 0:
                        self.get_logger().info(f'PROGRESS: Processed {self.matched_pairs} image pairs (Left: {self.total_left_received}, Right: {self.total_right_received}, Saved: {self.saved_count})')
                    
                    # Log every 10th pair for debugging (reduced frequency)
                    if self.matched_pairs % 10 == 0:
                        self.get_logger().info(f'MATCHED PAIR #{self.matched_pairs}: Left TS={left_ts:.3f}, Right TS={right_ts:.3f}, Diff={abs(left_ts-right_ts):.3f}s')
                
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
