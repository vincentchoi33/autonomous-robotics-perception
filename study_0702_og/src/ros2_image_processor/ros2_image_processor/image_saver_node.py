#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create output directory
        self.output_dir = '/ros2_ws/visualization_output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Image counter
        self.image_counter = 0
        self.save_interval = 30  # Save every 30 frames (3 seconds at 10 FPS)
        self.frame_count = 0
        
        # Create subscriber for processed image
        self.processed_image_sub = self.create_subscription(
            Image,
            '/processed_image',
            self.processed_image_callback,
            10
        )
        
        self.get_logger().info('Image Saver Node started')
        self.get_logger().info(f'Images will be saved to: {self.output_dir}')
        
    def processed_image_callback(self, msg):
        """Callback for processed images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            self.frame_count += 1
            
            # Save image every save_interval frames
            if self.frame_count % self.save_interval == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"processed_image_{timestamp}_{self.image_counter:04d}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save the image
                cv2.imwrite(filepath, cv_image)
                
                self.get_logger().info(f'Saved image: {filename} (Size: {cv_image.shape[1]}x{cv_image.shape[0]})')
                self.image_counter += 1
                
                # Keep only last 10 images to avoid disk space issues
                if self.image_counter > 10:
                    self.cleanup_old_images()
                    
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def cleanup_old_images(self):
        """Remove old images to save disk space"""
        try:
            files = os.listdir(self.output_dir)
            image_files = [f for f in files if f.endswith('.jpg')]
            image_files.sort()
            
            # Remove oldest files, keep only last 10
            if len(image_files) > 10:
                for old_file in image_files[:-10]:
                    os.remove(os.path.join(self.output_dir, old_file))
                    self.get_logger().info(f'Removed old image: {old_file}')
        except Exception as e:
            self.get_logger().error(f'Error cleaning up old images: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 