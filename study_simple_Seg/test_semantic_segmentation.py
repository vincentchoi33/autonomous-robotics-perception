#!/usr/bin/env python3

"""
Test script for Semantic Segmentation functionality
"""

import cv2
import numpy as np
import time
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'ros2_image_processor', 'ros2_image_processor'))

from semantic_segmentation import SemanticSegmentation

def test_semantic_segmentation():
    """Test semantic segmentation with a sample image"""
    
    print("🧪 Testing Semantic Segmentation...")
    
    # Initialize segmentation
    segmentation = SemanticSegmentation(num_classes=5)
    
    # Create a test image (or load from file if available)
    print("📸 Creating test image...")
    test_image = create_test_image()
    
    # Perform segmentation
    print("🔍 Performing segmentation...")
    start_time = time.time()
    
    segmentation_map, color_mask = segmentation.segment_image(test_image)
    
    processing_time = time.time() - start_time
    print(f"⏱️  Processing time: {processing_time:.3f} seconds")
    
    # Get statistics
    stats = segmentation.get_class_statistics(segmentation_map)
    print(f"📊 Segmentation statistics: {stats}")
    
    # Create overlay
    overlay = segmentation.overlay_segmentation(test_image, segmentation_map, alpha=0.6)
    
    # Add legend
    final_image = segmentation.draw_legend(overlay, stats)
    
    # Save results
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, "test_image.jpg"), test_image)
    cv2.imwrite(os.path.join(output_dir, "segmentation_mask.jpg"), color_mask)
    cv2.imwrite(os.path.join(output_dir, "overlay.jpg"), overlay)
    cv2.imwrite(os.path.join(output_dir, "final_result.jpg"), final_image)
    
    print(f"💾 Results saved to {output_dir}/")
    print("✅ Semantic segmentation test completed!")
    
    return True

def create_test_image():
    """Create a synthetic test image with different regions"""
    
    # Create a 640x480 test image
    height, width = 480, 640
    
    # Create different regions
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background (dark gray)
    image[:, :] = [50, 50, 50]
    
    # Road/ground (gray)
    image[height//2:, :] = [100, 100, 100]
    
    # Building/structure (brown)
    image[:height//3, width//4:3*width//4] = [139, 69, 19]
    
    # Vehicle (red)
    cv2.rectangle(image, (width//3, height//2), (2*width//3, 3*height//4), (0, 0, 255), -1)
    
    # Person (blue)
    cv2.circle(image, (width//2, height//3), 30, (255, 0, 0), -1)
    
    # Add some noise and texture
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Apply slight blur for realism
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image

def test_performance():
    """Test performance with different image sizes"""
    
    print("\n🚀 Performance Test...")
    
    segmentation = SemanticSegmentation(num_classes=5)
    
    sizes = [(320, 240), (640, 480), (1280, 720)]
    
    for width, height in sizes:
        print(f"\n📐 Testing size: {width}x{height}")
        
        # Create test image
        test_image = create_test_image()
        test_image = cv2.resize(test_image, (width, height))
        
        # Measure performance
        times = []
        for i in range(5):  # Run 5 times for average
            start_time = time.time()
            segmentation.segment_image(test_image)
            processing_time = time.time() - start_time
            times.append(processing_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"   Average time: {avg_time:.3f}s")
        print(f"   FPS: {fps:.1f}")

if __name__ == "__main__":
    try:
        # Test basic functionality
        test_semantic_segmentation()
        
        # Test performance
        test_performance()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 