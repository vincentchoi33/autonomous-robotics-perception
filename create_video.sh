#!/bin/bash

# Video Creation Script for ROS 2 Image Processor
# Converts saved images to MP4 videos

echo "Creating videos from processed images..."
echo "=========================================="

# Function to check if processing is complete
check_processing_complete() {
    echo "Checking if image processing is complete..."
    
    # Get initial file counts
    initial_stitched_count=$(docker exec ros2_image_processor bash -c "ls -1 /ros2_ws/visualization_output/stitched/*.jpg 2>/dev/null | wc -l")
    initial_segmented_count=$(docker exec ros2_image_processor bash -c "ls -1 /ros2_ws/visualization_output/segmented/*.jpg 2>/dev/null | wc -l")
    
    echo "Initial counts - Stitched: $initial_stitched_count, Segmented: $initial_segmented_count"
    
    # Wait and check for changes
    for i in {1..30}; do  # Check for 30 seconds
        sleep 2
        
        current_stitched_count=$(docker exec ros2_image_processor bash -c "ls -1 /ros2_ws/visualization_output/stitched/*.jpg 2>/dev/null | wc -l")
        current_segmented_count=$(docker exec ros2_image_processor bash -c "ls -1 /ros2_ws/visualization_output/segmented/*.jpg 2>/dev/null | wc -l")
        
        if [ "$current_stitched_count" -eq "$initial_stitched_count" ] && [ "$current_segmented_count" -eq "$initial_segmented_count" ]; then
            echo "Processing appears to be complete!"
            echo "Final counts - Stitched: $current_stitched_count, Segmented: $current_segmented_count"
            return 0
        else
            echo "Processing in progress... Stitched: $current_stitched_count, Segmented: $current_segmented_count"
            initial_stitched_count=$current_stitched_count
            initial_segmented_count=$current_segmented_count
        fi
    done
    
    echo "Processing may still be in progress, but proceeding with video creation..."
    return 1
}

# Check if visualization_output directory exists
if [ ! -d "visualization_output" ]; then
    echo "Error: visualization_output directory not found!"
    echo "Please run the image processor first to generate images."
    exit 1
fi

# Check if stitched images exist
if [ ! -d "visualization_output/stitched" ] || [ -z "$(ls -A visualization_output/stitched/ 2>/dev/null)" ]; then
    echo "Error: No stitched images found in visualization_output/stitched/"
    echo "Please run the image processor first to generate images."
    exit 1
fi

# Create videos directory with proper permissions
mkdir -p visualization_output/videos
chmod 777 visualization_output/videos
echo "Created videos directory: visualization_output/videos/"

# Function to create video from images
create_video() {
    local input_dir=$1
    local output_name=$2
    local fps=${3:-10}
    
    echo "Creating $output_name video from $input_dir..."
    
    # Count images
    image_count=$(ls -1 $input_dir/*.jpg 2>/dev/null | wc -l)
    if [ $image_count -eq 0 ]; then
        echo "No images found in $input_dir, skipping..."
        return
    fi
    
    echo "Found $image_count images"
    
    # Get first image to determine dimensions
    first_image=$(ls -1 $input_dir/*.jpg | head -1)
    if [ -z "$first_image" ]; then
        echo "No images found in $input_dir"
        return
    fi
    
    # Create video using ffmpeg
    output_path="visualization_output/videos/${output_name}.mp4"
    
    echo "Processing images with ffmpeg..."
    echo "   Input: $input_dir/*.jpg"
    echo "   Output: $output_path"
    echo "   FPS: $fps"
    
    # Use ffmpeg in Docker container to create video
    docker exec ros2_image_processor bash -c "
        cd /ros2_ws && 
        ffmpeg -y \
            -framerate $fps \
            -pattern_type glob \
            -i '$input_dir/*.jpg' \
            -c:v libx264 \
            -pix_fmt yuv420p \
            -crf 23 \
            -preset medium \
            '$output_path' 2>/dev/null
    "
    
    if [ $? -eq 0 ]; then
        # Get video info from container
        video_size=$(docker exec ros2_image_processor bash -c "ls -lh '$output_path' | awk '{print \$5}'")
        echo "Successfully created: $output_name.mp4 ($video_size)"
    else
        echo "Failed to create $output_name.mp4"
        echo "Check if images exist and ffmpeg is working in container"
    fi
}

# Check if Docker container is running
if ! docker ps --format "table {{.Names}}" | grep -q "^ros2_image_processor$"; then
    echo "Error: Docker container 'ros2_image_processor' is not running!"
    echo "Please run the image processor first: ./start_cpu.sh or ./start_gpu.sh"
    exit 1
fi

echo "Using Docker container for video creation..."

# Check if ffmpeg is installed in the container
if ! docker exec ros2_image_processor command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg in Docker container..."
    docker exec ros2_image_processor bash -c "apt update && apt install -y ffmpeg"
    
    if [ $? -ne 0 ]; then
        echo "Failed to install ffmpeg in container"
        exit 1
    fi
fi

echo "ffmpeg found in container, proceeding with video creation..."

# Create stitched video (147 seconds for 1712 images = ~11.65 FPS)
create_video "visualization_output/stitched" "stitched_video" 11.65

# Create segmented video if segmented images exist
if [ -d "visualization_output/segmented" ] && [ ! -z "$(ls -A visualization_output/segmented/ 2>/dev/null)" ]; then
    create_video "visualization_output/segmented" "segmented_video" 11.65
else
    echo "No segmented images found, skipping segmented video creation"
fi

# Create side-by-side comparison video if both exist
if [ -f "visualization_output/videos/stitched_video.mp4" ] && [ -f "visualization_output/videos/segmented_video.mp4" ]; then
    echo "Creating side-by-side comparison video..."
    
    # Get video dimensions from container
    stitched_width=$(docker exec ros2_image_processor bash -c "cd /ros2_ws && ffprobe -v quiet -select_streams v:0 -show_entries stream=width -of csv=p=0 'visualization_output/videos/stitched_video.mp4'")
    stitched_height=$(docker exec ros2_image_processor bash -c "cd /ros2_ws && ffprobe -v quiet -select_streams v:0 -show_entries stream=height -of csv=p=0 'visualization_output/videos/stitched_video.mp4'")
    
    echo "Video dimensions: ${stitched_width}x${stitched_height}"
    
    # Create side-by-side video in container
    docker exec ros2_image_processor bash -c "
        cd /ros2_ws && 
        ffmpeg -y \
            -i 'visualization_output/videos/stitched_video.mp4' \
            -i 'visualization_output/videos/segmented_video.mp4' \
            -filter_complex '[0:v][1:v]hstack=inputs=2' \
            -c:v libx264 \
            -pix_fmt yuv420p \
            -crf 23 \
            'visualization_output/videos/comparison_video.mp4' 2>/dev/null
    "
    
    if [ $? -eq 0 ]; then
        comparison_size=$(docker exec ros2_image_processor bash -c "ls -lh '/ros2_ws/visualization_output/videos/comparison_video.mp4' | awk '{print \$5}'")
        echo "Successfully created: comparison_video.mp4 ($comparison_size)"
    else
        echo "Failed to create comparison video"
    fi
fi

# Show final results
echo ""
echo "Video creation completed!"
echo "============================"
echo "Videos saved in: visualization_output/videos/"
echo ""

if [ -d "visualization_output/videos" ]; then
    echo "Created videos:"
    docker exec ros2_image_processor bash -c "ls -lh /ros2_ws/visualization_output/videos/*.mp4 2>/dev/null" | while read line; do
        echo "   $line"
    done
fi

echo ""
echo "Tips:"
echo "   - Use VLC or any video player to view the videos"
echo "   - Adjust FPS in the script for different playback speeds"
echo "   - Higher FPS = faster playback, Lower FPS = slower playback"
echo "   - Re-run this script anytime after processing new images" 
echo "   - Re-run this script anytime after processing new images" 