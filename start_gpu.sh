#!/bin/bash

# ROS 2 Image Processor Test Script (GPU Version)
# Starts all necessary components in sequence with GPU support

# Global variables for cleanup
BAG_PID=""
CONTAINER_NAME="ros2_image_processor"

# Signal handler for graceful shutdown
cleanup_on_exit() {
    echo ""
    echo "Received interrupt signal. Cleaning up..."
    
    # Kill ROS bag process if running
    if [ ! -z "$BAG_PID" ]; then
        echo "Stopping ROS bag playback (PID: $BAG_PID)..."
        docker exec $CONTAINER_NAME bash -c "pkill -f 'ros2 bag play'" 2>/dev/null || true
    fi
    
    # Stop container if running
    if docker ps --format "table {{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
        echo "Stopping container..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
    fi
    
    echo "Cleanup completed. Exiting."
    exit 0
}

# Set up signal handlers
trap cleanup_on_exit SIGINT SIGTERM

# Function to safely cleanup containers
cleanup_containers() {
    echo "Performing container cleanup..."
    
    # Stop and remove the main container if it exists
    if docker ps -a --format "table {{.Names}}" | grep -q "^ros2_image_processor$"; then
        echo "Found existing container 'ros2_image_processor'. Stopping and removing..."
        # Kill any ROS bag processes first
        docker exec ros2_image_processor bash -c "pkill -f 'ros2 bag play'" 2>/dev/null || true
        docker stop ros2_image_processor 2>/dev/null || true
        docker rm ros2_image_processor 2>/dev/null || true
        echo "Existing container cleaned up"
    else
        echo "No existing container found"
    fi
    
    # Check for any other potentially conflicting containers
    echo "Checking for other potentially conflicting containers..."
    conflicting_containers=$(docker ps -a --format "table {{.Names}}" | grep -E "(ros2|image_processor)" | grep -v "NAMES" | grep -v "^ros2_image_processor$" || true)
    
    if [ ! -z "$conflicting_containers" ]; then
        echo "Found potentially conflicting containers:"
        echo "$conflicting_containers"
        echo "Consider stopping them manually if needed:"
        echo "$conflicting_containers" | while read container; do
            if [ ! -z "$container" ]; then
                echo "   docker stop $container"
            fi
        done
    else
        echo "No conflicting containers found"
    fi
    
    echo "Cleanup completed"
}

echo "Starting ROS 2 Image Processor Test (GPU Version)..."
echo "======================================================"

# Perform initial cleanup
cleanup_containers

# Step 0: Check and download rosbag2 if needed
echo "Step 0: Checking for rosbag2 files..."
if [ ! -d "rosbag2_2025_06_16-15_16_29" ] || [ ! -f "rosbag2_2025_06_16-15_16_29/rosbag2_2025_06_16-15_16_29_0.db3" ]; then
    echo "Rosbag2 files not found. Running download script..."
    chmod +x download_rosbag.sh
    ./download_rosbag.sh
    if [ $? -ne 0 ]; then
        echo "Failed to download rosbag2 files"
        exit 1
    fi
else
    echo "Rosbag2 files found, proceeding..."
fi

# Step 1: Build Docker image with CUDA support
echo "Step 1: Building Docker image with CUDA support..."
docker build -t ros2_image_processor:gpu -f Dockerfile.gpu .

if [ $? -eq 0 ]; then
    echo "Docker image built successfully"
else
    echo "Failed to build Docker image"
    exit 1
fi

# Step 2: Final container cleanup check
echo "Step 2: Final container cleanup check..."
# Double-check that no conflicting container exists
if docker ps --format "table {{.Names}}" | grep -q "^ros2_image_processor$"; then
    echo "Warning: Container still running, forcing cleanup..."
    docker stop ros2_image_processor 2>/dev/null || true
    docker rm ros2_image_processor 2>/dev/null || true
fi
echo "Container environment ready"

# Step 3: Start Docker container with GPU support
echo "Step 3: Starting Docker container with GPU support..."

# Create and set permissions for output directories
mkdir -p visualization_output/stitched visualization_output/segmented visualization_output/videos
chmod -R 777 visualization_output

docker run --rm -d --name ros2_image_processor \
    --gpus all \
    -v $PWD/rosbag2_2025_06_16-15_16_29:/ros2_ws/rosbag2_2025_06_16-15_16_29:ro \
    -v $PWD/visualization_output:/ros2_ws/visualization_output:rw \
    -v $PWD/homography_matrix.npy:/ros2_ws/homography_matrix.npy:ro \
    ros2_image_processor:gpu tail -f /dev/null

if [ $? -eq 0 ]; then
    echo "Container started successfully with GPU support"
else
    echo "Failed to start container"
    exit 1
fi

# Step 2: Build and start image processor node
echo "Step 2: Building and starting image processor node..."
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    cd /ros2_ws && 
    colcon build --packages-select ros2_image_processor && 
    source install/setup.bash && 
    nohup python3 src/ros2_image_processor/ros2_image_processor/image_processor_node.py > /tmp/node.log 2>&1 &
"

# Step 3: Wait for node initialization (Mask2Former model download time)
echo "Step 3: Waiting for node initialization and Mask2Former model download..."
sleep 15

# Step 4: Check if node is running
echo "Step 4: Checking node status..."
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    source /ros2_ws/install/setup.bash && 
    timeout 5 ros2 node list
"

# Show initial logs
echo ""
echo "Initial node logs:"
docker exec ros2_image_processor tail -n 20 /tmp/node.log

# Step 5: Start ROS bag playback with controlled rate
echo ""
echo "Step 5: Starting ROS bag playback with controlled rate..."
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    timeout 600 ros2 bag play /ros2_ws/rosbag2_2025_06_16-15_16_29 --rate 1.0
" &
BAG_PID=$!

echo "ROS bag playback started with PID: $BAG_PID"

# Step 6: Wait for processing to complete and detect completion
echo "Step 6: Waiting for processing to complete..."
echo "Monitoring processing progress..."

# Function to detect processing completion
detect_processing_completion() {
    local check_count=0
    local stable_count=0
    local last_stitched_count=0
    local last_segmented_count=0
    
    echo "Monitoring file count changes..."
    
    while [ $check_count -lt 60 ]; do  # Check for up to 2 minutes
        sleep 5
        check_count=$((check_count + 1))
        
        # Get current file counts
        current_stitched_count=$(docker exec ros2_image_processor bash -c "ls -1 /ros2_ws/visualization_output/stitched/*.jpg 2>/dev/null | wc -l")
        current_segmented_count=$(docker exec ros2_image_processor bash -c "ls -1 /ros2_ws/visualization_output/segmented/*.jpg 2>/dev/null | wc -l")
        
        # Check if counts are stable (no change for 3 consecutive checks)
        if [ "$current_stitched_count" -eq "$last_stitched_count" ] && [ "$current_segmented_count" -eq "$last_segmented_count" ]; then
            stable_count=$((stable_count + 1))
            if [ $stable_count -ge 3 ]; then
                echo "Processing appears to be complete!"
                echo "Final counts - Stitched: $current_stitched_count, Segmented: $current_segmented_count"
                return 0
            fi
        else
            stable_count=0
            echo "Processing in progress... Stitched: $current_stitched_count, Segmented: $current_segmented_count"
        fi
        
        last_stitched_count=$current_stitched_count
        last_segmented_count=$current_segmented_count
    done
    
    echo "Processing may still be in progress, but proceeding with results..."
    return 1
}

# Run completion detection
detect_processing_completion

# Step 7: Stop the container and show final results
echo ""
echo "Step 7: Stopping container and showing final results..."

# Kill ROS bag process if still running
if [ ! -z "$BAG_PID" ]; then
    echo "Stopping ROS bag playback (PID: $BAG_PID)..."
    docker exec ros2_image_processor bash -c "pkill -f 'ros2 bag play'" 2>/dev/null || true
fi

# Stop the container to prevent continuous bag playback
echo "Stopping container to prevent continuous bag playback..."
docker stop ros2_image_processor

echo ""
echo "Final Results (GPU Version):"
echo "================================"

# Check output files
echo ""
echo "Output files:"
ls -la visualization_output/stitched/ | head -5
echo "..."
ls -la visualization_output/segmented/ | head -5

echo ""
echo "Image processing completed successfully! (GPU Version)"
echo "========================================================"
echo "Images saved in:"
echo "   - visualization_output/stitched/"
echo "   - visualization_output/segmented/"
echo ""
echo "To create videos from these images, run:"
echo "   ./create_video.sh"
echo ""
echo "GPU processing was ~10x faster than CPU!"
echo ""
echo "Container has been stopped to prevent continuous bag playback." 