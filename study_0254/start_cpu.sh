#!/bin/bash

# ROS 2 Image Processor Test Script
# Starts all necessary components in sequence

echo "Starting ROS 2 Image Processor Test..."
echo "========================================"

# Step 0: Build Docker image (to reflect any code changes)
echo "Step 0: Building Docker image to reflect code changes..."
docker build -t study_0702-ros2_image_processor .

if [ $? -eq 0 ]; then
    echo "Docker image built successfully"
else
    echo "Failed to build Docker image"
    exit 1
fi

# Step 0.5: Clean up any existing container
echo "Step 0.5: Cleaning up any existing container..."
docker stop ros2_image_processor 2>/dev/null || true
docker rm ros2_image_processor 2>/dev/null || true
echo "Cleanup completed"

# Step 1: Start Docker container
echo "Step 1: Starting Docker container..."
docker run --rm -d --name ros2_image_processor \
    -v $PWD/rosbag2_2025_06_16-15_16_29:/ros2_ws/rosbag2_2025_06_16-15_16_29:ro \
    -v $PWD/visualization_output:/ros2_ws/visualization_output:rw \
    -v $PWD/homography_matrix.npy:/ros2_ws/homography_matrix.npy:ro \
    study_0702-ros2_image_processor tail -f /dev/null

if [ $? -eq 0 ]; then
    echo "Container started successfully"
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
    timeout 600 ros2 bag play /ros2_ws/rosbag2_2025_06_16-15_16_29 --rate 2.0
"

# Step 6: Wait a moment for processing to complete
echo "Step 6: Waiting for processing to complete..."
sleep 5

# Step 7: Show final results
echo ""
echo "Final Results:"
echo "=================="

# Check ROS topics
echo "Active ROS topics:"
docker exec ros2_image_processor bash -c "
    source /opt/ros/humble/setup.bash && 
    timeout 5 ros2 topic list
"

# Show final logs
echo ""
echo "Final node logs:"
docker exec ros2_image_processor tail -n 30 /tmp/node.log

# Check output files
echo ""
echo "Output files:"
ls -la visualization_output/stitched/ | head -5
echo "..."
ls -la visualization_output/segmented/ | head -5

echo ""
echo "Test completed!"
echo "Note: If you have a GPU, you can edit the Dockerfile to use CUDA-enabled torch for faster processing." 