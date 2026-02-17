# Use ROS 2 Humble as base image
FROM osrf/ros:humble-desktop

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    ros-humble-rclpy \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create workspace directory
WORKDIR /ros2_ws

# Copy package files
COPY src/ src/

# Pre-download MMSegmentation models during build
RUN python3 -c "import torch; from mmseg.apis import init_model; from mmseg.utils import register_all_modules; print('MMSegmentation installed successfully')"

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select ros2_image_processor && \
    . /ros2_ws/install/setup.sh

# Source the workspace
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Set the default command
CMD ["/bin/bash"] 