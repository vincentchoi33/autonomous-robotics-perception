# ROS 2 Image Processor - Technical Assessment

This project implements a ROS 2 node that processes dual camera streams from Intel RealSense cameras, performs image stitching, and implements object tracking for humans and vehicles with motion path visualization.

## 🎯 Features

- **Dual Camera Stream Processing**: Subscribes to two camera streams (`/realsense/left/color/image_raw_throttle` and `/realsense/right/color/image_raw_throttle`)
- **Image Stitching**: Creates a panoramic side-by-side view from both camera streams using homography or side-by-side methods
- **Semantic Segmentation**: Performs pixel-level classification using K-means clustering with color and texture features
- **Class Detection**: Identifies 5 classes: background, road/ground, building/structure, vehicle, person
- **Visualization**: Overlays segmentation results with color-coded masks and statistics legend
- **Real-time Processing**: Processes and publishes results at 10 FPS

## 📋 Prerequisites

- Docker and Docker Compose
- ROS 2 Humble (if running natively)
- Python 3.8+
- OpenCV 4.x
- NumPy

## 🚀 Quick Start with Docker

### 1. Build the Docker Image

```bash
docker-compose build
```

### 2. Run the Application

```bash
docker-compose up
```

This will:
- Build the ROS 2 workspace
- Launch the image processor node
- Play the ROS bag file automatically
- Display the processed images

### 3. View the Results

The processed images with object tracking annotations will be published to the `/processed_image` topic. You can view them using:

```bash
# In another terminal
ros2 run rqt_image_view rqt_image_view
```

## 🔧 Manual Setup (Without Docker)

### 1. Install Dependencies

```bash
# Install ROS 2 Humble (if not already installed)
sudo apt update
sudo apt install ros-humble-desktop

# Install Python dependencies
pip3 install opencv-python numpy opencv-contrib-python

# Install ROS 2 packages
sudo apt install ros-humble-cv-bridge ros-humble-image-transport
```

### 2. Build the Workspace

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Build the package
colcon build --packages-select ros2_image_processor

# Source the workspace
source install/setup.bash
```

### 3. Run the Application

```bash
# Terminal 1: Launch the image processor
ros2 launch ros2_image_processor image_processor.launch.py

# Terminal 2: Play the bag file (if not using launch file)
ros2 bag play rosbag2_2025_06_16-15_16_29
```

## 📁 Project Structure

```
study_0702/
├── src/ros2_image_processor/
│   ├── ros2_image_processor/
│   │   ├── __init__.py
│   │   └── image_processor_node.py
│   ├── launch/
│   ├── config/
│   ├── scripts/
│   ├── resource/
│   ├── package.xml
│   ├── setup.py
│   └── CMakeLists.txt
├── launch/
│   └── image_processor.launch.py
├── rosbag2_2025_06_16-15_16_29/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🎮 Usage

### Topics

**Subscribed Topics:**
- `/camera_left/image_raw` (sensor_msgs/Image): Left camera stream
- `/camera_right/image_raw` (sensor_msgs/Image): Right camera stream

**Published Topics:**
- `/processed_image` (sensor_msgs/Image): Stitched image with object tracking annotations

### Parameters

- `left_camera_topic`: Topic name for left camera (default: `/camera_left/image_raw`)
- `right_camera_topic`: Topic name for right camera (default: `/camera_right/image_raw`)
- `processed_image_topic`: Topic name for processed image (default: `/processed_image`)

### Launch Arguments

- `bag_file`: Path to the ROS bag file (default: `rosbag2_2025_06_16-15_16_29`)

## 🔍 Semantic Segmentation Details

### Segmentation Method
- **K-means Clustering**: Uses unsupervised learning for pixel classification
- **Feature Extraction**: Combines color (BGR) and texture (Sobel gradients) features
- **5 Classes**: background, road/ground, building/structure, vehicle, person
- **Real-time Processing**: Optimized for 10 FPS performance

### Feature Engineering
- **Color Features**: BGR color values for each pixel
- **Texture Features**: Sobel edge detection for gradient magnitude
- **Preprocessing**: Gaussian blur for noise reduction and size normalization

### Visualization
- **Color-coded Masks**: Each class has a distinct color
- **Overlay Display**: 60% transparency overlay on original image
- **Statistics Legend**: Shows percentage distribution of each class
- **Performance Metrics**: Real-time FPS and processing time display

## 🎨 Visualization Features

- **Color-coded Segmentation**: Each class has a distinct color overlay
- **Class Statistics**: Real-time percentage distribution display
- **Performance Metrics**: Processing time and FPS monitoring
- **Legend Display**: Color-coded legend with class names and percentages
- **Real-time Updates**: 10 FPS processing rate with smooth visualization

## 🐛 Troubleshooting

### Common Issues

1. **OpenCV Haar Cascade Not Found**
   - The system will automatically fall back to contour-based detection
   - Check logs for warnings about cascade loading

2. **No Images Displayed**
   - Verify the bag file contains the expected topics
   - Check topic names match the expected format
   - Use `ros2 topic list` to see available topics

3. **Docker Display Issues**
   - Ensure X11 forwarding is enabled
   - Run `xhost +local:docker` before starting the container

### Debug Commands

```bash
# List available topics
ros2 topic list

# Check topic info
ros2 topic info /camera_left/image_raw

# Monitor topic messages
ros2 topic echo /processed_image

# Check node status
ros2 node list
ros2 node info /image_processor_node
```

## 📊 Performance Considerations

- Processing rate: 10 FPS
- Memory usage: ~200MB for tracking history
- CPU usage: Moderate (depends on image resolution and object count)
- GPU acceleration: Not currently implemented (can be added for better performance)

## 🔮 Future Improvements

- GPU acceleration using CUDA
- More sophisticated object detection (YOLO, SSD)
- Multi-object tracking algorithms (SORT, DeepSORT)
- Semantic segmentation implementation
- Depth estimation from stereo cameras
- Configurable detection parameters
- Recording processed videos

## 📝 License

This project is licensed under the Apache License 2.0.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Note**: This implementation focuses on object tracking as specified in the requirements. The code structure is modular and can be easily extended to implement semantic segmentation or depth estimation as alternative processing options. 