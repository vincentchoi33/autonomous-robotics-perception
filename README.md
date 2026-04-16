# ROS 2 Stereo Segmentation

> Real-time perception pipeline for autonomous driving: dual Intel RealSense camera stitching + semantic segmentation, built on ROS 2.

<p align="center">

![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue?logo=ros)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

</p>

<p align="center">
  <img src="assets/comparison.gif" alt="Pipeline Demo" width="720">
  <br>
  <em>Left: Homography-stitched panorama — Right: Mask2Former semantic segmentation</em>
</p>

<p align="center">

| Metric | Value |
|:---:|:---:|
| **Segmentation Model** | Mask2Former (Swin-Tiny, Cityscapes 19-class) |
| **Processed Frames** | 1,500 / 1,712 pairs (87.6%) |
| **GPU Inference** | 25-30 FPS (RTX 3090) |
| **End-to-End Pipeline** | 15-20 FPS |
| **GPU Memory** | ~2 GB VRAM |
| **Rosbag Duration** | 2 min 27 sec |

</p>

## Repository Scope

This repository is structured as a **reproducible experiment repository**:

- it includes the ROS 2 node, Docker setup, launch files, and sample calibration artifacts
- the large rosbag binary is **downloaded separately** via `download_rosbag.sh`
- the committed `rosbag2_2025_06_16-15_16_29/metadata.yaml` is kept only as a lightweight reference to the expected bag layout
- runtime paths and topic names can be overridden via ROS 2 parameters and environment variables such as `HOMOGRAPHY_MATRIX_PATH`, `VISUALIZATION_OUTPUT_DIR`, and `ROS2_WS_ROOT`

## Processing flow

```
RealSense Left  ─┐                                                          ┌─→  /processed_image
                  ├─→  Homography Stitching  ─→  Mask2Former Segmentation  ─┤
RealSense Right ─┘                                                          └─→  /segmented_image
```

1. **Subscribes** to left / right RealSense camera streams via ROS 2
2. **Stitches** both views into a panoramic image using a pre-computed homography matrix
3. **Segments** the panorama with Mask2Former
4. **Publishes** stitched and segmented outputs as ROS 2 topics and saves image outputs to disk

## Performance

### GPU vs CPU

| | Stitching | Segmentation | Total Pipeline | Processing Time | Success Rate |
|---|:---:|:---:|:---:|:---:|:---:|
| **RTX 3090** | 15-20 FPS | **25-30 FPS** | **15-20 FPS** | ~3-5 min | **87.6%** |
| **CPU only** | 15-20 FPS | 2-3 FPS | 2-3 FPS | ~10-15 min | Varies |

### Resource Footprint

| | CPU Usage | RAM | GPU Memory | Model Load |
|---|:---:|:---:|:---:|:---:|
| **GPU** | 20-30% | ~2 GB | ~2 GB VRAM | ~30s |
| **CPU** | 80-90% | ~4 GB | N/A | ~30s |

> **GPU Speedup**: ~10x faster inference. Real-time capable at 15-20 FPS.

## Quick Start

```bash
git clone https://github.com/VincentChoi33/ros2-stereo-segmentation.git
cd ros2-stereo-segmentation

# Download rosbag data (~2.5GB)
chmod +x download_rosbag.sh && ./download_rosbag.sh

# Run (GPU)
./start_gpu.sh

# Or CPU-only
./start_cpu.sh
```

Results are written to `visualization_output/stitched/` and `visualization_output/segmented/`. Use `./create_video.sh` to assemble MP4 outputs.

## Architecture

```
src/ros2_image_processor/
├── image_processor_node.py   # Main ROS 2 node — subscribe, stitch, segment, publish
├── stitching_utils.py        # Homography-based panorama stitching
└── seg.py                    # Mask2Former wrapper (Swin-Tiny, Cityscapes)
```

**ROS 2 Topics:**

| Direction | Topic | Description |
|---|---|---|
| Subscribe | `/realsense/left/color/image_raw_throttle` | Left camera stream |
| Subscribe | `/realsense/right/color/image_raw_throttle` | Right camera stream |
| Publish | `/processed_image` | Stitched panoramic image |
| Publish | `/segmented_image` | Segmented image with overlays |

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Mask2Former over YOLO/tracking** | Pixel-level scene understanding > bounding boxes for navigation |
| **Homography stitching + fallback** | Efficient for fixed camera geometry, graceful degradation to side-by-side |
| **Thread-safe dual buffers** | Concurrent left/right stream processing, 100-frame auto-cleanup |
| **Lazy model loading** | Segmentation model loaded only when enabled, faster startup |
| **Dockerized CPU & GPU** | One-command reproducible setup, zero host dependency |

## Troubleshooting

```bash
docker exec ros2_image_processor bash -c "source /opt/ros/humble/setup.bash && ros2 node list"
docker logs ros2_image_processor
docker exec ros2_image_processor nvidia-smi
```

## License

Apache License 2.0
