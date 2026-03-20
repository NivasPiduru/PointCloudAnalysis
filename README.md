# PointCloudAnalysis
# Assignment 1: Cylinder Detection using RANSAC
**Course:** RAS598 - Mobile Robotics, ASU Spring 2026

## Overview
A ROS 2 perception pipeline that detects and classifies colored cylinders 
from 3D point cloud data (OAK-D camera) using RANSAC and NumPy.

## Pipeline
```
Raw PointCloud → Box Filter → Voxel Downsample → RANSAC Floor Removal
→ Euclidean Clustering → Cylinder RANSAC → HSV Color Classification
```

## Tasks Implemented
- **Task 0:** Box filter + Voxel grid downsampling
- **Task 1:** RANSAC plane segmentation (floor removal)
- **Task 2:** Euclidean clustering using cKDTree
- **Task 3:** Cylinder RANSAC using surface normals (SVD)
- **Task 4:** RGB → HSV color classification (Red/Green/Blue/Pink)

## Dependencies
- ROS 2 Jazzy
- NumPy
- SciPy (cKDTree only)

## Run
```bash
ros2 run assignment1_cyclinder cylinder_detector
ros2 bag play rgbd_bag_0.mcap --loop --rate 0.5
```

## Results
- Detects cylinders in all 3 rosbags
- Correctly classifies Red, Green, Blue cylinders
- Bonus: Pink cylinder detected in rgbd_bag_2
## Results
![Cylinder Detection](images/Screenshot%20from%202026-03-19%2022-08-37.png)
