# 3D Reconstruction 📷 📦

3D Reconstruction is a computer vision project that estimates camera motion and reconstructs 3D points from a stereo image pair using camera calibration, the 8-point algorithm, and linear triangulation.

## Features

- Stereo image correspondence matching
- Camera calibration using a checkerboard
- Intrinsic matrix estimation
- Essential matrix estimation with the 8-point algorithm
- Camera rotation and translation recovery
- 3D point triangulation
- 3D point cloud visualization
- Epipolar constraint and reprojection error checks

## Tech Stack

- Python
- OpenCV
- NumPy
- Matplotlib

## RUN

Install dependencies:

```bash
pip install opencv-python numpy matplotlib
```

## **Run camera calibration:**
```bash
python calibrate.py
```

## **Run camera calibration:**
```bash
python main.py
```

## Notes
- *calibrate.py* estimates the camera intrinsic parameters from checkerboard images.
- *main.py* computes the Essential Matrix, recovers camera pose, triangulates matched points, and generates the 3D point cloud.

## Author
`Kelvin Ihezue`