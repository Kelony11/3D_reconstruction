import cv2
import glob
import numpy as np

CHECKERBOARD = (10, 7)
STEREO_IMAGE_SIZE = (1536, 1024)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objpoints = []
imgpoints = []
successful_files = []

images = sorted(
    glob.glob("calibration/*.JPEG")
    + glob.glob("calibration/*.jpeg")
    + glob.glob("calibration/*.JPG")
    + glob.glob("calibration/*.jpg")
)

print(f"Found images: {len(images)}")

image_size = None

for filename in images:
    img = cv2.imread(filename)

    if img is None:
        print(f"Could not read: {filename}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    scale = 0.5
    small_gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    found, corners = cv2.findChessboardCornersSB(
        small_gray,
        CHECKERBOARD,
        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
    )

    if found:
        corners = corners / scale
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        successful_files.append(filename)
        print(f"Detected: {filename}")
    else:
        print(f"Failed: {filename}")

if len(objpoints) < 3:
    raise RuntimeError(
        "Too few successful checkerboard detections. "
        "Retake clearer calibration photos and try again."
    )

rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("\nNumber of successful images:")
print(len(objpoints))

print("\nImage size:")
print(image_size)

print("\nCalibration RMS error:")
print(rms)

print("\nCamera intrinsic matrix K:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(dist_coeffs)


errors = []

print("\nReprojection error for each image:")

for i, (obj_pts, img_pts) in enumerate(zip(objpoints, imgpoints)):

    projected_points, _ = cv2.projectPoints(
        obj_pts,
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    # Make both arrays have the same shape: (N, 2)
    img_pts_2d = img_pts.reshape(-1, 2).astype(np.float32)
    projected_points_2d = projected_points.reshape(-1, 2).astype(np.float32)

    error = cv2.norm(
        img_pts_2d,
        projected_points_2d,
        cv2.NORM_L2
    ) / len(projected_points_2d)

    errors.append(error)

    print(
        f"{successful_files[i]} -> "
        f"reprojection error: {error:.4f} pixels"
    )

mean_error = float(np.mean(errors))

print("\nMean reprojection error:")
print(f"{mean_error:.4f} pixels")

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print("\nIntrinsic parameters:")
print(f"fx = {fx}")
print(f"fy = {fy}")
print(f"cx = {cx}")
print(f"cy = {cy}")

sx = STEREO_IMAGE_SIZE[0] / image_size[0]
sy = STEREO_IMAGE_SIZE[1] / image_size[1]

K_scaled = camera_matrix.copy()
K_scaled[0, 0] *= sx
K_scaled[0, 2] *= sx
K_scaled[1, 1] *= sy
K_scaled[1, 2] *= sy

print("\nScaled intrinsic matrix for 1536 x 1024 stereo images:")
print(K_scaled)

np.savez(
    "camera_calibration.npz",
    K=camera_matrix,
    K_scaled=K_scaled,
    dist_coeffs=dist_coeffs,
    rms=rms,
    mean_reprojection_error=mean_error,
    calibration_image_size=np.array(image_size),
    stereo_image_size=np.array(STEREO_IMAGE_SIZE)
)

print("\nSaved calibration results to camera_calibration.npz")
