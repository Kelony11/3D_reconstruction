import numpy as np
import matplotlib.pyplot as plt

try:
    calibration = np.load("camera_calibration.npz")
    K = calibration["K_scaled"].astype(np.float64)
    print("Loaded K from camera_calibration.npz")
except FileNotFoundError:
    K = np.array([
        [3314.8324, 0.0, 768.0],
        [0.0, 3315.5140, 512.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    print("camera_calibration.npz not found.")
    print("Using the saved assignment K instead.")

print("\nIntrinsic matrix K:")
print(K)

points1_all = np.array([
    [557.7, 440.2],
    [971.4, 325.5],
    [820.6, 723.4],
    [1282.4, 578.0],
    [548.9, 685.7],
    [797.5, 970.5],
    [1252.3, 802.7],
    [964.5, 551.8],
    [666.2, 584.9],
    [1136.9, 442.5],
    [1036.1, 651.8],
    [778.3, 390.2]
], dtype=np.float64)

points2_all = np.array([
    [386.5, 396.3],
    [872.9, 372.5],
    [452.0, 694.9],
    [1017.6, 679.6],
    [377.3, 611.1],
    [441.2, 945.1],
    [980.7, 905.0],
    [851.4, 598.0],
    [405.8, 539.5],
    [959.9, 521.0],
    [727.5, 685.7],
    [648.2, 387.9]
], dtype=np.float64)

points1_8 = points1_all[:8]
points2_8 = points2_all[:8]
points1_3d = points1_all[8:]
points2_3d = points2_all[8:]

def normalize_points(points, K):
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    K_inv = np.linalg.inv(K)
    return (K_inv @ points_h.T).T

x1 = normalize_points(points1_8, K)
x2 = normalize_points(points2_8, K)

print("\nNormalized Image 1 points P1-P8:")
print(x1)

print("\nNormalized Image 2 points P1-P8:")
print(x2)

A = []

for p1, p2 in zip(x1, x2):
    x, y, _ = p1
    xp, yp, _ = p2

    A.append([
        xp * x,
        xp * y,
        xp,
        yp * x,
        yp * y,
        yp,
        x,
        y,
        1.0
    ])

A = np.array(A, dtype=np.float64)

print("\nMatrix A:")
print(A)

print("\nShape of A:")
print(A.shape)

_, _, Vt_A = np.linalg.svd(A)

E_initial = Vt_A[-1].reshape(3, 3)

print("\nInitial Essential Matrix E:")
print(E_initial)

print("\nInitial singular values:")
print(np.linalg.svd(E_initial)[1])

U_E, S_E, Vt_E = np.linalg.svd(E_initial)

s = (S_E[0] + S_E[1]) / 2.0

E = U_E @ np.diag([s, s, 0.0]) @ Vt_E

print("\nFinal Essential Matrix E:")
print(E)

print("\nFinal singular values:")
print(np.linalg.svd(E)[1])

print("\nEpipolar constraint errors:")

for i in range(8):
    error = x2[i].T @ E @ x1[i]
    print(f"P{i + 1}: x2^T E x1 = {error:.8f}")

U, _, Vt = np.linalg.svd(E)

if np.linalg.det(U) < 0:
    U[:, -1] *= -1

if np.linalg.det(Vt) < 0:
    Vt[-1, :] *= -1

W = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])

R1 = U @ W @ Vt
R2 = U @ W.T @ Vt
t = U[:, 2].reshape(3, 1)

if np.linalg.det(R1) < 0:
    R1 *= -1

if np.linalg.det(R2) < 0:
    R2 *= -1

pose_candidates = [
    ("R1, +t", R1, t),
    ("R1, -t", R1, -t),
    ("R2, +t", R2, t),
    ("R2, -t", R2, -t)
]

P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

def triangulate_point(point1, point2, P1, P2):
    x1_coord, y1_coord, _ = point1
    x2_coord, y2_coord, _ = point2

    A_tri = np.array([
        x1_coord * P1[2] - P1[0],
        y1_coord * P1[2] - P1[1],
        x2_coord * P2[2] - P2[0],
        y2_coord * P2[2] - P2[1]
    ])

    _, _, Vt_tri = np.linalg.svd(A_tri)
    X_h = Vt_tri[-1]

    if abs(X_h[3]) < 1e-12:
        return None

    X_h = X_h / X_h[3]
    return X_h[:3]

def point_in_front_of_both_cameras(X, R, t):
    z1 = X[2]
    X2 = R @ X.reshape(3, 1) + t
    z2 = X2[2, 0]
    return z1 > 0 and z2 > 0

best_pose = None
best_count = -1

print("\nTesting all four camera poses:")
print("--------------------------------")

for name, R_candidate, t_candidate in pose_candidates:
    P2_candidate = np.hstack((R_candidate, t_candidate))
    positive_count = 0

    for p1, p2 in zip(x1, x2):
        X = triangulate_point(p1, p2, P1, P2_candidate)

        if X is not None and point_in_front_of_both_cameras(
            X,
            R_candidate,
            t_candidate
        ):
            positive_count += 1

    print(
        f"{name}: "
        f"{positive_count} / 8 points "
        f"in front of both cameras"
    )

    if positive_count > best_count:
        best_count = positive_count
        best_pose = (name, R_candidate, t_candidate)

best_name, R_best, t_best = best_pose

print("\nBest camera pose:")
print(best_name)

print("\nSelected Rotation Matrix R:")
print(R_best)

print("\nSelected Translation Vector t:")
print(t_best)

print("\nPositive-depth P1-P8 points:", best_count, "/ 8")

x1_test = normalize_points(points1_3d, K)
x2_test = normalize_points(points2_3d, K)

P2 = np.hstack((R_best, t_best))

points_3d = []

print("\nTriangulating P9-P12:")
print("--------------------------------")

for i, (p1, p2) in enumerate(zip(x1_test, x2_test), start=9):
    X = triangulate_point(p1, p2, P1, P2)
    points_3d.append(X)
    print(f"P{i}: {X}")

points_3d = np.array(points_3d)

print("\nDepth check:")
print("--------------------------------")

for i, X in enumerate(points_3d, start=9):
    z1 = X[2]
    X2 = R_best @ X.reshape(3, 1) + t_best
    z2 = X2[2, 0]

    print(
        f"P{i}: "
        f"Camera 1 Z = {z1:.6f}, "
        f"Camera 2 Z = {z2:.6f}"
    )

    if z1 > 0 and z2 > 0:
        print("     -> in front of BOTH cameras")
    else:
        print("     -> behind at least one camera")

print("\nFinal 3D Coordinates:")
print("--------------------------------")

for i, point in enumerate(points_3d, start=9):
    X, Y, Z = point

    print(
        f"P{i}: "
        f"X = {X:.6f}, "
        f"Y = {Y:.6f}, "
        f"Z = {Z:.6f}"
    )

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    points_3d[:, 0],
    points_3d[:, 1],
    points_3d[:, 2],
    s=70
)

for i, point in enumerate(points_3d, start=9):
    ax.text(
        point[0],
        point[1],
        point[2],
        f"P{i}"
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Reconstruction of P9-P12")

plt.tight_layout()
plt.savefig("3d_point_cloud.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved 3D point-cloud figure to 3d_point_cloud.png")
