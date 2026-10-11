# =====================================================
# vision_utils.py
# =====================================================
import cv2
import numpy as np
import torch
import torch.nn.functional as F

MIN_MASK_AREA = 2000

# =====================================================
# ابعاد واقعی پنجره (سانتی‌متر) → 85cm × 100cm
# =====================================================
WINDOW_WIDTH_CM  = 85.0
WINDOW_HEIGHT_CM = 100.0

_HALF_W = WINDOW_WIDTH_CM  / 2.0   # 42.5
_HALF_H = WINDOW_HEIGHT_CM / 2.0   # 50.0


# =====================================================
# ترتیب گوشه‌ها (TL, TR, BR, BL)
# =====================================================
def order_corners_clockwise(pts):
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


# =====================================================
# YAW با solvePnP — ابعاد پنجره 85×100 cm
# =====================================================
def calculate_yaw_error(raw_mask_tensor, frame_shape,
                        smoothed_yaw_ref=None,
                        min_area=MIN_MASK_AREA,
                        device="cpu",
                        window_w_cm=WINDOW_WIDTH_CM,
                        window_h_cm=WINDOW_HEIGHT_CM):
    """
    Calculate yaw error using solvePnP with WINDOW dimensions (85x100 cm).
    Returns: (yaw_error_degrees, corner_points)
    """
    h, w = frame_shape[:2]

    # Resize mask to frame size
    mask = F.interpolate(
        raw_mask_tensor.unsqueeze(0).unsqueeze(0).float().to(device),
        size=(h, w),
        mode="bilinear",
        align_corners=False
    ).squeeze()

    mask = (mask > 0.75).cpu().numpy().astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return 0.0, None

    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        approx = box.reshape(-1, 1, 2).astype(np.float32)

    image_points = order_corners_clockwise(approx).astype(np.float32)

    # -------------------------------------------------
    # نقاط واقعی پنجره (85cm × 100cm) در صفحه Z=0
    # ترتیب: TL, TR, BR, BL
    # -------------------------------------------------
    half_w = window_w_cm / 2.0
    half_h = window_h_cm / 2.0

    object_points = np.array([
        [-half_w, -half_h, 0],   # Top-Left
        [ half_w, -half_h, 0],   # Top-Right
        [ half_w,  half_h, 0],   # Bottom-Right
        [-half_w,  half_h, 0],   # Bottom-Left
    ], dtype=np.float32)

    # Tello camera matrix
    camera_matrix = np.array([
        [648.0, 0, 320.0],
        [0, 647.0, 240.0],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )
    if not success:
        return 0.0, image_points.astype(np.int32)

    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]  # local Z

    yaw_error = np.degrees(np.arctan2(normal[0], normal[2]))

    # Dead zone
    if abs(yaw_error) < 1.0:
        yaw_error = 0.0

    # EMA smoothing
    if smoothed_yaw_ref is not None:
        if smoothed_yaw_ref[0] is None:
            smoothed_yaw_ref[0] = yaw_error
        else:
            smoothed_yaw_ref[0] = 0.8 * smoothed_yaw_ref[0] + 0.2 * yaw_error
        yaw_error = smoothed_yaw_ref[0]

    return float(yaw_error), image_points.astype(np.int32)


# =====================================================
# ROLL از مرکز پنجره
# =====================================================
def calculate_roll_error(window_center, frame_shape):
    """
    Roll error = (dist_left - dist_right) / (dist_left + dist_right)
    range: [-1, +1]  → 0 = وسط
    """
    if window_center is None:
        return None, None, None

    h, w = frame_shape[:2]
    cx = window_center[0]

    dist_left = cx
    dist_right = w - cx
    total = dist_left + dist_right

    if total == 0:
        return 0.0, dist_left, dist_right

    roll_error = float(dist_left - dist_right) / total
    return roll_error, int(dist_left), int(dist_right)