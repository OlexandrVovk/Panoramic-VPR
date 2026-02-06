"""Equirectangular to perspective projection using OpenCV remap."""

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=64)
def _build_remap_tables(
    equirect_h: int,
    equirect_w: int,
    fov: float,
    yaw: float,
    pitch: float,
    out_w: int,
    out_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build remap tables for equirectangular-to-perspective projection.

    Cached by all geometry parameters so repeated calls with the same
    panorama resolution and view parameters reuse the lookup tables.

    Returns:
        (map_x, map_y) arrays of shape (out_h, out_w) in float32.
    """
    fov_rad = np.radians(fov)
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    # Focal length from FOV
    f = out_w / (2.0 * np.tan(fov_rad / 2.0))

    # Principal point
    cx = out_w / 2.0
    cy = out_h / 2.0

    # Pixel grid for output image
    u = np.arange(out_w, dtype=np.float64)
    v = np.arange(out_h, dtype=np.float64)
    u, v = np.meshgrid(u, v)

    # 3D ray directions in camera frame (z forward)
    x_cam = (u - cx) / f
    y_cam = (v - cy) / f
    z_cam = np.ones_like(x_cam)

    # Rotation: pitch around X-axis
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)
    x1 = x_cam
    y1 = y_cam * cos_p - z_cam * sin_p
    z1 = y_cam * sin_p + z_cam * cos_p

    # Rotation: yaw around Y-axis
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    x2 = x1 * cos_y + z1 * sin_y
    y2 = y1
    z2 = -x1 * sin_y + z1 * cos_y

    # Spherical coordinates
    theta = np.arctan2(x2, z2)  # longitude [-pi, pi]
    phi = np.arctan2(y2, np.sqrt(x2**2 + z2**2))  # latitude [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    map_x = ((theta / np.pi + 1.0) * 0.5 * equirect_w).astype(np.float32)
    map_y = ((0.5 - phi / np.pi) * equirect_h).astype(np.float32)

    # Wrap x-coordinates for seamless horizontal tiling
    map_x = np.mod(map_x, equirect_w).astype(np.float32)
    map_y = np.clip(map_y, 0, equirect_h - 1).astype(np.float32)

    return map_x, map_y


def equirect_to_perspective(
    equirect: np.ndarray,
    fov: float = 90.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """
    Extract a perspective view from an equirectangular panorama.

    Args:
        equirect: Input panorama image (H, W, 3) uint8.
        fov: Field of view in degrees.
        yaw: Horizontal rotation in degrees (0-360).
        pitch: Vertical rotation in degrees (-90 to 90).
        width: Output perspective image width.
        height: Output perspective image height.

    Returns:
        Perspective view image (height, width, 3) uint8.
    """
    eq_h, eq_w = equirect.shape[:2]
    map_x, map_y = _build_remap_tables(eq_h, eq_w, fov, yaw, pitch, width, height)
    return cv2.remap(
        equirect,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def generate_perspective_views(
    equirect: np.ndarray,
    num_views: int = 8,
    fov: float = 90.0,
    pitch: float = 0.0,
    width: int = 640,
    height: int = 480,
) -> list[tuple[np.ndarray, dict]]:
    """
    Generate K evenly-spaced perspective views from a panorama.

    Args:
        equirect: Input panorama image (H, W, 3).
        num_views: Number of perspective views (K).
        fov: Field of view in degrees.
        pitch: Vertical rotation in degrees.
        width: Output perspective image width.
        height: Output perspective image height.

    Returns:
        List of (view_image, metadata_dict) tuples where metadata contains yaw and pitch.
    """
    yaw_angles = [i * (360.0 / num_views) for i in range(num_views)]
    views = []
    for yaw in yaw_angles:
        view = equirect_to_perspective(equirect, fov, yaw, pitch, width, height)
        views.append((view, {"yaw": yaw, "pitch": pitch}))
    return views
