"""Joint-model pixels and raw-display coordinates, without automatic segmentation."""
import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def prepare_joint_slice_input(oriented_image, raw_to_oriented_xy, raw_shape_h_w,
                              output_shape_h_w, brush_mask=None, crop_xyxy=None,
                              raw_marks_xy=None):
    """Image is float32 [0,1], before the display contrast curve; brush is oriented.

    No automatic crop, foreground threshold, percentile transform or reflection
    is inferred. A caller-selected crop uses exclusive upper pixel boundaries.
    The returned affine maps raw_display pixel centres, not full-resolution pixels.
    Channel 1 is the one-pixel inner outline used by the synthetic training data,
    NOT the filled brush mask. Black exterior is applied after image resampling.
    """
    height, width = output_shape_h_w
    image = np.asarray(oriented_image, dtype=np.float32)
    x0, y0, x1, y1 = (0, 0, image.shape[1], image.shape[0]) if crop_xyxy is None else crop_xyxy
    sx, sy = width / (x1 - x0), height / (y1 - y0)
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
    crop = np.array([[1., 0, -x0], [0, 1., -y0], [0, 0, 1.]])
    raw_to_model = resize @ crop @ np.asarray(raw_to_oriented_xy)
    interpolation = cv2.INTER_AREA if sx <= 1 and sy <= 1 else cv2.INTER_LINEAR
    model_image = cv2.resize(image[y0:y1, x0:x1], (width, height), interpolation=interpolation)
    outline = np.zeros((height, width), dtype=np.float32)
    if brush_mask is not None:
        keep = cv2.resize(np.asarray(brush_mask, dtype=np.uint8)[y0:y1, x0:x1],
                          (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
        inner = cv2.erode(keep, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
        inner[[0, -1], :] = 0
        inner[:, [0, -1]] = 0
        outline = (keep - inner).astype(np.float32)
        model_image *= keep
    marks = np.empty((0, 2)) if raw_marks_xy is None else np.asarray(raw_marks_xy).reshape(-1, 2)
    homogeneous = np.column_stack((marks, np.ones(len(marks))))
    model_marks = (homogeneous @ raw_to_model.T)[:, :2]
    valid = (np.isfinite(marks).all(-1) & (marks >= 0).all(-1)
             & (marks <= np.array(raw_shape_h_w)[::-1] - 1).all(-1)
             & (model_marks >= 0).all(-1) & (model_marks <= [width - 1, height - 1]).all(-1))
    mark_image = np.zeros((height, width), dtype=np.float32)
    if valid.any():
        yy, xx = np.mgrid[:height, :width]
        for x, y in model_marks[valid]:
            mark_image = np.maximum(mark_image, np.exp(-((xx - x)**2 + (yy - y)**2) / 8).astype(np.float32))
    channels = np.stack((model_image, outline, np.full_like(outline, brush_mask is not None),
                         mark_image, np.full_like(outline, valid.any())))
    return {'channels': channels, 'raw_to_model_xy': raw_to_model,
            'raw_shape_h_w': tuple(raw_shape_h_w), 'model_shape_h_w': (height, width),
            'crop_xyxy': (x0, y0, x1, y1), 'model_marks_xy': model_marks,
            'mark_valid': valid, 'brush_available': brush_mask is not None}


def joint_raw_points_to_ccf(raw_points_xy, raw_to_model_xy, observed_surface_ccf_um, raw_shape_h_w):
    """Sample an already reflected/deformed H,W,3 surface. No second pose/flip.

    Invalid points stay NaN. Interpolation is restricted to the pixel-centre hull;
    this validity says nothing about anatomical confidence or tissue availability.
    """
    points = np.asarray(raw_points_xy, dtype=np.float64).reshape(-1, 2)
    model = (np.column_stack((points, np.ones(len(points)))) @ np.asarray(raw_to_model_xy).T)[:, :2]
    height, width = observed_surface_ccf_um.shape[:2]
    valid = (np.isfinite(points).all(-1) & np.isfinite(model).all(-1)
             & (points >= 0).all(-1) & (points <= np.array(raw_shape_h_w)[::-1] - 1).all(-1)
             & (model >= 0).all(-1) & (model <= [width - 1, height - 1]).all(-1))
    coordinates = np.full((len(points), 3), np.nan)
    coordinates[valid] = np.column_stack([
        map_coordinates(observed_surface_ccf_um[..., axis], model[valid, ::-1].T,
                        order=1, mode='constant', cval=np.nan, prefilter=False) for axis in range(3)])
    valid &= np.isfinite(coordinates).all(-1)
    return coordinates, valid
