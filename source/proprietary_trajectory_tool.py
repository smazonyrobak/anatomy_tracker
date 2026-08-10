from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import cv2
import nrrd
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import tifffile
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
DEFAULT_ATLAS_FOLDER = APP_DIR / "data" / "Allen Brain Atlas 25um"
VOXEL_UM = 25.0
# IBL Allen CCF bregma is ML, AP, DV in um; this NRRD is indexed AP, DV, ML.
DEFAULT_BREGMA_UM_ML_AP_DV = np.array([5739.0, 5400.0, 332.0], dtype=np.float64)
DEFAULT_BREGMA_VOXEL_AP_DV_ML = (
    np.array(
        [
            DEFAULT_BREGMA_UM_ML_AP_DV[1],
            DEFAULT_BREGMA_UM_ML_AP_DV[2],
            DEFAULT_BREGMA_UM_ML_AP_DV[0],
        ]
    )
    / VOXEL_UM
)
STEREOTAXIC_AXIS_SIGN_AP_DV_ML = np.array([-1.0, -1.0, 1.0], dtype=np.float64)
CHANNEL_KEY_COLUMNS = ["probe_name", "probe_channel_number"]
ANATOMY_MAPPING_COLUMNS = [
    "structure_id",
    "structure_name",
    "structure_acronym",
    "ccf_ap_index",
    "ccf_dv_index",
    "ccf_ml_index",
    "atlas_region_id",
    "atlas_region",
    "atlas_acronym",
    "atlas_ap",
    "atlas_dv",
    "atlas_ml",
    "stereotaxic_ap_um",
    "stereotaxic_dv_um",
    "stereotaxic_ml_um",
    "trajectory_distance_um",
    "probe_type",
    "anatomy_source",
    "anatomy_assignment_method",
    "anatomy_mapped_at",
]

pg.setConfigOptions(imageAxisOrder="row-major", background="#0f131a", foreground="#d7e7f5")


def _integer_series(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    rounded = np.rint(numeric.to_numpy(dtype=float)).astype(np.int64)
    if not np.allclose(numeric.to_numpy(dtype=float), rounded):
        raise ValueError(f"{label} contains non-integer channel identifiers")
    return pd.Series(rounded, index=values.index, dtype="int64")


def canonical_channel_keys(table: pd.DataFrame, *, units: bool = False) -> pd.DataFrame:
    table = table.copy()
    if "probe_name" not in table.columns:
        if "probe" not in table.columns:
            raise ValueError("Metadata needs probe_name or probe")
        table["probe_name"] = "imec" + _integer_series(table["probe"], "probe").astype(str)
    table["probe_name"] = table["probe_name"].astype(str).str.strip()
    if "probe" in table.columns:
        expected_probe_name = "imec" + _integer_series(table["probe"], "probe").astype(str)
        if not np.array_equal(expected_probe_name.to_numpy(), table["probe_name"].to_numpy()):
            raise ValueError("probe and probe_name disagree")

    if "probe_channel_number" not in table.columns:
        source = "peak_channel" if units else "ks_channel_id"
        if source not in table.columns:
            raise ValueError(f"Metadata needs probe_channel_number or {source}")
        table["probe_channel_number"] = _integer_series(table[source], source)
    else:
        table["probe_channel_number"] = _integer_series(
            table["probe_channel_number"], "probe_channel_number"
        )

    if units and "peak_channel" in table.columns:
        peak_channel = _integer_series(table["peak_channel"], "peak_channel")
        if not np.array_equal(peak_channel.to_numpy(), table["probe_channel_number"].to_numpy()):
            raise ValueError("unit peak_channel and probe_channel_number disagree")
    if not units and "ks_channel_id" in table.columns:
        ks_channel_id = _integer_series(table["ks_channel_id"], "ks_channel_id")
        if not np.array_equal(ks_channel_id.to_numpy(), table["probe_channel_number"].to_numpy()):
            raise ValueError("ks_channel_id and probe_channel_number disagree")

    if not units:
        if "probe_horizontal_position" not in table.columns and "x_um" in table.columns:
            table["probe_horizontal_position"] = pd.to_numeric(table["x_um"], errors="raise")
        if "probe_vertical_position" not in table.columns and "y_um" in table.columns:
            table["probe_vertical_position"] = pd.to_numeric(table["y_um"], errors="raise")
        if "structure_acronym" not in table.columns and "atlas_acronym" in table.columns:
            table["structure_acronym"] = table["atlas_acronym"]
        duplicates = table.duplicated(CHANNEL_KEY_COLUMNS, keep=False)
        if duplicates.any():
            keys = table.loc[duplicates, CHANNEL_KEY_COLUMNS].drop_duplicates().head(10).to_dict("records")
            raise ValueError(f"Duplicate probe/channel keys in channels.csv: {keys}")
    return table


def attach_peak_channel_metadata(channels: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    # Channel numbers repeat across probes, so the composite key is mandatory.
    channels = canonical_channel_keys(channels)
    units = canonical_channel_keys(units, units=True)
    if "unit_key" in units.columns and units["unit_key"].duplicated().any():
        raise ValueError("units.csv contains duplicate unit_key values")

    copy_columns = [
        name
        for name in [
            "probe_horizontal_position",
            "probe_vertical_position",
            "probe_shank",
            *ANATOMY_MAPPING_COLUMNS,
        ]
        if name in channels.columns
    ]
    units = units.drop(columns=[name for name in copy_columns if name in units.columns])
    before = len(units)
    units = units.merge(
        channels[CHANNEL_KEY_COLUMNS + copy_columns],
        on=CHANNEL_KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if len(units) != before:
        raise ValueError("Peak-channel join changed the number of units")
    unresolved = units[
        units["probe_horizontal_position"].isna()
        | units["probe_vertical_position"].isna()
    ]
    if len(unresolved):
        keys = unresolved[["unit_key", *CHANNEL_KEY_COLUMNS]].head(10).to_dict("records")
        raise ValueError(f"Units have peak channels absent from channels.csv: {keys}")
    return units


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_anatomy_sidecars(data_folder: Path, channels: pd.DataFrame, units: pd.DataFrame) -> None:
    anatomy_dir = data_folder / "anatomy"
    anatomy_dir.mkdir(exist_ok=True)
    channel_columns = [
        name
        for name in [
            *CHANNEL_KEY_COLUMNS,
            "probe_horizontal_position",
            "probe_vertical_position",
            "probe_shank",
            *ANATOMY_MAPPING_COLUMNS,
        ]
        if name in channels.columns
    ]
    unit_columns = [
        name
        for name in [
            "unit_key",
            "unit_id",
            *CHANNEL_KEY_COLUMNS,
            "peak_channel_index",
            "probe_horizontal_position",
            "probe_vertical_position",
            "probe_shank",
            *ANATOMY_MAPPING_COLUMNS,
        ]
        if name in units.columns
    ]
    write_csv_atomic(channels[channel_columns], anatomy_dir / "channel_brain_regions.csv")
    write_csv_atomic(units[unit_columns], anatomy_dir / "unit_brain_region_assignments.csv")


def as_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if image.ndim == 3:
        if image.shape[-1] in (3, 4):
            image = image[..., :3].astype(np.float32).mean(axis=-1)
        else:
            image = image[0]
    return image


def normalize_u8(image: np.ndarray) -> np.ndarray:
    image = as_gray(image).astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(image[finite], [0.2, 99.8])
    if hi <= lo:
        lo = float(np.nanmin(image))
        hi = float(np.nanmax(image))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip((image - lo) * 255.0 / (hi - lo), 0, 255)
    return image.astype(np.uint8)


def downsample_for_display(image: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    factor = max(1, int(np.ceil(max(h, w) / max_side)))
    if factor == 1:
        return image, 1.0
    return image[::factor, ::factor], float(factor)


def apply_curve(image_u8: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    points = sorted((float(x), float(y)) for x, y in points)
    xs = np.array([0.0] + [x for x, _ in points] + [255.0], dtype=np.float32)
    ys = np.array([0.0] + [y for _, y in points] + [255.0], dtype=np.float32)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    lut = np.interp(np.arange(256, dtype=np.float32), xs, ys)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[image_u8]


def slice_geometry_matrix(
    image_shape: tuple[int, int],
    angle_deg: float,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> tuple[tuple[int, int], np.ndarray]:
    h, w = image_shape[:2]
    if abs(angle_deg) < 0.05:
        out_h, out_w = h, w
        matrix = np.eye(3, dtype=np.float64)
    else:
        center = ((w - 1.0) / 2.0, (h - 1.0) / 2.0)
        rotation = np.eye(3, dtype=np.float64)
        rotation[:2, :] = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
        corners = np.array([[0.0, 0.0, 1.0], [w - 1.0, 0.0, 1.0], [0.0, h - 1.0, 1.0], [w - 1.0, h - 1.0, 1.0]])
        rotated_corners = (rotation @ corners.T).T[:, :2]
        min_xy = rotated_corners.min(axis=0)
        max_xy = rotated_corners.max(axis=0)
        out_w = max(1, int(np.ceil(max_xy[0] - min_xy[0] + 1.0)))
        out_h = max(1, int(np.ceil(max_xy[1] - min_xy[1] + 1.0)))
        translate = np.array([[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]])
        matrix = translate @ rotation

    if flip_horizontal:
        matrix = np.array([[-1.0, 0.0, out_w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ matrix
    if flip_vertical:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, out_h - 1.0], [0.0, 0.0, 1.0]]) @ matrix
    return (out_h, out_w), matrix


def transform_slice_image(
    image_u8: np.ndarray,
    angle_deg: float,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    out_shape, matrix = slice_geometry_matrix(image_u8.shape[:2], angle_deg, flip_horizontal, flip_vertical)
    if out_shape == image_u8.shape[:2] and np.allclose(matrix, np.eye(3)):
        return image_u8.copy(), matrix
    out_h, out_w = out_shape
    transformed = cv2.warpAffine(
        image_u8,
        matrix[:2, :].astype(np.float32),
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return transformed.astype(np.uint8), matrix


def transform_points(points: list[tuple[float, float]], matrix: np.ndarray) -> list[tuple[float, float]]:
    if not points:
        return []
    hom = np.column_stack([np.asarray(points, dtype=np.float64), np.ones(len(points), dtype=np.float64)])
    mapped = (matrix @ hom.T).T
    return [(float(x), float(y)) for x, y in mapped[:, :2]]


def mirror_points_in_shape(
    points: list[tuple[float, float]],
    image_shape: tuple[int, int],
    mirror_horizontal: bool,
    mirror_vertical: bool,
) -> list[tuple[float, float]]:
    if not points:
        return []
    h, w = image_shape[:2]
    mirrored = []
    for x, y in points:
        if mirror_horizontal:
            x = w - 1.0 - x
        if mirror_vertical:
            y = h - 1.0 - y
        mirrored.append((float(x), float(y)))
    return mirrored


def red_rgba(image_u8: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*image_u8.shape, 4), dtype=np.uint8)
    rgba[..., 0] = image_u8
    rgba[..., 3] = np.where(image_u8 > 0, np.maximum(image_u8, 35), 0).astype(np.uint8)
    return rgba


def gray_rgba(image_u8: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*image_u8.shape, 4), dtype=np.uint8)
    rgba[..., 0] = image_u8
    rgba[..., 1] = image_u8
    rgba[..., 2] = image_u8
    rgba[..., 3] = 255
    return rgba


def atlas_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "coronal":
        return volume[index, :, :]
    if plane == "horizontal":
        return volume[:, index, :]
    return volume[:, :, index]


def coronal_oblique_slice(
    volume: np.ndarray,
    index: int,
    tilt_ml_deg: float,
    tilt_dv_deg: float,
    *,
    order: int,
) -> np.ndarray:
    dv_size, ml_size = volume.shape[1:]
    dv, ml = np.mgrid[0:dv_size, 0:ml_size].astype(np.float64)
    ap = (
        float(index)
        + np.tan(np.deg2rad(tilt_ml_deg)) * (ml - (ml_size - 1) / 2.0)
        + np.tan(np.deg2rad(tilt_dv_deg)) * (dv - (dv_size - 1) / 2.0)
    )
    return map_coordinates(
        volume,
        [ap, dv, ml],
        order=order,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def plane_axis(plane: str) -> int:
    return {"coronal": 0, "horizontal": 1, "sagittal": 2}[plane]


def plane_axis_name(plane: str) -> str:
    return {"coronal": "AP", "horizontal": "DV", "sagittal": "ML"}[plane]


def volume_to_stereotaxic_um(coord: np.ndarray, bregma_voxel: np.ndarray) -> np.ndarray:
    return (np.asarray(coord, dtype=np.float64) - bregma_voxel.astype(np.float64)) * VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML


def point_to_volume(
    point: tuple[float, float],
    plane: str,
    index: int,
    volume_shape: tuple[int, int, int] | None = None,
    tilt_ml_deg: float = 0.0,
    tilt_dv_deg: float = 0.0,
) -> np.ndarray:
    x, y = point
    if plane == "coronal":
        ap = float(index)
        if volume_shape is not None:
            ap += np.tan(np.deg2rad(tilt_ml_deg)) * (x - (volume_shape[2] - 1) / 2.0)
            ap += np.tan(np.deg2rad(tilt_dv_deg)) * (y - (volume_shape[1] - 1) / 2.0)
        return np.array([ap, y, x], dtype=np.float64)
    if plane == "horizontal":
        return np.array([y, index, x], dtype=np.float64)
    return np.array([y, x, index], dtype=np.float64)


def section_plane_corners(
    shape: tuple[int, int, int],
    plane: str,
    index: int,
    tilt_ml_deg: float = 0.0,
    tilt_dv_deg: float = 0.0,
) -> np.ndarray:
    ap_max, dv_max, ml_max = (float(size - 1) for size in shape)
    if plane == "coronal":
        corners = [(0.0, 0.0), (0.0, ml_max), (dv_max, ml_max), (dv_max, 0.0)]
        return np.asarray(
            [
                point_to_volume(
                    (ml, dv),
                    plane,
                    index,
                    shape,
                    tilt_ml_deg,
                    tilt_dv_deg,
                )
                for dv, ml in corners
            ],
            dtype=np.float32,
        )
    if plane == "horizontal":
        return np.asarray(
            [[0.0, index, 0.0], [0.0, index, ml_max], [ap_max, index, ml_max], [ap_max, index, 0.0]],
            dtype=np.float32,
        )
    return np.asarray(
        [[0.0, 0.0, index], [0.0, dv_max, index], [ap_max, dv_max, index], [ap_max, 0.0, index]],
        dtype=np.float32,
    )


def volume_to_gl(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return np.column_stack([points[:, 2], points[:, 0], points[:, 1]])


def clamp_volume(coord: np.ndarray, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    ap = int(np.clip(round(float(coord[0])), 0, shape[0] - 1))
    dv = int(np.clip(round(float(coord[1])), 0, shape[1] - 1))
    ml = int(np.clip(round(float(coord[2])), 0, shape[2] - 1))
    return ap, dv, ml


@dataclass
class SliceSession:
    name: str
    path: str = ""
    display_scale: float = 1.0
    raw_display: np.ndarray | None = None
    adjusted: np.ndarray | None = None
    rotated: np.ndarray | None = None
    weight_image: np.ndarray | None = None
    rotation_deg: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    slice_transform: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    curve_points: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (255.0, 255.0)])
    atlas_plane: str = "coronal"
    atlas_index: int = 0
    atlas_tilt_ml_deg: float = 0.0
    atlas_tilt_dv_deg: float = 0.0
    atlas_landmarks: list[tuple[float, float]] = field(default_factory=list)
    slice_landmarks: list[tuple[float, float]] = field(default_factory=list)
    probe_atlas_points: list[tuple[float, float]] = field(default_factory=list)
    probe_slice_points: list[tuple[float, float]] = field(default_factory=list)
    probe_volume_points: list[list[float]] = field(default_factory=list)
    probe_signal_values: list[float] = field(default_factory=list)
    point_history: list[str] = field(default_factory=list)
    transformed_overlay: np.ndarray | None = None
    slice_to_atlas_x: Rbf | None = None
    slice_to_atlas_y: Rbf | None = None
    atlas_to_slice_x: Rbf | None = None
    atlas_to_slice_y: Rbf | None = None


class ImagePanel(QtWidgets.QWidget):
    clicked = QtCore.Signal(float, float)

    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet("font-weight:600;")
        layout.addWidget(self.title)

        self.widget = pg.GraphicsLayoutWidget()
        self.widget.setBackground("#05070a")
        layout.addWidget(self.widget, 1)
        self.view = self.widget.addViewBox(lockAspect=True)
        self.view.invertY(True)
        self.base_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item.setZValue(5)
        self.overlay_item.hide()
        self.landmark_item = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("#ffe66d"), pen=pg.mkPen("#111820", width=1))
        self.probe_item = pg.ScatterPlotItem(size=9, brush=pg.mkBrush("#ff4d8d"), pen=pg.mkPen("#ffffff", width=1))
        self.landmark_item.setZValue(20)
        self.probe_item.setZValue(25)
        self.view.addItem(self.base_item)
        self.view.addItem(self.overlay_item)
        self.view.addItem(self.landmark_item)
        self.view.addItem(self.probe_item)
        self.labels: list[pg.TextItem] = []
        self.image_shape: tuple[int, int] | None = None
        self.widget.scene().sigMouseClicked.connect(self._mouse_clicked)

    def set_base(self, image: np.ndarray | None) -> None:
        if image is None:
            self.base_item.clear()
            self.overlay_item.hide()
            self.image_shape = None
            return
        self.image_shape = image.shape[:2]
        self.base_item.setImage(image, autoLevels=False, levels=(0, 255))
        self.base_item.setRect(QtCore.QRectF(0, 0, image.shape[1], image.shape[0]))
        self.view.autoRange()

    def set_overlay(self, image: np.ndarray | None, opacity: float = 0.55) -> None:
        if image is None:
            self.overlay_item.hide()
            return
        rgba = red_rgba(image) if image.ndim == 2 else image
        self.overlay_item.setImage(rgba, autoLevels=False)
        self.overlay_item.setRect(QtCore.QRectF(0, 0, rgba.shape[1], rgba.shape[0]))
        self.overlay_item.setOpacity(opacity)
        self.overlay_item.show()

    def set_overlay_opacity(self, opacity: float) -> None:
        self.overlay_item.setOpacity(opacity)

    def set_points(self, landmarks: list[tuple[float, float]], probes: list[tuple[float, float]]) -> None:
        self.landmark_item.setData([{"pos": point} for point in landmarks])
        self.probe_item.setData([{"pos": point} for point in probes])
        for label in self.labels:
            self.view.removeItem(label)
        self.labels.clear()
        for i, (x, y) in enumerate(landmarks, start=1):
            label = pg.TextItem(
                str(i),
                color="#fff4a3",
                anchor=(-0.25, 1.1),
            )
            font = QtGui.QFont()
            font.setPointSize(11)
            font.setBold(True)
            label.setFont(font)
            label.setZValue(30)
            label.setPos(x, y)
            self.view.addItem(label)
            self.labels.append(label)

    def _mouse_clicked(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton or self.image_shape is None:
            return
        point = self.view.mapSceneToView(event.scenePos())
        x = float(point.x())
        y = float(point.y())
        h, w = self.image_shape
        if 0 <= x < w and 0 <= y < h:
            self.clicked.emit(x, y)


class CurveCanvas(QtWidgets.QWidget):
    points_changed = QtCore.Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(165)
        self.setMouseTracking(True)
        self.points: list[tuple[float, float]] = [(0.0, 0.0), (255.0, 255.0)]
        self.hist: np.ndarray | None = None
        self.drag_index: int | None = None

    def set_histogram(self, image_u8: np.ndarray | None) -> None:
        if image_u8 is None:
            self.hist = None
        else:
            hist, _ = np.histogram(image_u8.ravel(), bins=256, range=(0, 255))
            hist = hist.astype(np.float32)
            self.hist = hist / hist.max() if hist.max() > 0 else hist
        self.update()

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self.points = [(float(x), float(y)) for x, y in points]
        self.update()

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#0f131a"))
        graph = self._graph_rect()
        painter.setPen(QtGui.QPen(QtGui.QColor("#2d3a4c"), 1))
        painter.drawRect(graph)
        if self.hist is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor("#697789"), 1))
            for i, value in enumerate(self.hist):
                x = graph.left() + i / 255.0 * graph.width()
                y = graph.bottom() - float(value) * graph.height()
                painter.drawLine(QtCore.QPointF(x, graph.bottom()), QtCore.QPointF(x, y))
        sorted_points = sorted(self.points)
        polyline = QtGui.QPolygonF([self._data_to_pos(x, y) for x, y in sorted_points])
        painter.setPen(QtGui.QPen(QtGui.QColor("#49b9ff"), 2))
        painter.drawPolyline(polyline)
        painter.setBrush(QtGui.QColor("#49b9ff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        for x, y in self.points:
            pos = self._data_to_pos(x, y)
            painter.drawEllipse(pos, 5, 5)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        distances = [QtCore.QLineF(pos, self._data_to_pos(x, y)).length() for x, y in self.points]
        if distances and min(distances) <= 12:
            self.drag_index = int(np.argmin(distances))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.drag_index is None:
            return
        x, y = self._pos_to_data(event.position())
        self.points[self.drag_index] = (x, y)
        self.update()
        self.points_changed.emit(sorted(self.points))

    def mouseReleaseEvent(self, _: QtGui.QMouseEvent) -> None:
        self.drag_index = None
        self.points = sorted(self.points)
        self.points_changed.emit(self.points)

    def _graph_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(28, 8, max(1, self.width() - 38), max(1, self.height() - 32))

    def _data_to_pos(self, x: float, y: float) -> QtCore.QPointF:
        graph = self._graph_rect()
        return QtCore.QPointF(
            graph.left() + np.clip(x, 0, 255) / 255.0 * graph.width(),
            graph.bottom() - np.clip(y, 0, 255) / 255.0 * graph.height(),
        )

    def _pos_to_data(self, pos: QtCore.QPointF) -> tuple[float, float]:
        graph = self._graph_rect()
        x = (pos.x() - graph.left()) / graph.width() * 255.0
        y = (graph.bottom() - pos.y()) / graph.height() * 255.0
        return float(np.clip(x, 0, 255)), float(np.clip(y, 0, 255))


class CurveEditor(QtWidgets.QWidget):
    points_changed = QtCore.Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self._updating = False
        self.points: list[tuple[float, float]] = [(0.0, 0.0), (255.0, 255.0)]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QtWidgets.QHBoxLayout()
        self.point_count = QtWidgets.QSpinBox()
        self.point_count.setKeyboardTracking(False)
        self.point_count.setRange(2, 12)
        self.point_count.setValue(2)
        row.addWidget(QtWidgets.QLabel("Curve points"))
        row.addWidget(self.point_count)
        row.addStretch(1)
        layout.addLayout(row)

        self.canvas = CurveCanvas()
        layout.addWidget(self.canvas)

        self.table = QtWidgets.QTableWidget(2, 2)
        self.table.setHorizontalHeaderLabels(["Input", "Output"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(135)
        layout.addWidget(self.table)

        self.point_count.valueChanged.connect(self._set_count)
        self.table.cellChanged.connect(self._table_changed)
        self.canvas.points_changed.connect(self._canvas_changed)
        self.set_points(self.points)

    def set_histogram(self, image_u8: np.ndarray | None) -> None:
        self.canvas.set_histogram(image_u8)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._updating = True
        self.points = sorted((float(x), float(y)) for x, y in points)
        self.point_count.setValue(len(self.points))
        self.table.setRowCount(len(self.points))
        for row, (x, y) in enumerate(self.points):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{x:.1f}"))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{y:.1f}"))
        self._updating = False
        self._refresh_plot()

    def _set_count(self, count: int) -> None:
        if self._updating:
            return
        if count == len(self.points):
            return
        xs = np.linspace(0, 255, count)
        ys = np.interp(xs, [p[0] for p in self.points], [p[1] for p in self.points])
        self.set_points(list(zip(xs, ys)))
        self.points_changed.emit(self.points)

    def _table_changed(self, *_: object) -> None:
        if self._updating:
            return
        points: list[tuple[float, float]] = []
        for row in range(self.table.rowCount()):
            x_item = self.table.item(row, 0)
            y_item = self.table.item(row, 1)
            if x_item is None or y_item is None:
                continue
            x = float(x_item.text().replace(",", "."))
            y = float(y_item.text().replace(",", "."))
            points.append((np.clip(x, 0, 255), np.clip(y, 0, 255)))
        if len(points) >= 2:
            self.points = sorted(points)
            self._refresh_plot()
            self.points_changed.emit(self.points)

    def _canvas_changed(self, points: list[tuple[float, float]]) -> None:
        if self._updating:
            return
        self.set_points(points)
        self.points_changed.emit(self.points)

    def _refresh_plot(self) -> None:
        self.canvas.set_points(self.points)


class TrajectoryTrackerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        default_atlas_folder: str | Path = DEFAULT_ATLAS_FOLDER,
        default_slices_folder: str | Path = "",
        default_run_folder: str | Path = "",
    ) -> None:
        super().__init__()
        self.setWindowTitle("Proprietary neuropixels trajectory tracker")
        self.resize(1780, 980)
        self.atlas_folder = Path(default_atlas_folder)
        self.default_slices_folder = Path(default_slices_folder) if str(default_slices_folder).strip() else Path()
        self.atlas_volume: np.ndarray | None = None
        self.annotation_volume: np.ndarray | None = None
        self.bregma_voxel = DEFAULT_BREGMA_VOXEL_AP_DV_ML.copy()
        self.region_names: dict[int, tuple[str, str]] = {}
        self.current_atlas_image: np.ndarray | None = None
        self.current_annotation_image: np.ndarray | None = None
        self.sessions: list[SliceSession] = []
        self.current_session_index = -1
        self.dynamic_gl_items: list[object] = []
        self.brain_mesh_item: gl.GLMeshItem | None = None

        self._build_ui(default_run_folder)
        if self.atlas_folder.exists():
            self.load_atlas_folder(self.atlas_folder)

    def _build_ui(self, default_run_folder: str | Path) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)

        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QGridLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)

        self.atlas_path = QtWidgets.QLineEdit(str(self.atlas_folder))
        self.load_atlas_btn = QtWidgets.QPushButton("Load atlas")
        self.browse_atlas_btn = QtWidgets.QPushButton("Browse")
        toolbar_layout.addWidget(QtWidgets.QLabel("Atlas"), 0, 0)
        toolbar_layout.addWidget(self.atlas_path, 0, 1)
        toolbar_layout.addWidget(self.browse_atlas_btn, 0, 2)
        toolbar_layout.addWidget(self.load_atlas_btn, 0, 3)

        self.plane_box = QtWidgets.QComboBox()
        self.plane_box.addItems(["coronal", "sagittal", "horizontal"])
        self.section_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, toolbar)
        self.section_slider.hide()
        self.axis_label = QtWidgets.QLabel("AP position")
        self.axis_position_um = QtWidgets.QSpinBox()
        self.axis_position_um.setKeyboardTracking(False)
        self.axis_position_um.setRange(-999999, 999999)
        self.axis_position_um.setSingleStep(int(VOXEL_UM))
        self.axis_position_um.setSuffix(" um")
        toolbar_layout.addWidget(QtWidgets.QLabel("Plane"), 1, 0)
        toolbar_layout.addWidget(self.plane_box, 1, 1)
        toolbar_layout.addWidget(self.axis_label, 1, 2)
        toolbar_layout.addWidget(self.axis_position_um, 1, 3)

        self.add_slice_btn = QtWidgets.QPushButton("Add/load slice")
        self.slice_list = QtWidgets.QComboBox()
        self.rotation = QtWidgets.QDoubleSpinBox()
        self.rotation.setKeyboardTracking(False)
        self.rotation.setRange(-3600.0, 3600.0)
        self.rotation.setDecimals(1)
        self.rotation.setSingleStep(0.1)
        self.rotation.setSuffix(" deg")
        self.flip_horizontal = QtWidgets.QCheckBox("Flip H")
        self.flip_vertical = QtWidgets.QCheckBox("Flip V")
        toolbar_layout.addWidget(self.add_slice_btn, 2, 0)
        toolbar_layout.addWidget(self.slice_list, 2, 1)
        toolbar_layout.addWidget(QtWidgets.QLabel("Slice rotation"), 2, 2)
        toolbar_layout.addWidget(self.rotation, 2, 3)
        toolbar_layout.addWidget(self.flip_horizontal, 2, 4)
        toolbar_layout.addWidget(self.flip_vertical, 2, 5)

        mode_box = QtWidgets.QGroupBox("Point target")
        mode_layout = QtWidgets.QHBoxLayout(mode_box)
        mode_layout.setContentsMargins(6, 4, 6, 4)
        self.landmark_mode = QtWidgets.QPushButton("Transform landmarks")
        self.probe_mode = QtWidgets.QPushButton("Probe trajectory")
        self.landmark_mode.setCheckable(True)
        self.probe_mode.setCheckable(True)
        mode_button_style = "QPushButton:checked { background:#2b6f95; border:2px solid #80d4ff; color:#ffffff; }"
        self.landmark_mode.setStyleSheet(mode_button_style)
        self.probe_mode.setStyleSheet(mode_button_style)
        self.landmark_mode.setChecked(True)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.landmark_mode)
        self.mode_group.addButton(self.probe_mode)
        mode_layout.addWidget(self.landmark_mode)
        mode_layout.addWidget(self.probe_mode)
        self.transform_btn = QtWidgets.QPushButton("Transform slice to atlas coordinates")
        self.undo_point_btn = QtWidgets.QPushButton("Undo point")
        self.clear_points_btn = QtWidgets.QPushButton("Clear active points")
        atlas_opacity_box = QtWidgets.QWidget()
        atlas_opacity_layout = QtWidgets.QHBoxLayout(atlas_opacity_box)
        atlas_opacity_layout.setContentsMargins(0, 0, 0, 0)
        self.atlas_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.atlas_opacity.setRange(0, 100)
        self.atlas_opacity.setValue(65)
        self.atlas_opacity_value = QtWidgets.QLabel("65%")
        atlas_opacity_layout.addWidget(QtWidgets.QLabel("Atlas opacity"))
        atlas_opacity_layout.addWidget(self.atlas_opacity, 1)
        atlas_opacity_layout.addWidget(self.atlas_opacity_value)
        toolbar_layout.addWidget(mode_box, 3, 0)
        toolbar_layout.addWidget(self.transform_btn, 3, 1)
        toolbar_layout.addWidget(self.undo_point_btn, 3, 2)
        toolbar_layout.addWidget(self.clear_points_btn, 3, 3)
        toolbar_layout.addWidget(atlas_opacity_box, 3, 4)

        self.probe_type = QtWidgets.QComboBox()
        self.probe_type.addItems(["Neuropixels 1.0", "Neuropixels 2.0 single-shank", "Neuropixels 2.0 four-shank"])
        self.probe_name = QtWidgets.QComboBox()
        self.run_folder = QtWidgets.QLineEdit(str(default_run_folder))
        self.browse_run_btn = QtWidgets.QPushButton("Browse")
        self.map_btn = QtWidgets.QPushButton("Map channels/units")
        self.undo_mapping_btn = QtWidgets.QPushButton("Undo file mapping")
        toolbar_layout.addWidget(QtWidgets.QLabel("Run folder"), 4, 0)
        toolbar_layout.addWidget(self.run_folder, 4, 1, 1, 3)
        toolbar_layout.addWidget(self.browse_run_btn, 4, 4)
        toolbar_layout.addWidget(QtWidgets.QLabel("Probe type"), 5, 0)
        toolbar_layout.addWidget(self.probe_type, 5, 1)
        toolbar_layout.addWidget(QtWidgets.QLabel("Probe to map"), 5, 2)
        toolbar_layout.addWidget(self.probe_name, 5, 3)
        toolbar_layout.addWidget(self.map_btn, 5, 4)
        toolbar_layout.addWidget(self.undo_mapping_btn, 5, 5)

        self.endpoint_reference = QtWidgets.QComboBox()
        self.endpoint_reference.addItem("Deepest point is chanMap y=0 contact", "y0_contact")
        self.endpoint_reference.addItem("Deepest point is physical probe tip", "physical_tip")
        self.endpoint_reference.setCurrentIndex(-1)
        self.endpoint_reference.setPlaceholderText("Choose reference")
        self.marked_tip_to_y0_um = QtWidgets.QDoubleSpinBox()
        self.marked_tip_to_y0_um.setKeyboardTracking(False)
        self.marked_tip_to_y0_um.setRange(0.0, 2000.0)
        self.marked_tip_to_y0_um.setDecimals(1)
        self.marked_tip_to_y0_um.setSuffix(" um")
        self.marked_tip_to_y0_um.setValue(0.0)
        self.marked_tip_to_y0_um.setEnabled(False)
        tip_note = QtWidgets.QLabel(
            "If the deepest point is the physical tip, enter its distance to the lowest recording contact."
        )
        tip_note.setWordWrap(True)
        tip_note.setStyleSheet("color:#9fb4c8;")
        toolbar_layout.addWidget(QtWidgets.QLabel("Deepest marked point"), 6, 0)
        toolbar_layout.addWidget(self.endpoint_reference, 6, 1, 1, 2)
        toolbar_layout.addWidget(self.marked_tip_to_y0_um, 6, 3)
        toolbar_layout.addWidget(tip_note, 6, 4, 1, 2)

        self.point_counts = QtWidgets.QLabel("Transform atlas 0 / slice 0 | Probe 0")
        self.point_counts.setStyleSheet("color:#9fb4c8;")
        self.brightness_weighting = QtWidgets.QCheckBox("Brightness-weighted trajectory")
        self.status = QtWidgets.QLabel("Idle")
        self.status.setStyleSheet("color:#9fb4c8;")
        toolbar_layout.addWidget(self.point_counts, 7, 0, 1, 2)
        toolbar_layout.addWidget(self.brightness_weighting, 7, 2)
        toolbar_layout.addWidget(self.status, 7, 3, 1, 3)
        toolbar_layout.setColumnStretch(1, 2)
        toolbar_layout.setColumnStretch(2, 3)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        atlas_group = QtWidgets.QGroupBox("Atlas")
        atlas_layout = QtWidgets.QVBoxLayout(atlas_group)
        self.atlas_panel = ImagePanel("Loaded atlas")
        atlas_layout.addWidget(self.atlas_panel, 1)
        self.section_scroll = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        section_row = QtWidgets.QHBoxLayout()
        section_row.addWidget(QtWidgets.QLabel("Section position"))
        section_row.addWidget(self.section_scroll, 1)
        atlas_layout.addLayout(section_row)
        self.atlas_tilt_ml = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.atlas_tilt_ml.setRange(-150, 150)
        self.atlas_tilt_ml.setValue(0)
        self.atlas_tilt_ml.setToolTip("Coronal AP tilt across the left-right (ML) axis")
        self.atlas_tilt_ml_value = QtWidgets.QLabel("0.0°")
        tilt_ml_row = QtWidgets.QHBoxLayout()
        tilt_ml_row.addWidget(QtWidgets.QLabel("Coronal L–R tilt"))
        tilt_ml_row.addWidget(self.atlas_tilt_ml, 1)
        tilt_ml_row.addWidget(self.atlas_tilt_ml_value)
        atlas_layout.addLayout(tilt_ml_row)
        self.atlas_tilt_dv = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.atlas_tilt_dv.setRange(-150, 150)
        self.atlas_tilt_dv.setValue(0)
        self.atlas_tilt_dv.setToolTip("Coronal AP tilt across the dorsal-ventral (DV) axis")
        self.atlas_tilt_dv_value = QtWidgets.QLabel("0.0°")
        tilt_dv_row = QtWidgets.QHBoxLayout()
        tilt_dv_row.addWidget(QtWidgets.QLabel("Coronal D–V tilt"))
        tilt_dv_row.addWidget(self.atlas_tilt_dv, 1)
        tilt_dv_row.addWidget(self.atlas_tilt_dv_value)
        atlas_layout.addLayout(tilt_dv_row)
        splitter.addWidget(atlas_group)

        slice_group = QtWidgets.QGroupBox("Slice")
        slice_layout = QtWidgets.QVBoxLayout(slice_group)
        self.slice_panel = ImagePanel("Brain slice")
        slice_layout.addWidget(self.slice_panel, 1)
        self.curve_editor = CurveEditor()
        slice_layout.addWidget(self.curve_editor)
        splitter.addWidget(slice_group)

        view3d_group = QtWidgets.QGroupBox("3D trajectory")
        view3d_layout = QtWidgets.QVBoxLayout(view3d_group)
        self.view3d = gl.GLViewWidget()
        self.view3d.setBackgroundColor("#05070a")
        self.view3d.setCameraPosition(pos=QtGui.QVector3D(228, 264, 160), distance=760, elevation=22, azimuth=35)
        view3d_layout.addWidget(self.view3d, 1)
        brain_opacity_row = QtWidgets.QHBoxLayout()
        self.brain_opacity_label = QtWidgets.QLabel("Brain opacity")
        self.brain_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brain_opacity.setRange(0, 100)
        self.brain_opacity.setValue(45)
        self.brain_opacity_value = QtWidgets.QLabel("45%")
        self.reset_3d_view_btn = QtWidgets.QPushButton("Reset 3D view")
        brain_opacity_row.addWidget(self.brain_opacity_label)
        brain_opacity_row.addWidget(self.brain_opacity, 1)
        brain_opacity_row.addWidget(self.brain_opacity_value)
        brain_opacity_row.addWidget(self.reset_3d_view_btn)
        view3d_layout.addLayout(brain_opacity_row)
        splitter.addWidget(view3d_group)
        splitter.setSizes([620, 620, 540])

        self.atlas_panel.clicked.connect(self._atlas_clicked)
        self.slice_panel.clicked.connect(self._slice_clicked)
        self.browse_atlas_btn.clicked.connect(self._browse_atlas)
        self.load_atlas_btn.clicked.connect(lambda: self.load_atlas_folder(Path(self.atlas_path.text().strip())))
        self.plane_box.currentTextChanged.connect(self._plane_changed)
        self.section_slider.valueChanged.connect(self._section_changed)
        self.section_scroll.valueChanged.connect(self._section_changed)
        self.atlas_tilt_ml.valueChanged.connect(self._atlas_tilt_changed)
        self.atlas_tilt_dv.valueChanged.connect(self._atlas_tilt_changed)
        self.axis_position_um.valueChanged.connect(self._axis_um_changed)
        self.add_slice_btn.clicked.connect(self._load_slice_dialog)
        self.slice_list.currentIndexChanged.connect(self._switch_slice)
        self.rotation.valueChanged.connect(self._rotation_changed)
        self.flip_horizontal.toggled.connect(self._slice_geometry_changed)
        self.flip_vertical.toggled.connect(self._slice_geometry_changed)
        self.curve_editor.points_changed.connect(self._curve_changed)
        self.transform_btn.clicked.connect(self.transform_current_slice)
        self.undo_point_btn.clicked.connect(self.undo_last_point)
        self.clear_points_btn.clicked.connect(self.clear_current_points)
        self.atlas_opacity.valueChanged.connect(self._atlas_opacity_changed)
        self.brain_opacity.valueChanged.connect(self._brain_opacity_changed)
        self.reset_3d_view_btn.clicked.connect(self._reset_3d_camera)
        self.browse_run_btn.clicked.connect(self._browse_run)
        self.run_folder.editingFinished.connect(self._refresh_probe_names)
        self.endpoint_reference.currentIndexChanged.connect(
            lambda: self.marked_tip_to_y0_um.setEnabled(
                self.endpoint_reference.currentData() == "physical_tip"
            )
        )
        self.map_btn.clicked.connect(self._map_channels_units_clicked)
        self.undo_mapping_btn.clicked.connect(self.undo_file_mapping)
        self.brightness_weighting.toggled.connect(self._trajectory_weighting_changed)
        self.mode_group.buttonClicked.connect(self._point_target_changed)
        self.undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo_last_point)
        self._refresh_probe_names()

    def _map_channels_units_clicked(self) -> None:
        try:
            self.map_channels_units()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Anatomy mapping failed", str(exc))
            self.status.setText(f"Mapping failed: {exc}")

    def _browse_atlas(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select atlas folder", self.atlas_path.text())
        if path:
            self.atlas_path.setText(path)

    def _browse_run(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select preprocessed run folder", self.run_folder.text())
        if path:
            self.run_folder.setText(path)
            self._refresh_probe_names()

    def load_atlas_folder(self, folder: Path) -> None:
        template = folder / "average_template_25.nrrd"
        annotation = folder / "annotation_25.nrrd"
        if not template.exists() or not annotation.exists():
            QtWidgets.QMessageBox.warning(self, "Atlas files missing", f"Missing average_template_25.nrrd or annotation_25.nrrd in:\n{folder}")
            return
        self.atlas_folder = folder
        self.atlas_path.setText(str(folder))
        self.status.setText("Loading atlas")
        QtWidgets.QApplication.processEvents()
        self.atlas_volume = nrrd.read(str(template))[0]
        self.annotation_volume = nrrd.read(str(annotation))[0]
        self.bregma_voxel = self._default_bregma_for_shape(self.atlas_volume.shape)
        self.region_names = self._load_region_names(folder / "query.csv")
        self._setup_3d_static(folder)
        self._set_plane_limits()
        self._refresh_atlas()
        self._refresh_3d()
        self.status.setText(f"Loaded atlas: {folder}")

    def _default_bregma_for_shape(self, shape: tuple[int, int, int]) -> np.ndarray:
        if tuple(shape) == (528, 320, 456):
            return DEFAULT_BREGMA_VOXEL_AP_DV_ML.copy()
        return np.array([shape[0] / 2, 0, shape[2] / 2], dtype=np.float64)

    def _load_region_names(self, query_path: Path) -> dict[int, tuple[str, str]]:
        if not query_path.exists():
            return {}
        table = pd.read_csv(query_path)
        names: dict[int, tuple[str, str]] = {}
        for row in table.itertuples(index=False):
            region_id = int(getattr(row, "id"))
            names[region_id] = (str(getattr(row, "name", region_id)), str(getattr(row, "acronym", region_id)))
        return names

    def _setup_3d_static(self, folder: Path) -> None:
        self.view3d.clear()
        self.brain_mesh_item = None
        self._reset_3d_camera()
        grid = gl.GLGridItem()
        grid.setSize(x=456, y=528, z=1)
        grid.setSpacing(x=50, y=50, z=1)
        grid.translate(228, 264, 0)
        self.view3d.addItem(grid)
        self._add_volume_box()
        mesh_path = folder / "atlas_meshdata.pkl"
        if mesh_path.exists():
            with open(mesh_path, "rb") as handle:
                mesh_data = pickle.load(handle)
            if mesh_data is not None:
                self.brain_mesh_item = gl.GLMeshItem(
                    meshdata=mesh_data,
                    color=self._brain_mesh_color(),
                    smooth=True,
                    shader="balloon",
                )
                self.brain_mesh_item.setGLOptions("additive")
                self.view3d.addItem(self.brain_mesh_item)
        else:
            self.status.setText(f"Whole-brain mesh missing: {mesh_path}")
        self._refresh_3d()

    def _brain_mesh_color(self) -> tuple[float, float, float, float]:
        return (0.55, 0.62, 0.72, self.brain_opacity.value() / 100.0)

    def _brain_opacity_changed(self, value: int) -> None:
        self.brain_opacity_value.setText(f"{value}%")
        if self.brain_mesh_item is not None:
            self.brain_mesh_item.setVisible(value > 0)
            self.brain_mesh_item.setColor(self._brain_mesh_color())

    def _reset_3d_camera(self) -> None:
        center = QtGui.QVector3D(228, 264, 160)
        if self.atlas_volume is not None:
            ap, dv, ml = self.atlas_volume.shape
            center = QtGui.QVector3D(ml / 2, ap / 2, dv / 2)
        self.view3d.setCameraPosition(pos=center, distance=760, elevation=22, azimuth=35)

    def _add_volume_box(self) -> None:
        if self.atlas_volume is None:
            return
        ap, dv, ml = self.atlas_volume.shape
        corners = np.array(
            [
                [0, 0, 0],
                [ap, 0, 0],
                [ap, dv, 0],
                [0, dv, 0],
                [0, 0, ml],
                [ap, 0, ml],
                [ap, dv, ml],
                [0, dv, ml],
            ],
            dtype=np.float32,
        )
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        line_points = []
        for a, b in edges:
            line_points.extend([corners[a], corners[b]])
        item = gl.GLLinePlotItem(pos=volume_to_gl(np.array(line_points)), color=(0.7, 0.85, 1.0, 0.5), width=1, mode="lines")
        self.view3d.addItem(item)

    def _set_plane_limits(self) -> None:
        if self.atlas_volume is None:
            return
        axis = plane_axis(self.plane_box.currentText())
        index = int(np.clip(round(float(self.bregma_voxel[axis])), 0, self.atlas_volume.shape[axis] - 1))
        self.section_slider.blockSignals(True)
        self.section_scroll.blockSignals(True)
        self.section_slider.setRange(0, self.atlas_volume.shape[axis] - 1)
        self.section_scroll.setRange(0, self.atlas_volume.shape[axis] - 1)
        self.section_slider.setValue(index)
        self.section_scroll.setValue(index)
        self.section_slider.blockSignals(False)
        self.section_scroll.blockSignals(False)
        self._update_axis_control(index)
        self._update_tilt_controls()

    def _current_atlas_tilts(self) -> tuple[float, float]:
        if self.plane_box.currentText() != "coronal":
            return 0.0, 0.0
        return self.atlas_tilt_ml.value() / 10.0, self.atlas_tilt_dv.value() / 10.0

    def _update_tilt_controls(self) -> None:
        enabled = self.plane_box.currentText() == "coronal"
        self.atlas_tilt_ml.setEnabled(enabled)
        self.atlas_tilt_dv.setEnabled(enabled)
        self.atlas_tilt_ml_value.setEnabled(enabled)
        self.atlas_tilt_dv_value.setEnabled(enabled)

    def _atlas_tilt_changed(self) -> None:
        tilt_ml, tilt_dv = self._current_atlas_tilts()
        self.atlas_tilt_ml_value.setText(f"{self.atlas_tilt_ml.value() / 10.0:+.1f}°")
        self.atlas_tilt_dv_value.setText(f"{self.atlas_tilt_dv.value() / 10.0:+.1f}°")
        session = self.current_session()
        if session is not None:
            session.atlas_tilt_ml_deg = tilt_ml
            session.atlas_tilt_dv_deg = tilt_dv
            self._recompute_session_volume_points(session)
        self._refresh_atlas()
        self._refresh_3d()

    def _index_to_um(self, index: int) -> int:
        axis = plane_axis(self.plane_box.currentText())
        return int(round((index - float(self.bregma_voxel[axis])) * VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[axis]))

    def _um_to_index(self, value_um: int) -> int:
        if self.atlas_volume is None:
            return 0
        axis = plane_axis(self.plane_box.currentText())
        index = int(round(value_um / (VOXEL_UM * STEREOTAXIC_AXIS_SIGN_AP_DV_ML[axis]) + float(self.bregma_voxel[axis])))
        return int(np.clip(index, 0, self.atlas_volume.shape[axis] - 1))

    def _update_axis_control(self, index: int) -> None:
        if self.atlas_volume is None:
            return
        plane = self.plane_box.currentText()
        axis = plane_axis(plane)
        min_um = self._index_to_um(0)
        max_um = self._index_to_um(self.atlas_volume.shape[axis] - 1)
        if min_um > max_um:
            min_um, max_um = max_um, min_um
        self.axis_label.setText(f"{plane_axis_name(plane)} position")
        self.axis_position_um.blockSignals(True)
        self.axis_position_um.setRange(min_um, max_um)
        self.axis_position_um.setValue(self._index_to_um(index))
        self.axis_position_um.blockSignals(False)

    def _plane_changed(self) -> None:
        self._set_plane_limits()
        session = self.current_session()
        if session is not None:
            session.atlas_plane = self.plane_box.currentText()
            session.atlas_index = self.section_slider.value()
            self._recompute_session_volume_points(session)
        self._refresh_atlas()
        self._refresh_3d()

    def _section_changed(self, value: int) -> None:
        self.section_slider.blockSignals(True)
        self.section_scroll.blockSignals(True)
        self.section_slider.setValue(value)
        self.section_scroll.setValue(value)
        self.section_slider.blockSignals(False)
        self.section_scroll.blockSignals(False)
        self._update_axis_control(value)
        session = self.current_session()
        if session is not None:
            session.atlas_plane = self.plane_box.currentText()
            session.atlas_index = value
            self._recompute_session_volume_points(session)
        self._refresh_atlas()
        self._refresh_3d()

    def _axis_um_changed(self, value_um: int) -> None:
        self._section_changed(self._um_to_index(value_um))

    def _atlas_opacity_changed(self, value: int) -> None:
        self.atlas_opacity_value.setText(f"{value}%")
        self.atlas_panel.set_overlay_opacity(value / 100.0)

    def _refresh_atlas(self) -> None:
        if self.atlas_volume is None or self.annotation_volume is None:
            return
        plane = self.plane_box.currentText()
        index = self.section_slider.value()
        tilt_ml, tilt_dv = self._current_atlas_tilts()
        if plane == "coronal" and (tilt_ml != 0.0 or tilt_dv != 0.0):
            self.current_atlas_image = normalize_u8(
                coronal_oblique_slice(self.atlas_volume, index, tilt_ml, tilt_dv, order=1)
            )
            self.current_annotation_image = coronal_oblique_slice(
                self.annotation_volume, index, tilt_ml, tilt_dv, order=0
            ).astype(self.annotation_volume.dtype, copy=False)
        else:
            self.current_atlas_image = normalize_u8(atlas_slice(self.atlas_volume, plane, index))
            self.current_annotation_image = atlas_slice(self.annotation_volume, plane, index)
        session = self.current_session()
        overlay_available = (
            session is not None
            and session.transformed_overlay is not None
            and session.atlas_plane == plane
            and session.atlas_index == index
            and self.probe_mode.isChecked()
        )
        if overlay_available:
            self.atlas_panel.set_base(session.transformed_overlay)
            self.atlas_panel.set_overlay(gray_rgba(self.current_atlas_image), self.atlas_opacity.value() / 100.0)
        else:
            self.atlas_panel.set_base(self.current_atlas_image)
            self.atlas_panel.set_overlay(None)
        self._refresh_points()

    def _load_slice_dialog(self) -> None:
        start = str(self.default_slices_folder) if self.default_slices_folder else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select brain slice",
            start,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
        )
        if path:
            self.load_slice(Path(path))

    def load_slice(self, path: Path) -> None:
        raw = tifffile.imread(str(path)) if path.suffix.lower() in {".tif", ".tiff"} else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        raw = as_gray(raw)
        display_raw, scale = downsample_for_display(raw)
        display_u8 = normalize_u8(display_raw)
        session = SliceSession(
            name=path.name,
            path=str(path),
            display_scale=scale,
            raw_display=display_u8,
            atlas_plane=self.plane_box.currentText(),
            atlas_index=self.section_slider.value(),
            atlas_tilt_ml_deg=self._current_atlas_tilts()[0],
            atlas_tilt_dv_deg=self._current_atlas_tilts()[1],
        )
        self.sessions.append(session)
        self.slice_list.addItem(session.name)
        self.slice_list.setCurrentIndex(len(self.sessions) - 1)
        self._update_slice_image()
        self.status.setText(f"Loaded slice: {path.name}")

    def _switch_slice(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        self.current_session_index = index
        session = self.sessions[index]
        self.rotation.blockSignals(True)
        self.rotation.setValue(session.rotation_deg)
        self.rotation.blockSignals(False)
        self.flip_horizontal.blockSignals(True)
        self.flip_horizontal.setChecked(session.flip_horizontal)
        self.flip_horizontal.blockSignals(False)
        self.flip_vertical.blockSignals(True)
        self.flip_vertical.setChecked(session.flip_vertical)
        self.flip_vertical.blockSignals(False)
        self.curve_editor.set_points(session.curve_points)
        self.curve_editor.set_histogram(session.raw_display)
        self.plane_box.blockSignals(True)
        self.plane_box.setCurrentText(session.atlas_plane)
        self.plane_box.blockSignals(False)
        self._set_plane_limits()
        self.section_slider.blockSignals(True)
        self.section_scroll.blockSignals(True)
        self.section_slider.setValue(session.atlas_index)
        self.section_scroll.setValue(session.atlas_index)
        self.section_slider.blockSignals(False)
        self.section_scroll.blockSignals(False)
        self.atlas_tilt_ml.blockSignals(True)
        self.atlas_tilt_dv.blockSignals(True)
        self.atlas_tilt_ml.setValue(round(session.atlas_tilt_ml_deg * 10))
        self.atlas_tilt_dv.setValue(round(session.atlas_tilt_dv_deg * 10))
        self.atlas_tilt_ml.blockSignals(False)
        self.atlas_tilt_dv.blockSignals(False)
        self.atlas_tilt_ml_value.setText(f"{session.atlas_tilt_ml_deg:+.1f}°")
        self.atlas_tilt_dv_value.setText(f"{session.atlas_tilt_dv_deg:+.1f}°")
        self._update_tilt_controls()
        self._update_axis_control(session.atlas_index)
        self._refresh_atlas()
        self._update_slice_image()

    def current_session(self) -> SliceSession | None:
        if 0 <= self.current_session_index < len(self.sessions):
            return self.sessions[self.current_session_index]
        return None

    def _curve_changed(self, points: list[tuple[float, float]]) -> None:
        session = self.current_session()
        if session is None:
            return
        session.curve_points = [(float(x), float(y)) for x, y in points]
        self._update_slice_image(clear_transform=True)

    def _rotation_changed(self, value: float) -> None:
        session = self.current_session()
        if session is None:
            return
        self._apply_slice_geometry(float(value), self.flip_horizontal.isChecked(), self.flip_vertical.isChecked())

    def _slice_geometry_changed(self) -> None:
        self._apply_slice_geometry(self.rotation.value(), self.flip_horizontal.isChecked(), self.flip_vertical.isChecked())

    def _apply_slice_geometry(self, rotation_deg: float, flip_horizontal: bool, flip_vertical: bool) -> None:
        session = self.current_session()
        if session is None:
            return
        if (
            abs(session.rotation_deg - float(rotation_deg)) < 0.05
            and session.flip_horizontal == bool(flip_horizontal)
            and session.flip_vertical == bool(flip_vertical)
        ):
            return
        had_transform = session.slice_to_atlas_x is not None and session.atlas_to_slice_x is not None and session.transformed_overlay is not None
        old_flip_horizontal = session.flip_horizontal
        old_flip_vertical = session.flip_vertical
        self._mirror_atlas_landmarks_for_flip_change(
            session,
            old_flip_horizontal != bool(flip_horizontal),
            old_flip_vertical != bool(flip_vertical),
        )
        session.rotation_deg = float(rotation_deg)
        session.flip_horizontal = bool(flip_horizontal)
        session.flip_vertical = bool(flip_vertical)
        self._update_slice_image(clear_transform=True)
        if had_transform and self._rebuild_slice_transform(session):
            self._recompute_probe_points_from_slice_points(session)
            self._refresh_atlas()
            self._refresh_points()
            self._refresh_3d()
            self.status.setText("Slice geometry changed; transform and probe coordinates were rebuilt.")
        else:
            self.status.setText("Slice geometry changed; transform landmarks were moved with the slice.")

    def _update_slice_image(self, *, clear_transform: bool = False) -> None:
        session = self.current_session()
        if session is None or session.raw_display is None:
            return
        session.adjusted = apply_curve(session.raw_display, session.curve_points)
        session.rotated, session.slice_transform = transform_slice_image(
            session.adjusted,
            session.rotation_deg,
            session.flip_horizontal,
            session.flip_vertical,
        )
        session.weight_image, _ = transform_slice_image(
            session.raw_display,
            session.rotation_deg,
            session.flip_horizontal,
            session.flip_vertical,
        )
        if clear_transform:
            session.transformed_overlay = None
            session.slice_to_atlas_x = None
            session.slice_to_atlas_y = None
            session.atlas_to_slice_x = None
            session.atlas_to_slice_y = None
            self._clear_derived_probe_coordinates(session)
        if session.probe_slice_points:
            session.probe_signal_values = [self._probe_point_signal(session, point) for point in session.probe_slice_points]
        self.slice_panel.set_base(session.rotated)
        self.curve_editor.set_histogram(session.raw_display)
        self._refresh_atlas()
        self._refresh_points()

    def _clear_derived_probe_coordinates(self, session: SliceSession) -> None:
        session.probe_atlas_points.clear()
        session.probe_volume_points.clear()

    def _mirror_atlas_landmarks_for_flip_change(
        self,
        session: SliceSession,
        mirror_horizontal: bool,
        mirror_vertical: bool,
    ) -> None:
        if not mirror_horizontal and not mirror_vertical:
            return
        image_shape = self._atlas_image_shape_for_session(session)
        if image_shape is None:
            return
        session.atlas_landmarks = mirror_points_in_shape(session.atlas_landmarks, image_shape, mirror_horizontal, mirror_vertical)

    def _atlas_image_shape_for_session(self, session: SliceSession) -> tuple[int, int] | None:
        if self.atlas_volume is None:
            return self.current_atlas_image.shape[:2] if self.current_atlas_image is not None else None
        index = int(np.clip(session.atlas_index, 0, self.atlas_volume.shape[plane_axis(session.atlas_plane)] - 1))
        return atlas_slice(self.atlas_volume, session.atlas_plane, index).shape[:2]

    def _slice_raw_to_display_points(self, session: SliceSession, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return transform_points(points, session.slice_transform)

    def _slice_display_to_raw_point(self, session: SliceSession, point: tuple[float, float]) -> tuple[float, float]:
        inverse = np.linalg.inv(session.slice_transform)
        x, y, _ = inverse @ np.array([point[0], point[1], 1.0], dtype=np.float64)
        return float(x), float(y)

    def _atlas_clicked(self, x: float, y: float) -> None:
        session = self.current_session()
        if session is None:
            return
        if self.landmark_mode.isChecked():
            self._invalidate_transform_after_landmark_edit(session)
            session.atlas_landmarks.append((x, y))
            session.point_history.append("atlas_landmark")
            self.status.setText(f"Added atlas transform landmark {len(session.atlas_landmarks)}")
        else:
            self._add_probe_point(atlas_point=(x, y), slice_raw_point=None)
        self._refresh_points()

    def _slice_clicked(self, x: float, y: float) -> None:
        session = self.current_session()
        if session is None:
            return
        raw_point = self._slice_display_to_raw_point(session, (x, y))
        if self.landmark_mode.isChecked():
            self._invalidate_transform_after_landmark_edit(session)
            session.slice_landmarks.append(raw_point)
            session.point_history.append("slice_landmark")
            self.status.setText(f"Added slice transform landmark {len(session.slice_landmarks)}")
        else:
            self._add_probe_point(atlas_point=None, slice_raw_point=raw_point)
        self._refresh_points()

    def _invalidate_transform_after_landmark_edit(self, session: SliceSession) -> None:
        if session.slice_to_atlas_x is None and session.atlas_to_slice_x is None and session.transformed_overlay is None:
            return
        session.transformed_overlay = None
        session.slice_to_atlas_x = None
        session.slice_to_atlas_y = None
        session.atlas_to_slice_x = None
        session.atlas_to_slice_y = None
        self._clear_derived_probe_coordinates(session)
        self.atlas_panel.set_overlay(None)
        self._refresh_atlas()
        self._refresh_3d()
        self.status.setText("Transform landmarks changed; run transform again before adding new probe points.")

    def _add_probe_point(
        self,
        *,
        atlas_point: tuple[float, float] | None,
        slice_raw_point: tuple[float, float] | None,
    ) -> None:
        session = self.current_session()
        if session is None:
            return
        if session.slice_to_atlas_x is None or session.atlas_to_slice_x is None:
            QtWidgets.QMessageBox.warning(self, "Transform missing", "Transform the current slice before adding probe points.")
            return
        slice_display_point: tuple[float, float] | None = None
        if atlas_point is None and slice_raw_point is not None:
            slice_display_point = self._slice_raw_to_display_points(session, [slice_raw_point])[0]
            sx, sy = slice_display_point
            atlas_point = (float(session.slice_to_atlas_x(sx, sy)), float(session.slice_to_atlas_y(sx, sy)))
        if slice_raw_point is None and atlas_point is not None:
            ax, ay = atlas_point
            slice_display_point = (float(session.atlas_to_slice_x(ax, ay)), float(session.atlas_to_slice_y(ax, ay)))
            slice_raw_point = self._slice_display_to_raw_point(session, slice_display_point)
        if atlas_point is None or slice_raw_point is None:
            return
        session.probe_atlas_points.append(atlas_point)
        session.probe_slice_points.append(slice_raw_point)
        if self.atlas_volume is not None:
            session.probe_volume_points.append(
                point_to_volume(
                    atlas_point,
                    session.atlas_plane,
                    session.atlas_index,
                    self.atlas_volume.shape,
                    session.atlas_tilt_ml_deg,
                    session.atlas_tilt_dv_deg,
                ).tolist()
            )
        session.probe_signal_values.append(self._probe_point_signal(session, slice_raw_point))
        session.point_history.append("probe")
        self._refresh_3d()
        self.status.setText(f"Added probe point {len(session.probe_atlas_points)}")

    def _probe_point_signal(self, session: SliceSession, slice_raw_point: tuple[float, float]) -> float:
        if session.weight_image is None:
            return 1.0
        slice_point = self._slice_raw_to_display_points(session, [slice_raw_point])[0]
        x, y = int(round(slice_point[0])), int(round(slice_point[1]))
        image = session.weight_image
        y0 = max(0, y - 3)
        y1 = min(image.shape[0], y + 4)
        x0 = max(0, x - 3)
        x1 = min(image.shape[1], x + 4)
        if x0 >= x1 or y0 >= y1:
            return 1.0
        return float(np.percentile(image[y0:y1, x0:x1], 75))

    def _refresh_points(self) -> None:
        session = self.current_session()
        if session is None:
            self.atlas_panel.set_points([], [])
            self.slice_panel.set_points([], [])
            self._refresh_point_counts()
            return
        self.atlas_panel.set_points(session.atlas_landmarks, session.probe_atlas_points)
        self.slice_panel.set_points(
            self._slice_raw_to_display_points(session, session.slice_landmarks),
            self._slice_raw_to_display_points(session, session.probe_slice_points),
        )
        self._refresh_point_counts()

    def _refresh_point_counts(self) -> None:
        session = self.current_session()
        if session is None:
            self.point_counts.setText("Transform atlas 0 / slice 0 | Probe 0")
            return
        n_pairs = min(len(session.atlas_landmarks), len(session.slice_landmarks))
        self.point_counts.setText(
            f"Transform atlas {len(session.atlas_landmarks)} / slice {len(session.slice_landmarks)} ({n_pairs} pairs) | "
            f"Probe {len(session.probe_slice_points)}"
        )

    def _point_target_changed(self, *_: object) -> None:
        self._refresh_atlas()
        if self.landmark_mode.isChecked():
            self.status.setText("Point target: transform landmarks")
        else:
            self.status.setText("Point target: probe trajectory")

    def _trajectory_weighting_changed(self, enabled: bool) -> None:
        self._refresh_3d()
        self.status.setText("Brightness-weighted trajectory on" if enabled else "Brightness-weighted trajectory off")

    def undo_last_point(self) -> None:
        session = self.current_session()
        if session is None:
            return
        while session.point_history:
            action = session.point_history.pop()
            if action == "atlas_landmark" and session.atlas_landmarks:
                session.atlas_landmarks.pop()
                self._invalidate_transform_after_landmark_edit(session)
                self.status.setText("Undid atlas transform landmark")
                break
            if action == "slice_landmark" and session.slice_landmarks:
                session.slice_landmarks.pop()
                self._invalidate_transform_after_landmark_edit(session)
                self.status.setText("Undid slice transform landmark")
                break
            if action == "probe" and session.probe_slice_points:
                if session.probe_atlas_points:
                    session.probe_atlas_points.pop()
                session.probe_slice_points.pop()
                if session.probe_volume_points:
                    session.probe_volume_points.pop()
                if session.probe_signal_values:
                    session.probe_signal_values.pop()
                self.status.setText("Undid probe point")
                break
        else:
            self.status.setText("No points to undo on current slice")
            return
        self._refresh_atlas()
        self._refresh_points()
        self._refresh_3d()

    def clear_current_points(self) -> None:
        session = self.current_session()
        if session is None:
            return
        if self.landmark_mode.isChecked():
            session.atlas_landmarks.clear()
            session.slice_landmarks.clear()
            session.point_history = [action for action in session.point_history if action not in {"atlas_landmark", "slice_landmark"}]
            self._invalidate_transform_after_landmark_edit(session)
            self.status.setText("Cleared transform landmarks on current slice")
        else:
            session.probe_atlas_points.clear()
            session.probe_slice_points.clear()
            session.probe_volume_points.clear()
            session.probe_signal_values.clear()
            session.point_history = [action for action in session.point_history if action != "probe"]
            self.status.setText("Cleared probe points on current slice")
        self._refresh_atlas()
        self._refresh_points()
        self._refresh_3d()

    def transform_current_slice(self) -> None:
        session = self.current_session()
        if session is None or session.rotated is None or self.current_atlas_image is None:
            return
        n = self._rebuild_slice_transform(session)
        if n is None:
            return
        self._recompute_probe_points_from_slice_points(session)
        self.probe_mode.setChecked(True)
        self._refresh_atlas()
        self._refresh_3d()
        self.status.setText(f"Transformed {session.name} using {n} point pairs")

    def _rebuild_slice_transform(self, session: SliceSession) -> int | None:
        if session.rotated is None or self.current_atlas_image is None:
            return None
        n = min(len(session.atlas_landmarks), len(session.slice_landmarks))
        if n < 3:
            QtWidgets.QMessageBox.warning(self, "More points needed", "Add at least 3 corresponding points on the atlas and slice.")
            return None
        atlas_points = np.asarray(session.atlas_landmarks[:n], dtype=np.float64)
        slice_points = np.asarray(self._slice_raw_to_display_points(session, session.slice_landmarks[:n]), dtype=np.float64)
        session.slice_to_atlas_x = Rbf(slice_points[:, 0], slice_points[:, 1], atlas_points[:, 0], function="thin_plate", smooth=0.0)
        session.slice_to_atlas_y = Rbf(slice_points[:, 0], slice_points[:, 1], atlas_points[:, 1], function="thin_plate", smooth=0.0)
        session.atlas_to_slice_x = Rbf(atlas_points[:, 0], atlas_points[:, 1], slice_points[:, 0], function="thin_plate", smooth=0.0)
        session.atlas_to_slice_y = Rbf(atlas_points[:, 0], atlas_points[:, 1], slice_points[:, 1], function="thin_plate", smooth=0.0)
        h, w = self.current_atlas_image.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = session.atlas_to_slice_x(xx, yy).astype(np.float32)
        map_y = session.atlas_to_slice_y(xx, yy).astype(np.float32)
        session.transformed_overlay = cv2.remap(
            session.rotated,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return n

    def _recompute_probe_points_from_slice_points(self, session: SliceSession) -> None:
        if session.slice_to_atlas_x is None or session.slice_to_atlas_y is None:
            return
        atlas_points: list[tuple[float, float]] = []
        for sx, sy in self._slice_raw_to_display_points(session, session.probe_slice_points):
            atlas_point = (float(session.slice_to_atlas_x(sx, sy)), float(session.slice_to_atlas_y(sx, sy)))
            atlas_points.append(atlas_point)
        session.probe_atlas_points = atlas_points
        self._recompute_session_volume_points(session)

    def _recompute_session_volume_points(self, session: SliceSession) -> None:
        if self.atlas_volume is None:
            return
        session.probe_volume_points = [
            point_to_volume(
                point,
                session.atlas_plane,
                session.atlas_index,
                self.atlas_volume.shape,
                session.atlas_tilt_ml_deg,
                session.atlas_tilt_dv_deg,
            ).tolist()
            for point in session.probe_atlas_points
        ]

    def all_probe_volume_points(self) -> np.ndarray:
        points = []
        for session in self.sessions:
            points.extend(session.probe_volume_points)
        return np.asarray(points, dtype=np.float64)

    def all_probe_signal_values(self) -> np.ndarray:
        values = []
        for session in self.sessions:
            values.extend(session.probe_signal_values)
        return np.asarray(values, dtype=np.float64)

    def probe_regression_weights(self, n_points: int) -> np.ndarray:
        if not self.brightness_weighting.isChecked():
            return np.ones(n_points, dtype=np.float64)
        values = self.all_probe_signal_values()
        if len(values) != n_points or n_points < 2:
            return np.ones(n_points, dtype=np.float64)
        values = np.nan_to_num(values, nan=np.nanmedian(values), posinf=np.nanmax(values), neginf=np.nanmin(values))
        lo, hi = np.percentile(values, [10, 95])
        if hi <= lo:
            return np.ones(n_points, dtype=np.float64)
        normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        return 0.15 + 0.85 * normalized**2

    def probe_regression(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        points = self.all_probe_volume_points()
        if len(points) < 2:
            return None, None
        weights = self.probe_regression_weights(len(points))
        center = np.average(points, axis=0, weights=weights)
        _, _, vh = np.linalg.svd((points - center) * np.sqrt(weights[:, None]), full_matrices=False)
        direction = vh[0]
        direction = direction / np.linalg.norm(direction)
        return center, direction

    def _refresh_3d(self) -> None:
        for item in self.dynamic_gl_items:
            self.view3d.removeItem(item)
        self.dynamic_gl_items.clear()
        if self.atlas_volume is not None:
            tilt_ml, tilt_dv = self._current_atlas_tilts()
            corners = volume_to_gl(
                section_plane_corners(
                    self.atlas_volume.shape,
                    self.plane_box.currentText(),
                    self.section_slider.value(),
                    tilt_ml,
                    tilt_dv,
                )
            )
            plane_item = gl.GLMeshItem(
                vertexes=corners,
                faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
                color=(0.10, 0.78, 1.0, 0.20),
                smooth=False,
                shader="shaded",
            )
            plane_item.setGLOptions("translucent")
            outline = gl.GLLinePlotItem(
                pos=np.vstack([corners, corners[0]]),
                color=(0.15, 0.85, 1.0, 0.95),
                width=2,
                antialias=True,
            )
            self.view3d.addItem(plane_item)
            self.view3d.addItem(outline)
            self.dynamic_gl_items.extend([plane_item, outline])
        points = self.all_probe_volume_points()
        if len(points) == 0:
            return
        weights = self.probe_regression_weights(len(points))
        scatter = gl.GLScatterPlotItem(pos=volume_to_gl(points), color=(1.0, 0.1, 0.45, 1.0), size=6 + 7 * weights)
        self.view3d.addItem(scatter)
        self.dynamic_gl_items.append(scatter)
        center, direction = self.probe_regression()
        if center is None or direction is None:
            return
        projection = (points - center) @ direction
        start = center + direction * (projection.min() - 80)
        stop = center + direction * (projection.max() + 220)
        line = gl.GLLinePlotItem(pos=volume_to_gl(np.vstack([start, stop])), color=(1.0, 0.0, 0.0, 1.0), width=4, antialias=True)
        self.view3d.addItem(line)
        self.dynamic_gl_items.append(line)

    def _resolve_data_folder(self, run_folder: Path) -> Path:
        if (run_folder / "channels.csv").exists() and (run_folder / "units.csv").exists():
            return run_folder
        return run_folder / "preprocessed_data"

    def _refresh_probe_names(self) -> None:
        selected = self.probe_name.currentText()
        self.probe_name.clear()
        data_folder = self._resolve_data_folder(Path(self.run_folder.text().strip()))
        channels_path = data_folder / "channels.csv"
        if not channels_path.exists():
            return
        try:
            channels = canonical_channel_keys(pd.read_csv(channels_path))
        except Exception as exc:
            self.status.setText(f"Could not read probe names: {exc}")
            return
        names = sorted(channels["probe_name"].dropna().astype(str).unique())
        self.probe_name.addItems(names)
        if selected in names:
            self.probe_name.setCurrentText(selected)

    def _sample_region(self, coord: np.ndarray) -> tuple[int | None, str, str, tuple[int | None, int | None, int | None]]:
        if self.annotation_volume is None:
            return None, "", "", (None, None, None)
        index = np.rint(np.asarray(coord, dtype=float)).astype(int)
        if np.any(index < 0) or np.any(index >= np.asarray(self.annotation_volume.shape)):
            return None, "", "", (None, None, None)
        ap, dv, ml = (int(value) for value in index)
        region_id = int(self.annotation_volume[ap, dv, ml])
        if region_id == 0:
            return None, "", "", (ap, dv, ml)
        name, acronym = self.region_names.get(region_id, (str(region_id), ""))
        return region_id, name, acronym, (ap, dv, ml)

    def map_channels_units(self) -> None:
        run_folder = Path(self.run_folder.text().strip())
        data_folder = self._resolve_data_folder(run_folder)
        channels_path = data_folder / "channels.csv"
        units_path = data_folder / "units.csv"
        if not channels_path.exists() or not units_path.exists():
            QtWidgets.QMessageBox.warning(self, "CSV files missing", f"Missing channels.csv or units.csv in:\n{data_folder}")
            return
        center, direction = self.probe_regression()
        points = self.all_probe_volume_points()
        if center is None or direction is None or len(points) < 2:
            QtWidgets.QMessageBox.warning(self, "Probe line missing", "Add at least two probe points before mapping channels.")
            return
        if self.probe_type.currentText() == "Neuropixels 2.0 four-shank":
            QtWidgets.QMessageBox.warning(
                self,
                "Four-shank mapping needs orientation",
                "This trajectory fit has no probe-roll measurement, so four-shank contacts cannot be assigned "
                "to atlas regions unambiguously.",
            )
            return
        selected_probe = self.probe_name.currentText().strip()
        if not selected_probe:
            QtWidgets.QMessageBox.warning(self, "Probe missing", "Select imec0, imec1, or another available probe.")
            return
        endpoint_reference = self.endpoint_reference.currentData()
        if endpoint_reference is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Trajectory reference required",
                "Specify whether the deepest marked point is the chanMap y=0 recording contact or the physical probe tip.",
            )
            return
        y0_offset_um = 0.0
        if endpoint_reference == "physical_tip":
            y0_offset_um = self.marked_tip_to_y0_um.value()
            if y0_offset_um <= 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Tip offset required",
                    "Enter the positive physical-tip to lowest-recording-contact distance for this probe.",
                )
                return
        projections = (points - center) @ direction
        line_points = np.vstack([center + direction * projections.min(), center + direction * projections.max()])
        marked_endpoint = line_points[np.argmax(line_points[:, 1])]
        surface_direction = direction if direction[1] < 0 else -direction
        y0_contact = marked_endpoint + surface_direction * (y0_offset_um / VOXEL_UM)

        channels = canonical_channel_keys(pd.read_csv(channels_path))
        units = canonical_channel_keys(pd.read_csv(units_path), units=True)
        selected = channels["probe_name"].eq(selected_probe)
        if not selected.any():
            QtWidgets.QMessageBox.warning(self, "Probe missing", f"{selected_probe} is not present in {channels_path}")
            return
        if "probe_vertical_position" not in channels.columns:
            raise ValueError("channels.csv has no probe_vertical_position/y_um geometry")
        distance_um = pd.to_numeric(
            channels.loc[selected, "probe_vertical_position"], errors="raise"
        ).to_numpy(dtype=float)

        coords = y0_contact[None, :] + surface_direction[None, :] * (distance_um[:, None] / VOXEL_UM)
        sampled = [self._sample_region(coord) for coord in coords]
        stereotaxic = np.asarray([volume_to_stereotaxic_um(coord, self.bregma_voxel) for coord in coords])
        mapped_at = datetime.now().isoformat(timespec="seconds")
        assignments = {
            "structure_id": [item[0] for item in sampled],
            "structure_name": [item[1] for item in sampled],
            "structure_acronym": [item[2] for item in sampled],
            "ccf_ap_index": [item[3][0] for item in sampled],
            "ccf_dv_index": [item[3][1] for item in sampled],
            "ccf_ml_index": [item[3][2] for item in sampled],
            "atlas_region_id": [item[0] for item in sampled],
            "atlas_region": [item[1] for item in sampled],
            "atlas_acronym": [item[2] for item in sampled],
            "atlas_ap": [item[3][0] for item in sampled],
            "atlas_dv": [item[3][1] for item in sampled],
            "atlas_ml": [item[3][2] for item in sampled],
            "stereotaxic_ap_um": stereotaxic[:, 0],
            "stereotaxic_dv_um": stereotaxic[:, 1],
            "stereotaxic_ml_um": stereotaxic[:, 2],
            "trajectory_distance_um": distance_um,
            "probe_type": self.probe_type.currentText(),
            "anatomy_source": "proprietary_HERBS",
            "anatomy_assignment_method": "peak_channel_on_trajectory_centerline",
            "anatomy_mapped_at": mapped_at,
        }
        for name, values in assignments.items():
            if name not in channels.columns:
                channels[name] = pd.Series(pd.NA, index=channels.index, dtype="object")
            channels.loc[selected, name] = values

        units = attach_peak_channel_metadata(channels, units)
        selected_units = units["probe_name"].eq(selected_probe)
        if not units.loc[selected_units, "structure_acronym"].fillna("").astype(str).str.len().gt(0).any():
            QtWidgets.QMessageBox.warning(
                self,
                "No atlas structures assigned",
                f"The {selected_probe} trajectory did not intersect a labelled atlas structure. No files were changed.",
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = data_folder / "anatomy" / "backups" / f"{timestamp}_{selected_probe}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(channels_path, backup_dir / "channels.csv")
        shutil.copy2(units_path, backup_dir / "units.csv")
        write_csv_atomic(channels, channels_path)
        write_csv_atomic(units, units_path)
        write_anatomy_sidecars(data_folder, channels, units)
        self._write_manifest(
            data_folder,
            selected_probe,
            str(endpoint_reference),
            marked_endpoint,
            y0_contact,
            surface_direction,
        )
        mapped_channels = channels.loc[selected, "structure_acronym"].fillna("").astype(str).str.len().gt(0).sum()
        mapped_units = units.loc[selected_units, "structure_acronym"].fillna("").astype(str).str.len().gt(0).sum()
        self.status.setText(
            f"Mapped {selected_probe}: {mapped_channels}/{selected.sum()} channels, "
            f"{mapped_units}/{selected_units.sum()} units; assignments saved by peak channel"
        )

    def undo_file_mapping(self) -> None:
        run_folder = Path(self.run_folder.text().strip())
        data_folder = self._resolve_data_folder(run_folder)
        paths = [data_folder / "channels.csv", data_folder / "units.csv"]
        existing_paths = [path for path in paths if path.exists()]
        if not existing_paths:
            QtWidgets.QMessageBox.warning(self, "CSV files missing", f"Missing channels.csv and units.csv in:\n{data_folder}")
            return

        tables: list[tuple[Path, pd.DataFrame, list[str]]] = []
        for path in existing_paths:
            table = pd.read_csv(path)
            drop_cols = [col for col in ANATOMY_MAPPING_COLUMNS if col in table.columns]
            if drop_cols:
                tables.append((path, table, drop_cols))

        if not tables:
            self.status.setText(f"No anatomy mapping columns found in {data_folder}")
            return

        anatomy_dir = data_folder / "anatomy"
        sidecars = [
            anatomy_dir / "channel_brain_regions.csv",
            anatomy_dir / "unit_brain_region_assignments.csv",
            *sorted(anatomy_dir.glob("proprietary_trajectory_manifest_*.json")),
        ]
        sidecars = [path for path in sidecars if path.exists()]
        summary = "\n".join(f"{path.name}: {', '.join(cols)}" for path, _, cols in tables)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Undo file mapping",
            "Remove only these anatomy mapping columns?\n\n"
            f"{summary}\n\n"
            "Timestamped backups will be written beside each CSV before any file is changed.",
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            self.status.setText("Undo file mapping cancelled")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        undo_log = {"data_folder": str(data_folder), "timestamp": timestamp, "files": []}
        backup_dir = anatomy_dir / "backups" / f"{timestamp}_undo_all"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path, table, drop_cols in tables:
            backup = backup_dir / path.name
            shutil.copy2(path, backup)
            write_csv_atomic(table.drop(columns=drop_cols), path)
            undo_log["files"].append({"csv": str(path), "backup": str(backup), "removed_columns": drop_cols})

        for path in sidecars:
            backup = backup_dir / path.name
            shutil.copy2(path, backup)
            path.unlink()
            undo_log["files"].append({"removed": str(path), "backup": str(backup)})

        log_path = anatomy_dir / f"undo_file_mapping_{timestamp}.json"
        log_path.write_text(json.dumps(undo_log, indent=2), encoding="utf-8")
        self.status.setText(f"Removed anatomy mapping; recoverable backups are in {backup_dir}")

    def _write_manifest(
        self,
        data_folder: Path,
        probe_name: str,
        endpoint_reference: str,
        marked_endpoint: np.ndarray,
        y0_contact: np.ndarray,
        surface_direction: np.ndarray,
    ) -> None:
        anatomy_dir = data_folder / "anatomy"
        anatomy_dir.mkdir(exist_ok=True)
        manifest = {
            "probe_name": probe_name,
            "atlas_folder": str(self.atlas_folder),
            "voxel_um": VOXEL_UM,
            "bregma_um_mlapdv": DEFAULT_BREGMA_UM_ML_AP_DV.tolist(),
            "bregma_voxel_ap_dv_ml": self.bregma_voxel.tolist(),
            "stereotaxic_axis_sign_ap_dv_ml": STEREOTAXIC_AXIS_SIGN_AP_DV_ML.tolist(),
            "probe_type": self.probe_type.currentText(),
            "channel_identity": ["probe_name", "probe_channel_number"],
            "unit_assignment": "structure_acronym inherited from the unit peak probe channel",
            "trajectory_sampling": "shank centerline at probe_vertical_position; horizontal position is retained but no probe-roll estimate is available",
            "vertical_reference": "probe_vertical_position is relative to the chanMap y=0 recording contact",
            "marked_endpoint_reference": endpoint_reference,
            "marked_endpoint_to_y0_contact_um": (
                self.marked_tip_to_y0_um.value() if endpoint_reference == "physical_tip" else 0.0
            ),
            "brightness_weighted_trajectory": self.brightness_weighting.isChecked(),
            "marked_endpoint_voxel_ap_dv_ml": marked_endpoint.tolist(),
            "marked_endpoint_stereotaxic_um_ap_dv_ml": volume_to_stereotaxic_um(marked_endpoint, self.bregma_voxel).tolist(),
            "y0_contact_voxel_ap_dv_ml": y0_contact.tolist(),
            "y0_contact_stereotaxic_um_ap_dv_ml": volume_to_stereotaxic_um(y0_contact, self.bregma_voxel).tolist(),
            "surface_direction_ap_dv_ml": surface_direction.tolist(),
            "slices": [
                {
                    "name": session.name,
                    "path": session.path,
                    "display_scale": session.display_scale,
                    "rotation_deg": session.rotation_deg,
                    "flip_horizontal": session.flip_horizontal,
                    "flip_vertical": session.flip_vertical,
                    "atlas_plane": session.atlas_plane,
                    "atlas_index": session.atlas_index,
                    "atlas_tilt_ml_deg": session.atlas_tilt_ml_deg,
                    "atlas_tilt_dv_deg": session.atlas_tilt_dv_deg,
                    "atlas_landmarks": session.atlas_landmarks,
                    "slice_landmarks": self._slice_raw_to_display_points(session, session.slice_landmarks),
                    "slice_landmarks_raw": session.slice_landmarks,
                    "probe_atlas_points": session.probe_atlas_points,
                    "probe_slice_points": self._slice_raw_to_display_points(session, session.probe_slice_points),
                    "probe_slice_points_raw": session.probe_slice_points,
                    "probe_volume_points": session.probe_volume_points,
                    "probe_signal_values": session.probe_signal_values,
                }
                for session in self.sessions
            ],
        }
        manifest_path = anatomy_dir / f"proprietary_trajectory_manifest_{probe_name}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    app = QtWidgets.QApplication([])
    app.setStyleSheet(
        """
        QWidget { background:#161b22; color:#d7e7f5; font-size:10pt; }
        QGroupBox { border:1px solid #2d3a4c; border-radius:7px; margin-top:8px; padding:8px; }
        QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {
            background:#0f131a; border:1px solid #2d3a4c; border-radius:5px; padding:4px;
        }
        QHeaderView::section { background:#1b2634; color:#d7e7f5; padding:5px; border:1px solid #2d3a4c; }
        QPushButton { background:#24415a; border:1px solid #41627f; border-radius:6px; padding:7px 12px; }
        QPushButton:hover { background:#2d526f; }
        QSplitter::handle { background:#2d3a4c; }
        """
    )
    window = TrajectoryTrackerWindow(
        default_atlas_folder=os.environ.get("TRAJECTORY_ATLAS_FOLDER", str(DEFAULT_ATLAS_FOLDER)),
        default_slices_folder=os.environ.get("TRAJECTORY_SLICES_FOLDER", ""),
        default_run_folder=os.environ.get("TRAJECTORY_RUN_FOLDER", ""),
    )
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
