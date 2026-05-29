from __future__ import annotations

from pathlib import Path
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Polygon, Rectangle
from PIL import Image, ImageDraw, ImageFont


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "slides" / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.face_segmentation_demo import _rgb_mouth_evidence, detect_face_bbox

PRIMARY_DEMO_PREFIX = "demo_trump"


def _demo_figure_name(suffix: str) -> str:
    return f"{PRIMARY_DEMO_PREFIX}_{suffix}.png"

def _load_rgb(name: str) -> np.ndarray:
    path = FIGURES_DIR / name
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return np.full((720, 960, 3), 245, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_gray(name: str) -> np.ndarray:
    path = FIGURES_DIR / name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.full((720, 960), 235, dtype=np.uint8)
    return image


def _save_matplotlib(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def _synthetic_soccer_scene() -> np.ndarray:
    h, w = 640, 640
    image = np.full((h, w, 3), (68, 142, 88), dtype=np.uint8)
    for y in range(0, h, 40):
        color = (58, 128, 78) if (y // 40) % 2 else (78, 154, 96)
        image[y : y + 40, :] = color

    cv2.line(image, (0, h // 2), (w, h // 2), (235, 245, 235), 3)
    cv2.circle(image, (w // 2, h // 2), 100, (235, 245, 235), 3)
    cv2.rectangle(image, (45, 170), (170, 470), (235, 245, 235), 3)
    cv2.rectangle(image, (470, 170), (595, 470), (235, 245, 235), 3)

    center = (330, 310)
    radius = 118
    cv2.circle(image, center, radius, (245, 245, 238), -1)
    cv2.circle(image, center, radius, (28, 38, 44), 4)

    pentagon = []
    for k in range(5):
        angle = -np.pi / 2 + 2 * np.pi * k / 5
        pentagon.append((int(center[0] + 34 * np.cos(angle)), int(center[1] + 34 * np.sin(angle))))
    cv2.fillPoly(image, [np.array(pentagon, dtype=np.int32)], (24, 31, 36))

    for k in range(5):
        angle = -np.pi / 2 + 2 * np.pi * k / 5
        p1 = pentagon[k]
        p2 = (int(center[0] + 96 * np.cos(angle)), int(center[1] + 96 * np.sin(angle)))
        cv2.line(image, p1, p2, (24, 31, 36), 5)
        side_center = (int(center[0] + 82 * np.cos(angle)), int(center[1] + 82 * np.sin(angle)))
        small_poly = []
        for j in range(5):
            a = angle + np.pi / 5 + 2 * np.pi * j / 5
            small_poly.append((int(side_center[0] + 18 * np.cos(a)), int(side_center[1] + 18 * np.sin(a))))
        cv2.fillPoly(image, [np.array(small_poly, dtype=np.int32)], (24, 31, 36))

    cv2.circle(image, (145, 130), 54, (242, 216, 92), -1)
    cv2.circle(image, (145, 130), 54, (140, 112, 24), 4)
    cv2.rectangle(image, (455, 475), (570, 545), (180, 70, 50), -1)
    cv2.rectangle(image, (455, 475), (570, 545), (90, 35, 25), 4)
    return image


def _soccer_edges_and_boxes() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    original = _synthetic_soccer_scene()
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 60, 150)
    boxes = [
        (211, 190, 238, 238),
        (92, 77, 106, 106),
        (455, 475, 115, 70),
        (50, 175, 125, 230),
    ]
    return original, edges, boxes


def _grid_point(origin: tuple[float, float], u: np.ndarray, v: np.ndarray, i: float, j: float) -> np.ndarray:
    return np.array(origin, dtype=float) + i * u + j * v


def generate_2d_convolution_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    ax.set_xlim(0, 8.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    u = np.array([0.56, 0.25])
    v = np.array([0.62, -0.32])
    bottom_origin = (0.7, 1.05)
    top_origin = (2.2, 3.58)

    def draw_grid(origin: tuple[float, float], rows: int, cols: int, fill: str, edge: str,
                  highlight: tuple[int, int, int, int] | None = None, highlight_fill: str = "#1f6f83") -> None:
        for r in range(rows):
            for c in range(cols):
                pts = [
                    _grid_point(origin, u, v, c, r),
                    _grid_point(origin, u, v, c + 1, r),
                    _grid_point(origin, u, v, c + 1, r + 1),
                    _grid_point(origin, u, v, c, r + 1),
                ]
                color = fill
                if highlight is not None:
                    hc, hr, hw, hh = highlight
                    if hc <= c < hc + hw and hr <= r < hr + hh:
                        color = highlight_fill
                ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=edge, linewidth=1.35))

    draw_grid(bottom_origin, 7, 7, fill="#2a9bd6", edge="#123744", highlight=(2, 2, 3, 3), highlight_fill="#207a99")
    draw_grid(top_origin, 3, 3, fill="#27aaa0", edge="#123744", highlight=(1, 1, 1, 1), highlight_fill="#1d807c")

    patch_corners = [
        _grid_point(bottom_origin, u, v, 2, 2),
        _grid_point(bottom_origin, u, v, 5, 2),
        _grid_point(bottom_origin, u, v, 5, 5),
        _grid_point(bottom_origin, u, v, 2, 5),
    ]
    kernel_corners = [
        _grid_point(top_origin, u, v, 0, 0),
        _grid_point(top_origin, u, v, 3, 0),
        _grid_point(top_origin, u, v, 3, 3),
        _grid_point(top_origin, u, v, 0, 3),
    ]
    for start, end in zip(kernel_corners, patch_corners):
        ax.plot([start[0], end[0]], [start[1], end[1]], color="#384348", linewidth=1.1, alpha=0.75)

    ax.text(0.95, 0.72, "input image patch $I$", fontsize=11, weight="bold", color="#123744")
    ax.text(2.25, 4.18, "kernel $K$", fontsize=11, weight="bold", color="#123744")
    ax.add_patch(FancyArrowPatch((5.7, 2.15), (6.55, 2.15), arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2, color="#0b6b4b"))
    ax.add_patch(FancyBboxPatch((6.65, 1.72), 1.1, 0.78, boxstyle="round,pad=0.08",
                                linewidth=2, edgecolor="#0b6b4b", facecolor="#eef8f2"))
    ax.text(7.2, 2.11, "$G[m,n]$", ha="center", va="center", fontsize=13, weight="bold", color="#0b6b4b")
    ax.text(4.9, 2.62, "weighted\nsum", ha="center", va="center", fontsize=10, color="#0b6b4b")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08)
    _save_matplotlib(FIGURES_DIR / "2d_convolution_diagram.png")


def generate_filtering_concept() -> None:
    original = _load_rgb(_demo_figure_name("original"))
    gray = _load_gray(_demo_figure_name("gray"))
    edges = _load_gray(_demo_figure_name("edges"))
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.4)

    fig = plt.figure(figsize=(8, 6.2))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]
    panels = [
        ("Gray image as 2D signal", gray, "gray"),
        ("Gaussian smoothing", blurred, "gray"),
        ("Derivative-driven edge response", edges, "gray"),
    ]
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.axis("off")
    fig.suptitle("Filtering concept for classical edge detection", fontsize=16, weight="bold")

    arrows = [
        ((0.43, 0.70), (0.52, 0.70), "suppress\nnoise", (0.475, 0.77)),
        ((0.72, 0.55), (0.72, 0.43), "highlight\nboundaries", (0.82, 0.49)),
    ]
    for start, end, label, label_pos in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color="#0b6b4b",
        )
        fig.add_artist(arrow)
        fig.text(*label_pos, label, ha="center", va="center", fontsize=10, color="#0b6b4b")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.06)
    _save_matplotlib(FIGURES_DIR / "filtering_concept.png")


def generate_rgb_edge_pipeline() -> None:
    original = _load_rgb(_demo_figure_name("original"))
    h, w = original.shape[:2]
    scale = 360 / max(h, w)
    resized = cv2.resize(original, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    channel_specs = [
        ("R channel", 0, np.array([1.0, 0.0, 0.0])),
        ("G channel", 1, np.array([0.0, 1.0, 0.0])),
        ("B channel", 2, np.array([0.0, 0.0, 1.0])),
    ]

    fig = plt.figure(figsize=(10.5, 5.8))
    grid = fig.add_gridspec(3, 3, width_ratios=[1.25, 1.0, 1.0], hspace=0.34, wspace=0.36)
    original_axis = fig.add_subplot(grid[:, 0])
    original_axis.imshow(resized)
    original_axis.set_title("Color image", fontsize=12, weight="bold")
    original_axis.axis("off")

    channel_axes = []
    edge_axes = []
    for row, (title, channel_index, tint) in enumerate(channel_specs):
        channel = resized[:, :, channel_index]
        tinted = (channel[:, :, None] * tint[None, None, :]).astype(np.uint8)
        edges = cv2.Canny(channel, 70, 150)
        edge_rgb = np.zeros((*edges.shape, 3), dtype=np.uint8)
        edge_rgb[edges > 0] = (255 * tint).astype(np.uint8)

        channel_axis = fig.add_subplot(grid[row, 1])
        channel_axis.imshow(tinted)
        channel_axis.set_title(title, fontsize=11)
        channel_axis.axis("off")
        channel_axes.append(channel_axis)

        edge_axis = fig.add_subplot(grid[row, 2])
        edge_axis.imshow(edge_rgb)
        edge_axis.set_title(f"{title} edges", fontsize=11)
        edge_axis.axis("off")
        edge_axes.append(edge_axis)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.06)
    original_pos = original_axis.get_position()
    middle_channel_pos = channel_axes[1].get_position()
    arrows = [
        (
            (original_pos.x1 + 0.03, middle_channel_pos.y0 + middle_channel_pos.height / 2),
            (middle_channel_pos.x0 - 0.04, middle_channel_pos.y0 + middle_channel_pos.height / 2),
            "RGB\nsplit",
            ((original_pos.x1 + middle_channel_pos.x0) / 2, middle_channel_pos.y0 + middle_channel_pos.height * 0.68),
        ),
    ]
    for channel_axis, edge_axis in zip(channel_axes, edge_axes):
        channel_pos = channel_axis.get_position()
        edge_pos = edge_axis.get_position()
        y_center = channel_pos.y0 + channel_pos.height / 2
        arrows.append(
            (
                (channel_pos.x1 + 0.03, y_center),
                (edge_pos.x0 - 0.04, y_center),
                "edge\noperator",
                ((channel_pos.x1 + edge_pos.x0) / 2, y_center + channel_pos.height * 0.27),
            )
        )
    for start, end, label, label_pos in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2,
            color="#0b6b4b",
        )
        fig.add_artist(arrow)
        fig.text(*label_pos, label, ha="center", va="center", fontsize=9, color="#0b6b4b")

    _save_matplotlib(FIGURES_DIR / "rgb_edge_pipeline.png")


def generate_gaussian_kernel_panel() -> None:
    sigma = 1.0
    axis_values = np.arange(-2, 3)
    x_grid, y_grid = np.meshgrid(axis_values, axis_values)
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    dense_axis = np.linspace(-3, 3, 90)
    dense_x, dense_y = np.meshgrid(dense_axis, dense_axis)
    gaussian_surface = np.exp(-(dense_x**2 + dense_y**2) / (2 * sigma**2))
    gaussian_surface = gaussian_surface / gaussian_surface.max()

    fig = plt.figure(figsize=(5.8, 6.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.055], height_ratios=[1.0, 1.12], hspace=0.2, wspace=0.06)

    kernel_axis = fig.add_subplot(grid[0, 0])
    image = kernel_axis.imshow(kernel, cmap="YlGnBu")
    kernel_axis.set_title("5x5 Gaussian smoothing kernel", fontsize=11, pad=8)
    kernel_axis.set_xticks(range(5), axis_values)
    kernel_axis.set_yticks(range(5), axis_values)
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            kernel_axis.text(col, row, f"{kernel[row, col]:.3f}", ha="center", va="center", fontsize=8)
    colorbar_axis = fig.add_subplot(grid[0, 1])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=8)

    surface_axis = fig.add_subplot(grid[1, :], projection="3d")
    surface_axis.plot_surface(dense_x, dense_y, gaussian_surface, cmap="YlGnBu", linewidth=0, antialiased=True)
    surface_axis.set_title("2D Gaussian probability surface", fontsize=10, pad=2)
    surface_axis.set_xlabel("x", labelpad=3)
    surface_axis.set_ylabel("y", labelpad=3)
    surface_axis.set_zlabel("")
    surface_axis.view_init(elev=28, azim=-45)
    surface_axis.tick_params(labelsize=7, pad=0)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.05)
    kernel_position = kernel_axis.get_position()
    colorbar_axis.set_position(
        [
            kernel_position.x1 + 0.025,
            kernel_position.y0,
            0.022,
            kernel_position.height,
        ]
    )
    surface_position = surface_axis.get_position()
    surface_axis.set_position(
        [
            surface_position.x0 - 0.015,
            surface_position.y0,
            surface_position.width,
            surface_position.height,
        ]
    )
    shifted_surface_position = surface_axis.get_position()
    fig.text(
        shifted_surface_position.x1 + 0.045,
        shifted_surface_position.y0 + shifted_surface_position.height * 0.53,
        "probability",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
    )
    _save_matplotlib(FIGURES_DIR / "gaussian_kernel_panel.png")


def generate_contour_transition() -> None:
    original, edges, boxes = _soccer_edges_and_boxes()

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_canvas = np.full_like(original, 255)
    cv2.drawContours(contour_canvas, contours, -1, (25, 132, 74), 2)

    candidate_canvas = original.copy()
    for x, y, bw, bh in boxes[:4]:
        cv2.rectangle(candidate_canvas, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4))
    panels = [
        ("Original image", original, None),
        ("Edge pixels", edges, "gray"),
        ("Connected contour", contour_canvas, None),
        ("Candidate region", candidate_canvas, None),
    ]
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        if cmap:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.03, hspace=0.22, wspace=0.08)
    _save_matplotlib(FIGURES_DIR / "contour_transition.png")


def _contour_boxes_from_edges(edges: np.ndarray) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = edges.shape
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < 0.002 * w * h or area > 0.12 * w * h:
            continue
        boxes.append((x, y, bw, bh))
    return sorted(boxes, key=lambda item: item[2] * item[3], reverse=True)


def generate_contour_feature_measurements() -> None:
    original, edges, boxes = _soccer_edges_and_boxes()
    boxes = boxes[:6]
    h, w = edges.shape

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.6), gridspec_kw={"height_ratios": [1.1, 1.0]})
    axes[0].imshow(original)
    axes[0].set_title("Measured contour candidates", fontsize=12, weight="bold")
    axes[0].axis("off")

    for idx, (x, y, bw, bh) in enumerate(boxes, start=1):
        color = "#00c7d9" if idx <= 4 else "#f0a000"
        axes[0].add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor=color, linewidth=2))
        axes[0].plot(x + bw / 2, y + bh / 2, marker="o", markersize=4, color=color)
        axes[0].text(x + 4, y + 14, f"C{idx}", fontsize=8, weight="bold", color="white",
                     bbox=dict(facecolor=color, edgecolor="none", pad=1.5))

    axes[1].axis("off")
    axes[1].set_title("Feature descriptor", fontsize=12, weight="bold")
    rows = [
        ("bounding box", r"$(x,y,w,h)$"),
        ("area", r"$w \times h$"),
        ("aspect ratio", r"$w / h$"),
        ("centroid", r"$(x+\frac{w}{2},\,y+\frac{h}{2})$"),
        ("relative position", "upper / middle / lower"),
    ]
    for row, (name, value) in enumerate(rows):
        y = 0.88 - row * 0.15
        axes[1].add_patch(Rectangle((0.04, y - 0.055), 0.92, 0.095, transform=axes[1].transAxes,
                                    facecolor="#eef8f2" if row % 2 == 0 else "#ffffff",
                                    edgecolor="#0b6b4b", linewidth=1))
        axes[1].text(0.08, y, name, transform=axes[1].transAxes, fontsize=10.5, weight="bold",
                     color="#0b6b4b", va="center")
        axes[1].text(0.56, y, value, transform=axes[1].transAxes, fontsize=10.5,
                     color="#25343b", va="center")
    axes[1].text(0.5, 0.055, "Pixels become measurable geometric evidence.",
                 transform=axes[1].transAxes, fontsize=10.5, weight="bold", color="#25343b",
                 ha="center")

    fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.04, hspace=0.24)
    _save_matplotlib(FIGURES_DIR / "contour_feature_measurements.png")


def generate_pattern_rule_flow() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    boxes = [
        (0.25, 3.45, 1.2, 0.85, "Contour", "connected\npixels"),
        (2.05, 3.45, 1.2, 0.85, "Features", "area, ratio,\ncentroid"),
        (3.85, 3.45, 1.2, 0.85, "Rule", "geometry +\nposition"),
        (5.65, 3.45, 1.2, 0.85, "Pattern", "candidate\nhypothesis"),
    ]
    for x, y, bw, bh, title, body in boxes:
        patch = FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.07",
                               linewidth=2, edgecolor="#0b6b4b", facecolor="#eef8f2")
        ax.add_patch(patch)
        ax.text(x + bw / 2, y + 0.58, title, ha="center", va="center",
                fontsize=11.5, weight="bold", color="#0b6b4b")
        ax.text(x + bw / 2, y + 0.23, body, ha="center", va="center",
                fontsize=8.5, color="#25343b")

    for x0, x1 in [(1.52, 1.98), (3.32, 3.78), (5.12, 5.58)]:
        ax.add_patch(FancyArrowPatch((x0, 3.5), (x1, 3.5), arrowstyle="-|>",
                                     mutation_scale=18, linewidth=2, color="#0b6b4b"))

    rules = [
        ("circle-like", "width ~= height + closed boundary"),
        ("ball-like", "large circle + internal texture"),
        ("non-ball", "rectangle or weak circular support"),
    ]
    for row, (label, rule) in enumerate(rules):
        y = 2.2 - row * 0.55
        ax.add_patch(Rectangle((0.8, y - 0.19), 5.6, 0.36, facecolor="#fff8e8",
                               edgecolor="#d89500", linewidth=1.2))
        ax.text(1.05, y, label, fontsize=10, weight="bold", color="#875a00", va="center")
        ax.text(2.55, y, rule, fontsize=9.2, color="#25343b", va="center")

    ax.text(3.6, 4.95, "Contour measurements become rule-based hypotheses",
            ha="center", fontsize=13, weight="bold", color="#25343b")
    _save_matplotlib(FIGURES_DIR / "pattern_rule_flow.png")


def generate_pattern_candidate_hypotheses() -> None:
    original, _, _ = _soccer_edges_and_boxes()

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.imshow(original)
    ax.axis("off")

    candidates = [
        ((211, 190, 238, 238), "ball-like", "#00a6d6"),
        ((92, 77, 106, 106), "circle-like", "#20a35b"),
        ((455, 475, 115, 70), "non-ball", "#d93f3f"),
    ]
    for (x, y, bw, bh), label, color in candidates:
        ax.add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor=color, linewidth=2.5))
        ax.text(x, max(12, y - 8), label, fontsize=9, weight="bold", color="white",
                bbox=dict(facecolor=color, edgecolor="none", pad=2))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    _save_matplotlib(FIGURES_DIR / "pattern_candidate_hypotheses.png")


def generate_face_zone_map() -> None:
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    face = Ellipse((5, 7), 6.2, 9.8, edgecolor="#f4c542", facecolor="#fff8d8", linewidth=3)
    ax.add_patch(face)
    ax.add_patch(Rectangle((2.2, 8.0), 5.6, 2.1, facecolor="#d8f3ff", edgecolor="#00c7ff", linewidth=2))
    ax.add_patch(Rectangle((3.2, 5.6), 3.6, 1.9, facecolor="#ddffd8", edgecolor="#2ed62e", linewidth=2))
    ax.add_patch(Rectangle((2.7, 3.2), 4.6, 1.5, facecolor="#ffe8bf", edgecolor="#ff9f00", linewidth=2))
    ax.add_patch(Rectangle((3.0, 8.35), 1.5, 0.55, facecolor="none", edgecolor="#00c7ff", linewidth=2))
    ax.add_patch(Rectangle((5.5, 8.35), 1.5, 0.55, facecolor="none", edgecolor="#00c7ff", linewidth=2))
    ax.add_patch(Rectangle((4.55, 6.1), 0.9, 0.9, facecolor="none", edgecolor="#2ed62e", linewidth=2))
    ax.add_patch(Rectangle((4.0, 3.5), 2.0, 0.7, facecolor="none", edgecolor="#ff9f00", linewidth=2))

    ax.text(5, 12.6, "Estimated face region", ha="center", fontsize=15, weight="bold", color="#8a6a00")
    ax.text(5, 9.95, "Upper zone: eye / eyebrow candidates", ha="center", fontsize=12, color="#008bb1", weight="bold")
    ax.text(5, 7.25, "Central zone: nose candidate", ha="center", fontsize=12, color="#169c16", weight="bold")
    ax.text(5, 4.55, "Lower zone: mouth candidate", ha="center", fontsize=12, color="#d17d00", weight="bold")
    ax.text(5, 0.95, "Geometric fallback keeps the pipeline deterministic\nwhen contour evidence is weak.", ha="center", fontsize=11, color="#2d3e46")
    fig.tight_layout()
    _save_matplotlib(FIGURES_DIR / "face_zone_map.png")


def generate_segmentation_pipeline_overview() -> None:
    original = _load_rgb(_demo_figure_name("original"))
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blur, 60, 150)
    segmentation = _load_rgb(_demo_figure_name("segmentation"))
    face_box = detect_face_bbox(gray, edges)
    fx, fy, fw, fh = face_box
    mouth_zone = (
        fx + int(0.18 * fw),
        fy + int(0.66 * fh),
        int(0.64 * fw),
        int(0.24 * fh),
    )
    mx, my, mw, mh = mouth_zone
    evidence = _rgb_mouth_evidence(original_bgr, edges, mouth_zone)
    evidence_norm = np.clip(evidence / max(1e-6, float(evidence.max())), 0.0, 1.0)

    edge_geometry = np.dstack([edges, edges, edges])
    fusion_img = original.copy()
    evidence_rgb = original.copy()
    heat = plt.get_cmap("inferno")(evidence_norm)[..., :3]
    patch = evidence_rgb[my : my + mh, mx : mx + mw].astype(np.float32) / 255.0
    fused_patch = 0.42 * patch + 0.58 * heat
    evidence_rgb[my : my + mh, mx : mx + mw] = np.clip(fused_patch * 255.0, 0, 255).astype(np.uint8)
    fusion_patch = fusion_img[my : my + mh, mx : mx + mw].astype(np.float32) / 255.0
    fusion_patch = 0.52 * fusion_patch + 0.48 * heat
    fusion_img[my : my + mh, mx : mx + mw] = np.clip(fusion_patch * 255.0, 0, 255).astype(np.uint8)

    face_margin = max(10, int(0.06 * fw))
    cx0 = max(0, fx - face_margin)
    cy0 = max(0, fy - face_margin)
    cx1 = min(original.shape[1], fx + fw + face_margin)
    cy1 = min(original.shape[0], fy + fh + face_margin)
    face_crop = original[cy0:cy1, cx0:cx1]
    face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
    face_smooth = cv2.cvtColor(cv2.GaussianBlur(face_crop_bgr, (7, 7), 1.2), cv2.COLOR_BGR2RGB)

    tile_h = 120
    tile_w = max(1, int(face_crop.shape[1] * tile_h / face_crop.shape[0]))
    rgb_tiles = []
    channel_colors = [(255, 90, 90), (70, 210, 110), (70, 120, 255)]
    for idx, (r, g, b) in enumerate(channel_colors):
        channel = cv2.resize(face_crop[:, :, idx], (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
        tinted = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tinted[:, :, 0] = (channel.astype(np.float32) * (r / 255.0)).astype(np.uint8)
        tinted[:, :, 1] = (channel.astype(np.float32) * (g / 255.0)).astype(np.uint8)
        tinted[:, :, 2] = (channel.astype(np.float32) * (b / 255.0)).astype(np.uint8)
        rgb_tiles.append(tinted)
    smooth_tile = cv2.resize(face_smooth, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
    gap = 12
    rgb_panel = np.full((tile_h * 2 + gap, tile_w * 2 + gap, 3), 255, dtype=np.uint8)
    rgb_panel[0:tile_h, 0:tile_w] = rgb_tiles[0]
    rgb_panel[0:tile_h, tile_w + gap: tile_w * 2 + gap] = rgb_tiles[1]
    rgb_panel[tile_h + gap: tile_h * 2 + gap, 0:tile_w] = rgb_tiles[2]
    rgb_panel[tile_h + gap: tile_h * 2 + gap, tile_w + gap: tile_w * 2 + gap] = smooth_tile

    fig = plt.figure(figsize=(15.2, 8.4))
    fig.suptitle("Classical facial-part segmentation pipeline", fontsize=18, weight="bold", y=0.972)

    axes = {
        "input": fig.add_axes([0.03, 0.23, 0.19, 0.62]),
        "gray": fig.add_axes([0.28, 0.62, 0.15, 0.25]),
        "edge": fig.add_axes([0.47, 0.59, 0.18, 0.30]),
        "rgb": fig.add_axes([0.25, 0.18, 0.40, 0.35]),
        "fusion": fig.add_axes([0.73, 0.31, 0.14, 0.43]),
        "overlay": fig.add_axes([0.89, 0.23, 0.16, 0.62]),
    }
    panels = {
        "input": ("Input face image", original, None),
        "gray": ("Grayscale + blur", blur, "gray"),
        "edge": ("Face region + geometry cues", edge_geometry, None),
        "rgb": ("RGB channels + optional smoothing", rgb_panel, None),
        "fusion": ("Cue fusion", fusion_img, None),
        "overlay": ("Segmentation overlay", segmentation, None),
    }
    for key, ax in axes.items():
        title, image, cmap = panels[key]
        if cmap:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=12, weight="bold", pad=5)
        ax.axis("off")

    face_rect = Rectangle((fx, fy), fw, fh, linewidth=2.0, edgecolor="#ffe85a", facecolor="none")
    axes["edge"].add_patch(face_rect)
    axes["edge"].add_patch(Rectangle((mx, my), mw, mh, linewidth=2.0, edgecolor="#3ddc84", facecolor="none"))
    axes["rgb"].text(10, 18, "R", color="#cc2b2b", fontsize=12, weight="bold")
    axes["rgb"].text(tile_w + gap + 10, 18, "G", color="#159941", fontsize=12, weight="bold")
    axes["rgb"].text(10, tile_h + gap + 18, "B", color="#2459d1", fontsize=12, weight="bold")
    axes["rgb"].text(tile_w + gap + 10, tile_h + gap + 18, "smooth", color="#875a00", fontsize=11, weight="bold")
    axes["fusion"].add_patch(Rectangle((fx, fy), fw, fh, linewidth=2.0, edgecolor="#ffe85a", facecolor="none"))
    axes["fusion"].add_patch(Rectangle((mx, my), mw, mh, linewidth=2.2, edgecolor="#ffb22c", facecolor="none"))

    def center_right(ax):
        pos = ax.get_position()
        return pos.x1, pos.y0 + pos.height / 2

    def center_left(ax):
        pos = ax.get_position()
        return pos.x0, pos.y0 + pos.height / 2

    arrow_specs = [
        (center_right(axes["input"]), center_left(axes["gray"])),
        (center_right(axes["input"]), center_left(axes["rgb"])),
        (center_right(axes["gray"]), center_left(axes["edge"])),
        (center_right(axes["edge"]), center_left(axes["fusion"])),
        (center_right(axes["rgb"]), center_left(axes["fusion"])),
        (center_right(axes["fusion"]), center_left(axes["overlay"])),
    ]
    for start, end in arrow_specs:
        fig.add_artist(
            FancyArrowPatch(
                start,
                end,
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.2,
                color="#0b6b4b",
                shrinkA=8,
                shrinkB=8,
            )
        )

    fig.text(0.225, 0.515, "split", fontsize=11, color="#0b6b4b", weight="bold")
    fig.text(0.678, 0.505, "merge", fontsize=11, color="#0b6b4b", weight="bold")
    _save_matplotlib(FIGURES_DIR / "segmentation_pipeline_overview_v3.png")


def generate_classical_vs_neural_cards() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    left = FancyBboxPatch((0.8, 0.7), 4.8, 4.6, boxstyle="round,pad=0.25", linewidth=2.0, edgecolor="#0b6b4b", facecolor="#eef8f2")
    right = FancyBboxPatch((6.4, 0.7), 4.8, 4.6, boxstyle="round,pad=0.25", linewidth=2.0, edgecolor="#875a00", facecolor="#fff6e5")
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(3.2, 4.95, "Classical pipeline", ha="center", va="center", fontsize=16, weight="bold", color="#0b6b4b")
    ax.text(8.8, 4.95, "Neural segmentation", ha="center", va="center", fontsize=16, weight="bold", color="#875a00")

    classical_lines = [
        "interpretable rules",
        "lightweight computation",
        "no training data",
        "easy to explain in class",
        "fragile under complex inputs",
    ]
    neural_lines = [
        "learned semantic features",
        "stronger robustness",
        "requires data and training",
        "higher model complexity",
        "less directly interpretable",
    ]
    for idx, line in enumerate(classical_lines):
        ax.text(1.2, 4.2 - idx * 0.7, f"• {line}", fontsize=13, color="#2d3e46")
    for idx, line in enumerate(neural_lines):
        ax.text(6.8, 4.2 - idx * 0.7, f"• {line}", fontsize=13, color="#2d3e46")
    ax.text(6.0, 0.18, "This project stays on the classical side to make the pipeline transparent and teachable.", ha="center", fontsize=12, color="#2d3e46")
    fig.tight_layout()
    _save_matplotlib(FIGURES_DIR / "classical_vs_neural_cards.png")


def generate_failure_cases_grid() -> None:
    base = Image.open(FIGURES_DIR / _demo_figure_name("original")).convert("RGB")
    width, height = base.size

    def shadow(img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        mask = np.linspace(0.35, 1.0, arr.shape[1], dtype=np.float32)[None, :, None]
        arr *= mask
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def low_res(img: Image.Image) -> Image.Image:
        small = img.resize((width // 5, height // 5), Image.Resampling.BILINEAR)
        return small.resize((width, height), Image.Resampling.NEAREST)

    def blur(img: Image.Image) -> Image.Image:
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        arr = cv2.GaussianBlur(arr, (17, 17), 5)
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    def occlusion(img: Image.Image) -> Image.Image:
        out = img.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle((width * 0.16, height * 0.18, width * 0.8, height * 0.31), fill=(25, 25, 25))
        draw.rectangle((width * 0.14, height * 0.52, width * 0.83, height * 0.67), fill=(55, 30, 20))
        return out

    def non_frontal(img: Image.Image) -> Image.Image:
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        shear = np.float32([[1.0, 0.18, -0.09 * width], [0.0, 1.0, 0.0]])
        warped = cv2.warpAffine(arr, shear, (width, height), borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    def clutter(img: Image.Image) -> Image.Image:
        out = img.copy()
        draw = ImageDraw.Draw(out)
        for offset in range(6):
            draw.line((width * 0.05, height * (0.16 + offset * 0.08), width * 0.95, height * (0.08 + offset * 0.09)), fill=(210, 170, 40), width=6)
        return out

    cases = [
        ("Shadow", shadow(base)),
        ("Blur", blur(base)),
        ("Low resolution", low_res(base)),
        ("Occlusion", occlusion(base)),
        ("Non-frontal", non_frontal(base)),
        ("Background clutter", clutter(base)),
    ]

    canvas = Image.new("RGB", (1020, 760), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    cell_w, cell_h = 300, 320
    positions = [(20, 70), (360, 70), (700, 70), (20, 410), (360, 410), (700, 410)]
    draw.text((20, 18), "Typical failure conditions for a rule-based classical pipeline", fill=(15, 80, 54), font=font)
    for (label, image), (x, y) in zip(cases, positions):
        scale = min(cell_w / image.width, cell_h / image.height)
        thumb_w = int(image.width * scale)
        thumb_h = int(image.height * scale)
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        paste_x = x + (cell_w - thumb_w) // 2
        paste_y = y + (cell_h - thumb_h) // 2
        canvas.paste(thumb, (paste_x, paste_y))
        draw.rectangle((x, y, x + cell_w, y + cell_h), outline=(15, 80, 54), width=3)
        draw.text((x, y - 24), label, fill=(45, 62, 70), font=font)
    canvas.save(FIGURES_DIR / "failure_cases_grid_v3.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_2d_convolution_diagram()
    generate_filtering_concept()
    generate_rgb_edge_pipeline()
    generate_gaussian_kernel_panel()
    generate_contour_transition()
    generate_contour_feature_measurements()
    generate_pattern_rule_flow()
    generate_pattern_candidate_hypotheses()
    generate_face_zone_map()
    generate_segmentation_pipeline_overview()
    generate_failure_cases_grid()
    print("Generated slide figures.")


if __name__ == "__main__":
    main()
