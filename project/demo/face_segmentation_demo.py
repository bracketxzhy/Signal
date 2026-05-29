from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"
SLIDE_FIGURE_DIR = PROJECT_ROOT / "slides" / "figures"


@dataclass
class SegmentationResult:
    original: np.ndarray
    gray: np.ndarray
    edges: np.ndarray
    overlay: np.ndarray


def resize_image(image: np.ndarray, target_width: int = 640) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= target_width:
        return image.copy()
    scale = target_width / float(width)
    target_height = int(height * scale)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def detect_face_bbox(gray: np.ndarray, edges: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(cascade_path)

    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(60, w // 8), max(60, h // 8)),
    )
    if len(faces) > 0:
        cx_target = w / 2.0
        cy_target = h * 0.45
        scored = []
        for (x, y, fw, fh) in faces:
            cx = x + fw / 2
            cy = y + fh / 2
            center_penalty = abs(cx - cx_target) / w + abs(cy - cy_target) / h
            area_bonus = (fw * fh) / float(w * h)
            score = area_bonus - 0.55 * center_penalty
            scored.append((score, (x, y, fw, fh)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[tuple[float, tuple[int, int, int, int]]] = None
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < 0.08 * w * h or area > 0.75 * w * h:
            continue
        ratio = bw / float(max(1, bh))
        if ratio < 0.55 or ratio > 1.45:
            continue
        cx = x + bw / 2
        cy = y + bh / 2
        center_penalty = abs(cx - w / 2.0) / w + abs(cy - h * 0.45) / h
        score = area / float(w * h) - 0.45 * center_penalty
        if best is None or score > best[0]:
            best = (score, (x, y, bw, bh))
    if best is not None:
        return best[1]

    fw = int(0.5 * w)
    fh = int(0.62 * h)
    x = (w - fw) // 2
    y = int(0.16 * h)
    return x, y, fw, fh


def _find_box_in_zone(
    edges: np.ndarray,
    zone: tuple[int, int, int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect: float,
    max_aspect: float,
    prefer_wide: bool = False,
    center_bias: float = 0.4,
) -> Optional[tuple[int, int, int, int]]:
    zx, zy, zw, zh = zone
    patch = edges[zy : zy + zh, zx : zx + zw]
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    patch_area = float(max(1, zw * zh))
    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / float(max(1, h))
        area_ratio = area / patch_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if ratio < min_aspect or ratio > max_aspect:
            continue
        cx = x + w / 2
        cy = y + h / 2
        center_penalty = abs(cx - zw / 2) / zw + abs(cy - zh / 2) / zh
        shape_bonus = ratio if prefer_wide else 1.0 / max(1e-6, abs(1.0 - ratio) + 0.2)
        score = area_ratio + 0.025 * shape_bonus - center_bias * center_penalty
        if best is None or score > best[0]:
            best = (score, (zx + x, zy + y, w, h))
    return None if best is None else best[1]


def _normalize_score_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    low, high = np.percentile(values, [5, 95])
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _rgb_mouth_evidence(image_bgr: np.ndarray, edges: np.ndarray, zone: tuple[int, int, int, int]) -> np.ndarray:
    zx, zy, zw, zh = zone
    patch = image_bgr[zy : zy + zh, zx : zx + zw]
    edge_patch = edges[zy : zy + zh, zx : zx + zw].astype(np.float32) / 255.0
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue

    red_dominance = red - 0.5 * (green + blue)
    channel_spread = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    darkness = 255.0 - luminance

    evidence = (
        0.44 * _normalize_score_map(red_dominance)
        + 0.24 * _normalize_score_map(channel_spread)
        + 0.18 * _normalize_score_map(darkness)
        + 0.14 * edge_patch
    )
    return evidence


def _find_mouth_box_rgb(
    image_bgr: np.ndarray,
    edges: np.ndarray,
    zone: tuple[int, int, int, int],
) -> Optional[tuple[int, int, int, int]]:
    zx, zy, zw, zh = zone
    evidence = _rgb_mouth_evidence(image_bgr, edges, zone)
    threshold = max(0.48, float(np.percentile(evidence, 86)))
    mask = (evidence >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    patch_area = float(max(1, zw * zh))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / patch_area
        aspect = w / float(max(1, h))
        if area_ratio < 0.008 or area_ratio > 0.22:
            continue
        if aspect < 1.35 or aspect > 7.0:
            continue
        cx = x + w / 2
        cy = y + h / 2
        center_penalty = abs(cx - zw / 2) / zw + 0.65 * abs(cy - zh * 0.44) / zh
        mean_evidence = float(evidence[y : y + h, x : x + w].mean())
        score = mean_evidence + 0.06 * min(aspect, 4.0) + 0.45 * area_ratio - 0.38 * center_penalty
        if best is None or score > best[0]:
            pad_x = int(0.18 * w)
            pad_y = int(0.30 * h)
            bx = max(0, x - pad_x)
            by = max(0, y - pad_y)
            bw = min(zw - bx, w + 2 * pad_x)
            bh = min(zh - by, h + 2 * pad_y)
            best = (score, (zx + bx, zy + by, bw, bh))
    return None if best is None else best[1]


def detect_eye_boxes(edges: np.ndarray, face_box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x, y, w, h = face_box
    zone = (
        x + int(0.08 * w),
        y + int(0.18 * h),
        int(0.84 * w),
        int(0.30 * h),
    )
    zx, zy, zw, zh = zone
    patch = edges[zy : zy + zh, zx : zx + zw]
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    patch_area = float(max(1, zw * zh))
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        area_ratio = (bw * bh) / patch_area
        aspect = bw / float(max(1, bh))
        if area_ratio < 0.003 or area_ratio > 0.07:
            continue
        if aspect < 1.1 or aspect > 5.2:
            continue
        cx = bx + bw / 2
        cy = by + bh / 2
        score = area_ratio + 0.015 * aspect - 0.22 * abs(cy - zh * 0.5) / zh
        candidates.append((score, (zx + bx, zy + by, bw, bh)))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[int, int, int, int]] = []
    for _, candidate in candidates:
        if len(selected) >= 2:
            break
        cx = candidate[0] + candidate[2] / 2
        too_close = any(abs(cx - (s[0] + s[2] / 2)) < 0.12 * w for s in selected)
        if not too_close:
            selected.append(candidate)

    if len(selected) == 2:
        selected.sort(key=lambda b: b[0])
        return selected

    ew = int(0.27 * w)
    eh = int(0.12 * h)
    ey = y + int(0.30 * h)
    left = (x + int(0.16 * w), ey, ew, eh)
    right = (x + int(0.57 * w), ey, ew, eh)
    return [left, right]


def enforce_min_box(
    box: tuple[int, int, int, int],
    face_box: tuple[int, int, int, int],
    min_w_ratio: float,
    min_h_ratio: float,
    clamp_y0: int,
    clamp_y1: int,
) -> tuple[int, int, int, int]:
    fx, fy, fw, fh = face_box
    x, y, w, h = box
    min_w = max(1, int(min_w_ratio * fw))
    min_h = max(1, int(min_h_ratio * fh))

    if w < min_w:
        cx = x + w // 2
        w = min_w
        x = cx - w // 2
    if h < min_h:
        cy = y + h // 2
        h = min_h
        y = cy - h // 2

    x = max(fx, min(x, fx + fw - w))
    y = max(fy + clamp_y0, min(y, fy + clamp_y1 - h))
    return x, y, w, h


def detect_nose_box(edges: np.ndarray, face_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = face_box
    zone = (
        x + int(0.30 * w),
        y + int(0.43 * h),
        int(0.40 * w),
        int(0.30 * h),
    )
    result = _find_box_in_zone(
        edges,
        zone,
        min_area_ratio=0.01,
        max_area_ratio=0.20,
        min_aspect=0.38,
        max_aspect=1.45,
        prefer_wide=False,
        center_bias=0.5,
    )
    if result is not None:
        return result
    return (
        x + int(0.38 * w),
        y + int(0.47 * h),
        int(0.24 * w),
        int(0.24 * h),
    )


def detect_mouth_box(
    edges: np.ndarray,
    face_box: tuple[int, int, int, int],
    image_bgr: Optional[np.ndarray] = None,
) -> tuple[int, int, int, int]:
    x, y, w, h = face_box
    zone = (
        x + int(0.18 * w),
        y + int(0.66 * h),
        int(0.64 * w),
        int(0.24 * h),
    )
    if image_bgr is not None:
        result = _find_mouth_box_rgb(image_bgr, edges, zone)
        if result is not None:
            return result

    result = _find_box_in_zone(
        edges,
        zone,
        min_area_ratio=0.01,
        max_area_ratio=0.24,
        min_aspect=1.15,
        max_aspect=6.2,
        prefer_wide=True,
        center_bias=0.35,
    )
    if result is not None:
        return result
    return (
        x + int(0.23 * w),
        y + int(0.72 * h),
        int(0.54 * w),
        int(0.16 * h),
    )


def run_classical_segmentation(image_bgr: np.ndarray) -> SegmentationResult:
    original = resize_image(image_bgr, target_width=640)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blur, 60, 150)

    face_box = detect_face_bbox(gray, edges)
    eye_boxes = detect_eye_boxes(edges, face_box)
    nose_box = detect_nose_box(edges, face_box)
    mouth_box = detect_mouth_box(edges, face_box, original)

    fx, fy, fw, fh = face_box
    # Merge two eye detections into one candidate zone for a segmentation-like view.
    left, right = eye_boxes[0], eye_boxes[1]
    ex0 = min(left[0], right[0])
    ey0 = min(left[1], right[1])
    ex1 = max(left[0] + left[2], right[0] + right[2])
    ey1 = max(left[1] + left[3], right[1] + right[3])
    eye_zone = (ex0, ey0, ex1 - ex0, ey1 - ey0)

    eye_zone = enforce_min_box(
        eye_zone,
        face_box,
        min_w_ratio=0.58,
        min_h_ratio=0.16,
        clamp_y0=int(0.16 * fh),
        clamp_y1=int(0.56 * fh),
    )
    nose_box = enforce_min_box(
        nose_box,
        face_box,
        min_w_ratio=0.16,
        min_h_ratio=0.16,
        clamp_y0=int(0.36 * fh),
        clamp_y1=int(0.78 * fh),
    )
    mouth_box = enforce_min_box(
        mouth_box,
        face_box,
        min_w_ratio=0.44,
        min_h_ratio=0.14,
        clamp_y0=int(0.58 * fh),
        clamp_y1=int(0.96 * fh),
    )

    overlay = original.copy()
    x, y, w, h = face_box
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(overlay, "Face Region", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    ex, ey, ew, eh = eye_zone
    cv2.rectangle(overlay, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
    cv2.putText(overlay, "Eye Candidate Zone", (ex, max(18, ey - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 0), 2)

    nx, ny, nw, nh = nose_box
    cv2.rectangle(overlay, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
    cv2.putText(overlay, "Nose Candidate Zone", (nx, max(18, ny - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2)

    mx, my, mw, mh = mouth_box
    cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (0, 165, 255), 2)
    cv2.putText(overlay, "Mouth Candidate Zone", (mx, max(18, my - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 165, 255), 2)

    return SegmentationResult(original=original, gray=gray, edges=edges, overlay=overlay)


def save_result_images(result: SegmentationResult, output_dir: Path, prefix: str = "demo") -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = output_dir / f"{prefix}_original.png"
    gray_path = output_dir / f"{prefix}_gray.png"
    edges_path = output_dir / f"{prefix}_edges.png"
    segmentation_path = output_dir / f"{prefix}_segmentation.png"
    summary_path = output_dir / f"{prefix}_summary.png"

    cv2.imwrite(str(original_path), result.original)
    cv2.imwrite(str(gray_path), result.gray)
    cv2.imwrite(str(edges_path), result.edges)
    cv2.imwrite(str(segmentation_path), result.overlay)

    summary = np.zeros((result.original.shape[0] * 2, result.original.shape[1] * 2, 3), dtype=np.uint8)
    gray_bgr = cv2.cvtColor(result.gray, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(result.edges, cv2.COLOR_GRAY2BGR)
    h, w = result.original.shape[:2]
    summary[0:h, 0:w] = result.original
    summary[0:h, w : 2 * w] = gray_bgr
    summary[h : 2 * h, 0:w] = edges_bgr
    summary[h : 2 * h, w : 2 * w] = result.overlay
    cv2.putText(summary, "Original", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(summary, "Gray", (w + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(summary, "Edges", (18, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(summary, "Segmentation", (w + 18, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(summary_path), summary)

    return {
        "original": original_path,
        "gray": gray_path,
        "edges": edges_path,
        "segmentation": segmentation_path,
        "summary": summary_path,
    }


def bgr_to_photoimage(image_bgr: np.ndarray, max_size: tuple[int, int] = (320, 240)) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image=image)


class FaceSegmentationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Classical Facial Part Segmentation Demo")
        self.root.geometry("1080x700")

        self.input_path: Optional[Path] = None
        self.result: Optional[SegmentationResult] = None

        self._photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        control = tk.Frame(self.root, padx=10, pady=10)
        control.pack(fill=tk.X)

        tk.Button(control, text="Select Face Image", command=self.select_image, width=18).pack(side=tk.LEFT, padx=5)
        tk.Button(control, text="Run Segmentation", command=self.run_segmentation, width=18).pack(side=tk.LEFT, padx=5)
        tk.Button(control, text="Save Results", command=self.save_results, width=18).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Status: waiting for input image.")
        tk.Label(control, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, padx=12)

        panel = tk.Frame(self.root, padx=12, pady=8)
        panel.pack(fill=tk.BOTH, expand=True)

        self.label_original = self._build_image_panel(panel, "Original", 0)
        self.label_edges = self._build_image_panel(panel, "Edge Map", 1)
        self.label_overlay = self._build_image_panel(panel, "Segmentation Overlay", 2)

    def _build_image_panel(self, parent: tk.Frame, title: str, col: int) -> tk.Label:
        frame = tk.LabelFrame(parent, text=title, padx=8, pady=8)
        frame.grid(row=0, column=col, sticky="nsew", padx=6, pady=6)
        parent.grid_columnconfigure(col, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        label = tk.Label(frame, text="No image", width=36, height=18, bg="#efefef")
        label.pack(fill=tk.BOTH, expand=True)
        return label

    def select_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a frontal face image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")],
        )
        if not path:
            self.status_var.set("Status: no image selected.")
            return
        self.input_path = Path(path)
        image = cv2.imread(str(self.input_path))
        if image is None:
            messagebox.showerror("Read Error", "Cannot read the selected image.")
            self.status_var.set("Status: image read failed.")
            self.input_path = None
            return
        self._show_image(self.label_original, image, "original_input")
        self.status_var.set(f"Status: selected {self.input_path.name}")

    def run_segmentation(self) -> None:
        if self.input_path is None:
            messagebox.showerror("Missing Input", "Please select an image first.")
            self.status_var.set("Status: no image selected.")
            return
        image = cv2.imread(str(self.input_path))
        if image is None:
            messagebox.showerror("Read Error", "Cannot read the selected image.")
            self.status_var.set("Status: image read failed.")
            return

        self.result = run_classical_segmentation(image)
        edges_bgr = cv2.cvtColor(self.result.edges, cv2.COLOR_GRAY2BGR)
        self._show_image(self.label_original, self.result.original, "original")
        self._show_image(self.label_edges, edges_bgr, "edges")
        self._show_image(self.label_overlay, self.result.overlay, "overlay")
        self.status_var.set("Status: segmentation completed.")

    def save_results(self) -> None:
        if self.result is None:
            messagebox.showerror("Missing Result", "Run segmentation before saving results.")
            self.status_var.set("Status: no result to save.")
            return
        saved = save_result_images(self.result, OUTPUT_DIR, prefix="demo")
        save_result_images(self.result, FIGURE_DIR, prefix="demo")
        save_result_images(self.result, SLIDE_FIGURE_DIR, prefix="demo")
        self.status_var.set(f"Status: results saved to {saved['summary'].parent}")
        messagebox.showinfo("Saved", f"Saved results under:\n{saved['summary'].parent}")

    def _show_image(self, label: tk.Label, image_bgr: np.ndarray, key: str) -> None:
        photo = bgr_to_photoimage(image_bgr)
        self._photo_refs[key] = photo
        label.configure(image=photo, text="")


def run_batch(input_image: Path, prefix: str) -> dict[str, Path]:
    image = cv2.imread(str(input_image))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_image}")
    result = run_classical_segmentation(image)
    out_main = save_result_images(result, OUTPUT_DIR, prefix=prefix)
    save_result_images(result, FIGURE_DIR, prefix=prefix)
    save_result_images(result, SLIDE_FIGURE_DIR, prefix=prefix)
    return out_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical facial part segmentation demo.")
    parser.add_argument("--batch-input", type=Path, help="Optional batch input image path.")
    parser.add_argument("--batch-prefix", type=str, default="demo", help="Prefix for batch output files.")
    args = parser.parse_args()

    if args.batch_input:
        paths = run_batch(args.batch_input, args.batch_prefix)
        print("Batch processing complete.")
        for key, path in paths.items():
            print(f"{key}: {path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SLIDE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    app = FaceSegmentationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
