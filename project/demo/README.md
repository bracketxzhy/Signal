# Facial Segmentation Demo (Classical CV)

This demo uses **classical image processing methods only**.  
It demonstrates the transition from edge detection to pattern-based facial part segmentation.  
It is **not** a production-grade face parsing system.

## Features

- Select a frontal face image from local disk
- Run deterministic segmentation steps:
  - resize
  - grayscale
  - Gaussian blur
  - Canny edges
  - contour-based candidate extraction
  - geometric fallback zones
- Display:
  - original image
  - edge map
  - segmentation overlay
- Save outputs under:
  - `outputs/demo_original.png`
  - `outputs/demo_gray.png`
  - `outputs/demo_edges.png`
  - `outputs/demo_segmentation.png`
  - `outputs/demo_summary.png`

## Run from source

From project root:

```bash
python demo/face_segmentation_demo.py
```

Optional batch mode:

```bash
python demo/face_segmentation_demo.py --batch-input path/to/image.jpg --batch-prefix demo_biden
```

## Build single exe (Windows)

```bat
demo\build_exe.bat
```

Expected output:

- `dist/FacialSegmentationDemo.exe`
