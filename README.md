# Geometric Shape Detector

A Python application that detects geometric shapes in images or live video,
identifies their type, draws their contours, and calculates their geometric
properties.

## Features

1. **Image preprocessing** — grayscale conversion, Gaussian blur, Otsu
   thresholding, and Canny edge detection.
2. **Contour detection** — finds contours for object boundaries and filters
   out small noise contours (below `--min-area`).
3. **Polygon approximation & classification** — approximates contours with
   `approxPolyDP` and classifies them:
   - Triangle (3 vertices)
   - Square / Rectangle / Quadrilateral (4 vertices, checked via right angles
     and side equality)
   - Pentagon (5 vertices)
   - Hexagon (6 vertices)
   - Polygon with N sides (N > 6)
4. **Geometric measurements** — area, perimeter, bounding box (and bounding-box
   area) for every detected shape.

Detected shapes are drawn on the frame/image with:
- green contour outline, approximated polygon overlay, and bounding box
- a text label with shape name, area, and perimeter.

## Requirements

```bash
pip install opencv-python numpy
```

## Usage

```bash
# Detect shapes in a static image
python shape_detector.py --image photo.jpg

# Save the annotated result
python shape_detector.py --image photo.jpg --output result.jpg

# Real-time detection from webcam (press 'q' to quit)
python shape_detector.py --camera 0

# Generate a demo image containing triangles/squares/rectangles/pentagons/
# hexagons/polygons, run detection and save the output
python shape_detector.py --demo
```

### Options

| Option | Description |
|---|---|
| `--image` | Path to the input image. |
| `--camera` | Camera index for live video (default `0`). |
| `--min-area` | Minimum contour area in pixels to accept a shape (default `1000`). |
| `--output` | Path to save the annotated image. |
| `--demo` | Generate demo shapes, detect, save `demo_result.png`. |

## How it works

```
image -> [grayscale -> gaussian blur -> Otsu threshold -> Canny edges]
      -> [findContours + small-contour noise filter]
      -> [approxPolyDP polygon approximation]
      -> [classify: triangle / square / rectangle / quadrilateral / pentagon /
           hexagon / polygon]
      -> [area, perimeter, bounding box measurements]
      -> [draw contours, polygon, bounding box, label]
```

## Geometric measurements reported per shape

- **Area** — contour area in pixels (or polygon area via approximation).
- **Perimeter** — arc length of the contour (`cv2.arcLength(..., True)`).
- **Bounding box** — `cv2.boundingRect` of the contour: `(x, y, width, height)`
  plus bounding-box area `width * height`.

## Project structure

```
shape_detector.py   # main application (preprocessing, detection, classification, drawing)
README.md           # this file
```
