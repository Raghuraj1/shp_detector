#!/usr/bin/env python3
"""Geometric Shape Detector"""

import argparse
import cv2
import numpy as np

MIN_CONTOUR_AREA = 1000
COLORS = {
    "Triangle": (255, 0, 0),
    "Square": (0, 165, 255),
    "Rectangle": (0, 255, 0),
    "Quadrilateral": (0, 255, 255),
    "Pentagon": (255, 0, 255),
    "Hexagon": (255, 255, 0),
    "Polygon": (180, 180, 180),
    "Unknown": (0, 0, 255),
}


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.Canny(thresh, 50, 150)


def has_right_angles(approx):
    n = len(approx)
    for i in range(n):
        a = approx[(i + 1) % n][0] - approx[i][0]
        b = approx[(i + 2) % n][0] - approx[(i + 1) % n][0]
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        if norm == 0 or abs(dot) > 0.1 * norm:
            return False
    return True


def is_square(approx):
    sides = []
    n = len(approx)
    for i in range(n):
        p0 = approx[i][0]
        p1 = approx[(i + 1) % n][0]
        sides.append(float(np.linalg.norm(p1 - p0)))
    longest = max(sides)
    if longest == 0 or (max(sides) / min(sides)) > 1.05:
        return False
    return has_right_angles(approx)


def classify(approx):
    n = len(approx)
    if n == 3:
        return "Triangle"
    if n == 4:
        if is_square(approx):
            return "Square"
        if has_right_angles(approx):
            return "Rectangle"
        return "Quadrilateral"
    if n == 5:
        return "Pentagon"
    if n == 6:
        return "Hexagon"
    if n > 6:
        return f"Polygon ({n} sides)"
    return "Unknown"


def find_shapes(img, min_area=MIN_CONTOUR_AREA):
    edges = preprocess(img)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        peri = float(cv2.arcLength(c, True))
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        shapes.append({
            "contour": c,
            "approx": approx,
            "name": classify(approx),
            "vertices": int(len(approx)),
            "area": area,
            "perimeter": peri,
            "bbox": tuple(int(v) for v in cv2.boundingRect(c)),
        })
    shapes.sort(key=lambda s: s["area"], reverse=True)
    return shapes


def draw(frame, s):
    color = COLORS.get(s["name"], COLORS["Unknown"])
    cv2.drawContours(frame, [s["contour"]], -1, color, 2)
    cv2.polylines(frame, [s["approx"]], False, color, 2)
    x, y, w, h = s["bbox"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    label = f"{s['name']}  Area: {s['area']:.0f} px  Perim: {s['perimeter']:.0f} px"
    cv2.putText(frame, label, (max(x, 2), max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def process(img, min_area=MIN_CONTOUR_AREA):
    shapes = find_shapes(img, min_area)
    annotated = img.copy()
    for s in shapes:
        draw(annotated, s)
    return annotated, shapes


def run_image(path, min_area=MIN_CONTOUR_AREA, output=None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    annotated, shapes = process(img, min_area)
    for s in shapes:
        x, y, w, h = s["bbox"]
        print(f"[{s['name']}] vertices={s['vertices']} area={s['area']:.0f} "
              f"perimeter={s['perimeter']:.0f} bbox=({x},{y},{w},{h})")
    if output:
        cv2.imwrite(output, annotated)
    cv2.imshow("Shape Detector", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(camera_index=0, min_area=MIN_CONTOUR_AREA, window_name="Shape Detector"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        annotated, _ = process(frame, min_area)
        cv2.imshow(window_name, annotated)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def generate_demo_image():
    img = np.full((480, 640, 3), 255, dtype=np.uint8)

    def fill(points, color):
        cv2.fillPoly(img, [np.array(points, dtype=np.int32).reshape(-1, 1, 2)], color)

    def outline(points, color, thickness=2):
        cv2.polylines(img, [np.array(points, dtype=np.int32).reshape(-1, 1, 2)], False, color, thickness)

    tri = [[60, 60], [160, 60], [110, 150]]
    fill(tri, (0, 0, 255)); outline(tri, (0, 0, 255))

    sq = [[240, 60], [340, 60], [340, 160], [240, 160]]
    fill(sq, (0, 165, 255)); outline(sq, (0, 165, 255))

    rect = [[430, 60], [560, 60], [560, 180], [430, 180]]
    fill(rect, (0, 255, 0)); outline(rect, (0, 255, 0))

    pent = [[180, 240], [300, 240], [350, 310], [230, 350], [150, 310]]
    fill(pent, (255, 0, 255)); outline(pent, (255, 0, 255))

    hexg = [[450, 240], [570, 240], [620, 310], [570, 380], [450, 380], [400, 310]]
    fill(hexg, (255, 255, 0)); outline(hexg, (255, 255, 0))

    poly = [[60, 430], [180, 430], [180, 340], [120, 340], [120, 250], [60, 250]]
    fill(poly, (160, 160, 160)); outline(poly, (160, 160, 160))

    noise = [[560, 440], [585, 440], [585, 465], [560, 465]]
    fill(noise, (0, 0, 0)); outline(noise, (0, 0, 0))

    return img


def main():
    parser = argparse.ArgumentParser(description="Detect geometric shapes in images or live video.")
    parser.add_argument("--image", type=str, default=None, help="Path to the input image.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live video (0 = default).")
    parser.add_argument("--min-area", type=float, default=MIN_CONTOUR_AREA, help="Minimum contour area in px to accept a shape.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the annotated image.")
    parser.add_argument("--demo", action="store_true", help="Generate demo shapes, detect, save demo_result.png.")
    args = parser.parse_args()

    if args.demo:
        annotated, shapes = process(generate_demo_image(), min_area=args.min_area)
        for s in shapes:
            x, y, w, h = s["bbox"]
            print(f"[{s['name']}] vertices={s['vertices']} area={s['area']:.0f} perimeter={s['perimeter']:.0f} bbox=({x},{y},{w},{h})")
        cv2.imwrite("demo_result.png", annotated)
        print("Demo result saved to demo_result.png")
        return

    if args.image:
        run_image(args.image, min_area=args.min_area, output=args.output)
    else:
        run_camera(args.camera, min_area=args.min_area)


if __name__ == "__main__":
    main()
