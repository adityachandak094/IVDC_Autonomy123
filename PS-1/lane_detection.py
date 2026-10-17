"""
IVDC Autonomy Induction 2026 - PS 1, Part B
Lane Marking and Background Isolation

Input : input/road.jpg            (the road photo from the problem statement)
Output: output/final_bitmap.png   (lane region BLACK, everything else WHITE)
        output/01_... to 09_...   (intermediate images of every pipeline stage)

Pipeline
    1. Preprocessing      - Gaussian blur, grayscale, HSV
    2. Thresholding       - road mask (low saturation) + white-marking mask (top-hat)
    3. Contour filtering  - keep only long, thin bright blobs as lane markings
    4. Edge detection     - Canny
    5. Region of Interest - keep only the part of the image where the road can be
    6. Lane detection     - Hough transform -> line segments -> left / right boundary
    7. Boundary fitting   - fit a smooth curve x = f(y) through each boundary
    8. Lane region        - fill the area between the two boundaries, find its centre
    9. Final bitmap       - black lane on a white background

Which "lane" is this?
    The camera sits left of the solid white line, so the vehicle's own lane (the
    ego lane) runs from the LEFT ROAD EDGE (where asphalt meets grass - there is no
    painted line there) to the WHITE LINE. That region is painted black.

Run:  python lane_detection.py            (saves all images)
      python lane_detection.py --show     (also opens a window with the overview)

Only OpenCV, NumPy and Matplotlib are used. No pretrained models.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Folder this script lives in, so paths work no matter where you run it from
# (for example when VS Code runs it from the repository's top folder).
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Settings. These are the numbers you can experiment with.
# ---------------------------------------------------------------------------
BLUR_KERNEL = (5, 5)        # Gaussian blur size; bigger = smoother, but blurrier edges
CANNY_LOW, CANNY_HIGH = 50, 150   # Canny hysteresis thresholds

ROAD_MAX_SATURATION = 110   # asphalt is grey (saturation ~60); grass is ~190
MARKING_TOPHAT_MIN = 25     # how much brighter than its surroundings a marking must be
MARKING_MAX_SATURATION = 80 # lane paint is white, so it has low saturation
TOPHAT_KERNEL = 21          # must be wider than the painted line (in pixels)

MIN_MARKING_LENGTH = 40     # contour must be at least this long (pixels) ...
MIN_MARKING_ELONGATION = 4  # ... and this many times longer than it is wide

HORIZON_RATIO = 0.46        # rows above 46 % of the height are sky -> ignored

HOUGH_THRESHOLD = 20        # votes needed for a line
HOUGH_MIN_LENGTH = 20       # shortest segment to keep (pixels)
HOUGH_MAX_GAP = 15          # join segments separated by gaps up to this size

# Slope = dy/dx in image coordinates (y grows DOWNWARDS).
#  - The left road edge rises to the right   -> negative slope.
#  - The white line rises slightly to the left -> large positive slope.
LEFT_SLOPE_RANGE = (-2.0, -0.2)
RIGHT_MIN_SLOPE = 1.0
CURVE_MIN_SPAN = 150        # fit a curve (degree 2) only if points span this many rows

# Used ONLY if one boundary cannot be found: the assumed lane width at the bottom
# row, as a fraction of the image width. A forward camera's own lane usually
# fills about the whole width of the bottom of the picture.
LANE_WIDTH_AT_BOTTOM = 1.0


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------
def load_image(path):
    """Read an image from disk and stop with a clear message if that fails."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: input image not found: {path}\n"
                 f"Put the road photo at PS1_Computer_Vision/input/road.jpg "
                 f"or pass another file with --input.")
    img = cv2.imread(str(path))           # OpenCV loads colour images in BGR order
    if img is None:
        sys.exit(f"ERROR: {path} exists but could not be read as an image.")
    return img


def preprocess(img):
    """Blur to remove small noise, then make grayscale and HSV versions."""
    blurred = cv2.GaussianBlur(img, BLUR_KERNEL, 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return blurred, gray, hsv


# ---------------------------------------------------------------------------
# 2. Thresholding
# ---------------------------------------------------------------------------
def road_mask_from_hsv(hsv):
    """
    Asphalt is grey, so its colour saturation (S in HSV) is low, while grass,
    crops and the blue sky are strongly coloured. Saturation barely changes when
    the scene gets brighter or darker, which makes this robust to lighting.
    """
    h, w = hsv.shape[:2]
    saturation = hsv[:, :, 1]
    mask = np.where(saturation < ROAD_MAX_SATURATION, 255, 0).astype(np.uint8)
    mask[: int(h * HORIZON_RATIO)] = 0        # nothing above the horizon is road

    # Morphology: "opening" removes small white specks, "closing" fills small holes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # Keep only the largest connected white region: the road itself.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask


def marking_mask_from_hsv(hsv):
    """
    Painted lines are thin and brighter than the road right next to them.
    A "top-hat" filter keeps exactly that: brightness minus a smoothed version
    of itself. A fixed threshold such as V > 215 failed when the image was made
    darker or brighter; the top-hat compares each pixel with its neighbourhood
    instead, so it keeps working.
    """
    h, w = hsv.shape[:2]
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    tophat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    mask = (tophat > MARKING_TOPHAT_MIN) & (saturation < MARKING_MAX_SATURATION)
    mask = mask.astype(np.uint8) * 255
    mask[: int(h * HORIZON_RATIO)] = 0
    return mask


# ---------------------------------------------------------------------------
# 3. Contour filtering
# ---------------------------------------------------------------------------
def keep_line_shaped_contours(mask):
    """
    Find every white blob (contour) and keep only the long, thin ones, because
    a lane marking is a stripe, not a patch. minAreaRect gives the smallest
    rotated rectangle around the blob, so we can measure its length and width.
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = np.zeros_like(mask)
    for contour in contours:
        _, (side_a, side_b), _ = cv2.minAreaRect(contour)
        length = max(side_a, side_b)
        width = max(min(side_a, side_b), 1.0)
        x, _, bw, _ = cv2.boundingRect(contour)
        touches_side = x <= 1 or x + bw >= w - 1      # filter artefacts at the image border
        if length >= MIN_MARKING_LENGTH and length / width >= MIN_MARKING_ELONGATION \
                and not touches_side:
            cv2.drawContours(kept, [contour], -1, 255, thickness=cv2.FILLED)
    return kept


# ---------------------------------------------------------------------------
# 5. Region of Interest
# ---------------------------------------------------------------------------
def region_of_interest(shape):
    """
    A polygon covering the lower part of the image, where the road can appear.
    Everything outside (sky, distant fields) is ignored. Corners are given as
    fractions of the image size, so it also works for other image sizes.
    """
    h, w = shape[:2]
    corners = np.array([[
        (0, h),
        (0, int(h * 0.70)),
        (int(w * 0.50), int(h * HORIZON_RATIO)),
        (int(w * 0.80), int(h * HORIZON_RATIO)),
        (w, int(h * 0.62)),
        (w, h),
    ]], dtype=np.int32)
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi, corners, 255)
    return roi, corners


# ---------------------------------------------------------------------------
# 6. Lane detection with the Hough transform
# ---------------------------------------------------------------------------
def hough_segments(edges):
    """Return line segments as an (N, 4) array of x1, y1, x2, y2 (or empty)."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LENGTH, maxLineGap=HOUGH_MAX_GAP)
    if lines is None:
        return np.empty((0, 4), dtype=int)
    return lines.reshape(-1, 4)   # works for OpenCV 4 (N,1,4) and OpenCV 5 (N,4)


def slope_of(segment):
    x1, y1, x2, y2 = segment
    if x2 == x1:
        return np.inf               # perfectly vertical segment
    return (y2 - y1) / (x2 - x1)


def split_left_right(border_segments, marking_segments, image_width):
    """
    Left boundary  = road/grass border segments with a negative slope.
    Right boundary = lane-marking segments with a steep positive slope.
    Segments lying on the image's left/right edge are dropped: they come from
    the road running out of the picture, not from a real boundary.
    """
    left, right = [], []
    for seg in border_segments:
        x1, _, x2, _ = seg
        if min(x1, x2) <= 2 or max(x1, x2) >= image_width - 3:
            continue
        if LEFT_SLOPE_RANGE[0] <= slope_of(seg) <= LEFT_SLOPE_RANGE[1]:
            left.append(seg)
    for seg in marking_segments:
        if slope_of(seg) >= RIGHT_MIN_SLOPE:
            right.append(seg)
    left, right = np.array(left).reshape(-1, 4), np.array(right).reshape(-1, 4)

    # A left-boundary segment must lie to the left of the white line.
    if len(left) and len(right):
        right_x_min = right[:, [0, 2]].min()
        left = left[left[:, [0, 2]].max(axis=1) < right_x_min]
    return left, right


# ---------------------------------------------------------------------------
# 7. Boundary fitting
# ---------------------------------------------------------------------------
def fit_boundary(segments):
    """
    Fit x = f(y) through the points of all segments on one side.
    We fit x as a function of y (not y of x) because lane lines are close to
    vertical in the image. If the points cover enough rows, a degree-2 curve
    follows the bend of the road; otherwise a straight line is safer.
    Returns (polynomial coefficients, lowest y with data) or None.
    """
    if len(segments) == 0:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        n = max(abs(int(y2) - int(y1)), abs(int(x2) - int(x1)), 2)
        xs.extend(np.linspace(x1, x2, n))
        ys.extend(np.linspace(y1, y2, n))
    xs, ys = np.array(xs), np.array(ys)
    degree = 2 if ys.max() - ys.min() >= CURVE_MIN_SPAN else 1
    coeffs = np.polyfit(ys, xs, degree)
    return coeffs, int(ys.min())


def assume_missing_boundary(fit, missing_side, height, width):
    """
    Bonus - missing boundary. ASSUMPTION: the lane has a constant real-world width.
    Because of perspective (Gate 5), a constant width looks narrower further
    away and shrinks to zero at the horizon. So in the image the lane width
    grows in a straight line from 0 at the horizon row to
    LANE_WIDTH_AT_BOTTOM x image width at the bottom row:

        width(y) = k * (y - y_horizon)

    and the missing boundary is the found one shifted sideways by width(y).
    """
    coeffs, top = fit
    y_horizon = height * HORIZON_RATIO
    k = LANE_WIDTH_AT_BOTTOM * width / (height - y_horizon)
    sign = 1 if missing_side == "right" else -1         # right side lies at larger x
    shift = np.array([sign * k, -sign * k * y_horizon])  # the line k*(y - y_horizon)
    return np.polyadd(coeffs, shift), top


# ---------------------------------------------------------------------------
# 8. Lane region and centre
# ---------------------------------------------------------------------------
def lane_polygon(left_fit, right_fit, height, width):
    """Walk from the bottom row upwards, collecting left and right x positions."""
    top = max(left_fit[1], right_fit[1])     # only as high as both sides have data
    ys = np.arange(height - 1, top - 1, -1)
    xl = np.polyval(left_fit[0], ys)
    xr = np.polyval(right_fit[0], ys)
    valid = xl < xr                          # stop if the boundaries ever cross
    ys, xl, xr = ys[valid], xl[valid], xr[valid]
    # The lane centre uses the TRUE boundary positions. Near the bottom the left
    # road edge lies outside the picture (negative x), and clipping it first
    # would bend the centre line.
    centre_x = np.clip((xl + xr) / 2, 0, width - 1)
    centre = np.column_stack([centre_x, ys]).astype(np.int32)
    # The polygon itself must stay inside the image.
    xl = np.clip(xl, 0, width - 1)
    xr = np.clip(xr, 0, width - 1)
    left_pts = np.column_stack([xl, ys])
    right_pts = np.column_stack([xr, ys])[::-1]
    polygon = np.vstack([left_pts, right_pts]).astype(np.int32)
    return polygon, centre


# ---------------------------------------------------------------------------
# The full pipeline
# ---------------------------------------------------------------------------
def detect_lane(img, missing=None):
    """
    Run every stage on one BGR image.
    `missing` ("left" or "right") deliberately throws one boundary away; it is
    only used by bonus_experiments.py to test the missing-boundary fallback.
    Returns a dictionary holding every intermediate result.
    """
    h, w = img.shape[:2]
    out = {"input": img}

    blurred, gray, hsv = preprocess(img)
    out.update(blurred=blurred, gray=gray, hsv=hsv)

    road = road_mask_from_hsv(hsv)
    marking_raw = marking_mask_from_hsv(hsv)
    marking = keep_line_shaped_contours(marking_raw)
    out.update(road_mask=road, marking_raw=marking_raw, marking=marking)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    out["edges"] = edges

    roi, roi_corners = region_of_interest(img.shape)
    out.update(roi=roi, roi_corners=roi_corners)

    # Edges of the painted line: Canny edges that lie on (or touch) the marking.
    near_marking = cv2.dilate(marking, np.ones((5, 5), np.uint8))
    marking_edges = cv2.bitwise_and(edges, cv2.bitwise_and(near_marking, roi))
    # Edges of the road itself: the outline of the road mask (asphalt vs grass).
    border_edges = cv2.bitwise_and(cv2.Canny(road, CANNY_LOW, CANNY_HIGH), roi)
    out.update(marking_edges=marking_edges, border_edges=border_edges,
               roi_edges=cv2.bitwise_or(marking_edges, border_edges))

    left_segs, right_segs = split_left_right(hough_segments(border_edges),
                                             hough_segments(marking_edges), w)
    if missing == "left":
        left_segs = left_segs[:0]
    if missing == "right":
        right_segs = right_segs[:0]
    out.update(left_segments=left_segs, right_segments=right_segs)

    left_fit, right_fit = fit_boundary(left_segs), fit_boundary(right_segs)
    assumed = None
    if left_fit is None and right_fit is None:
        raise RuntimeError("No lane boundary found on either side - "
                           "try changing the thresholds at the top of the file.")
    if left_fit is None:
        left_fit, assumed = assume_missing_boundary(right_fit, "left", h, w), "left"
    elif right_fit is None:
        right_fit, assumed = assume_missing_boundary(left_fit, "right", h, w), "right"
    out.update(left_fit=left_fit, right_fit=right_fit, assumed_side=assumed)

    polygon, centre = lane_polygon(left_fit, right_fit, h, w)
    out.update(polygon=polygon, centre=centre)

    # Final binary bitmap: start all white (255), paint the lane black (0).
    bitmap = np.full((h, w), 255, dtype=np.uint8)
    cv2.fillPoly(bitmap, [polygon], 0)
    out["bitmap"] = bitmap

    # How far the camera (image centre) is from the lane centre, at the bottom row.
    out["centre_offset_px"] = int(w // 2 - centre[0, 0])
    return out


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_overlay(result):
    """Original photo + green lane area + boundaries + yellow centre line."""
    img = result["input"].copy()
    tint = img.copy()
    cv2.fillPoly(tint, [result["polygon"]], (0, 200, 0))
    overlay = cv2.addWeighted(tint, 0.35, img, 0.65, 0)

    h = img.shape[0]
    top = result["polygon"][:, 1].min()
    ys = np.arange(top, h)
    for fit, colour, side in [(result["left_fit"], (255, 80, 0), "left"),
                              (result["right_fit"], (0, 0, 255), "right")]:
        pts = np.column_stack([np.polyval(fit[0], ys), ys]).astype(np.int32)
        # An assumed boundary is drawn in magenta, so it is easy to tell apart.
        if result["assumed_side"] == side:
            colour = (255, 0, 255)
        cv2.polylines(overlay, [pts], False, colour, 4)
    cv2.polylines(overlay, [result["centre"]], False, (0, 255, 255), 2)
    return overlay


def draw_segments(result):
    """Raw Hough segments: blue = left boundary, red = right boundary."""
    img = result["input"].copy()
    for segs, colour in [(result["left_segments"], (255, 80, 0)),
                         (result["right_segments"], (0, 0, 255))]:
        for x1, y1, x2, y2 in segs:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
    return img


def draw_roi(result):
    """The edge image with the ROI outline drawn on it in green."""
    vis = cv2.cvtColor(result["edges"], cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, result["roi_corners"], True, (0, 255, 0), 2)
    return vis


def save_outputs(result, out_dir):
    """Write every stage to disk with numbered names, in pipeline order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "01_gray.png": result["gray"],
        "02_blur.png": result["blurred"],
        "03_road_mask_hsv.png": result["road_mask"],
        "04_marking_mask_contours.png": result["marking"],
        "05_canny_edges.png": result["edges"],
        "06_roi_on_edges.png": draw_roi(result),
        "07_roi_edges.png": result["roi_edges"],
        "08_hough_segments.png": draw_segments(result),
        "09_lane_overlay.png": draw_overlay(result),
        "final_bitmap.png": result["bitmap"],   # PNG = lossless, stays exactly 0/255
    }
    for name, image in images.items():
        cv2.imwrite(str(out_dir / name), image)
    return images


def save_overview(result, path, show=False):
    """One figure with every stage side by side, for the README."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")       # draw without opening a window
    import matplotlib.pyplot as plt

    def rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image

    panels = [
        ("1. Input", result["input"]),
        ("2. Grayscale", result["gray"]),
        ("3. Road mask (HSV saturation)", result["road_mask"]),
        ("4. Marking (top-hat + contours)", result["marking"]),
        ("5. Canny edges", result["edges"]),
        ("6. Edges inside ROI", result["roi_edges"]),
        ("7. Hough segments", draw_segments(result)),
        ("8. Lane + centre line", draw_overlay(result)),
        ("9. Final bitmap", result["bitmap"]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    for ax, (title, image) in zip(axes.ravel(), panels):
        ax.imshow(rgb(image), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("PS 1 lane detection pipeline", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")   # save BEFORE show
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="PS 1 lane detection")
    parser.add_argument("--input", default=str(HERE / "input" / "road.jpg"),
                        help="road image to process")
    parser.add_argument("--output", default=str(HERE / "output"),
                        help="folder for the result images")
    parser.add_argument("--show", action="store_true",
                        help="open a window with the pipeline overview")
    args = parser.parse_args()

    img = load_image(args.input)
    result = detect_lane(img)
    out_dir = Path(args.output)
    save_outputs(result, out_dir)
    save_overview(result, out_dir / "00_pipeline_overview.png", show=args.show)

    print(f"Image size            : {img.shape[1]} x {img.shape[0]}")
    print(f"Left boundary segments: {len(result['left_segments'])}")
    print(f"Right boundary segments: {len(result['right_segments'])}")
    if result["assumed_side"]:
        print(f"NOTE: {result['assumed_side']} boundary not found - it was assumed.")
    black = int((result["bitmap"] == 0).sum())
    print(f"Lane pixels (black)   : {black} ({100 * black / result['bitmap'].size:.1f} % of image)")
    print(f"Camera offset from lane centre (bottom row): {result['centre_offset_px']} px "
          f"({'right' if result['centre_offset_px'] > 0 else 'left'} of centre)")
    print(f"Saved results to      : {out_dir}")


if __name__ == "__main__":
    main()
