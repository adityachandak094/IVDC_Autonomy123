"""
IVDC Autonomy Induction 2026 - PS 1, Part B, BONUS extensions

Uses the pipeline in lane_detection.py (it must be in the same folder) for the
three bonus tasks in the problem statement:

  Bonus 1 - Test the pipeline under changes in lighting or mild clutter
            -> output/bonus1_robustness.png, printed IoU score per test
  Bonus 2 - Handle a missing lane boundary using a reasonable assumption
            -> output/bonus2_missing_boundary.png
  Bonus 3 - Produce a simple drivable-area mask or cost representation
            -> output/bonus3_drivable_area.png, bonus3_cost_map.png

Run:  python bonus_experiments.py
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")               # save figures without opening windows
import matplotlib.pyplot as plt

import lane_detection as ld

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def iou(mask_a, mask_b):
    """
    Intersection over Union of two lane regions (black = lane).
    1.0 = identical lane, 0.0 = no overlap. A standard way to score masks.
    """
    a, b = mask_a == 0, mask_b == 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


# ---------------------------------------------------------------------------
# Bonus 1: lighting and clutter
# ---------------------------------------------------------------------------
def make_variants(img, seed=0):
    """Create harder versions of the same photo."""
    rng = np.random.default_rng(seed)        # fixed seed -> same result every run
    h, w = img.shape[:2]
    variants = {}

    variants["Darker (x0.55)"] = cv2.convertScaleAbs(img, alpha=0.55, beta=0)
    variants["Brighter (+45)"] = cv2.convertScaleAbs(img, alpha=1.0, beta=45)
    variants["Low contrast"] = cv2.convertScaleAbs(img, alpha=0.6, beta=60)

    noise = rng.normal(0, 12, img.shape)     # random sensor-like noise
    variants["Sensor noise"] = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Tree shadows: dark bands across the road.
    shadow = img.copy().astype(np.float32)
    for x0 in (60, 200):
        band = np.array([[x0, h], [x0 + 70, h], [x0 + 170, int(h * 0.55)],
                         [x0 + 130, int(h * 0.55)]], np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [band], 1)
        shadow[m == 1] *= 0.5
    variants["Shadows"] = shadow.astype(np.uint8)

    # Clutter: random light/dark patches and a few leaves on the road.
    clutter = img.copy()
    for _ in range(25):
        cx = int(rng.integers(20, w - 20))
        cy = int(rng.integers(int(h * 0.6), h - 10))
        colour = [int(v) for v in rng.integers(40, 200, 3)]
        cv2.circle(clutter, (cx, cy), int(rng.integers(3, 9)), colour, -1)
    for _ in range(4):
        x, y = int(rng.integers(40, 280)), int(rng.integers(int(h * 0.65), h - 30))
        cv2.rectangle(clutter, (x, y), (x + 30, y + 12), (95, 100, 105), -1)  # tar patch
    variants["Clutter"] = clutter
    return variants


def bonus1_robustness(img, reference):
    variants = make_variants(img)
    fig, axes = plt.subplots(len(variants), 2, figsize=(8.5, 3.2 * len(variants)))
    print("\nBonus 1 - robustness (IoU against the original result, 1.0 = identical)")
    for row, (name, variant) in enumerate(variants.items()):
        try:
            res = ld.detect_lane(variant)
            score = iou(res["bitmap"], reference["bitmap"])
            overlay, bitmap = ld.draw_overlay(res), res["bitmap"]
            note = f" (assumed {res['assumed_side']} boundary)" if res["assumed_side"] else ""
            label = f"{name}: IoU = {score:.2f}{note}"
        except RuntimeError as err:          # pipeline found nothing at all
            overlay, bitmap = variant, np.full(variant.shape[:2], 255, np.uint8)
            label = f"{name}: FAILED ({err})"
        print("  " + label)
        axes[row, 0].imshow(rgb(overlay))
        axes[row, 0].set_title(label, fontsize=10)
        axes[row, 1].imshow(bitmap, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("bitmap", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "bonus1_robustness.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bonus 2: missing boundary
# ---------------------------------------------------------------------------
def bonus2_missing_boundary(img, reference):
    print("\nBonus 2 - missing boundary (assume a constant real lane width, "
          "which shrinks to zero at the horizon in the image)")
    panels = [("Both boundaries detected", reference)]
    for side in ("left", "right"):
        res = ld.detect_lane(img, missing=side)
        score = iou(res["bitmap"], reference["bitmap"])
        print(f"  {side} boundary removed -> assumed from the other side, IoU = {score:.2f}")
        panels.append((f"{side.capitalize()} boundary removed, assumed (magenta)\n"
                       f"IoU vs. detected = {score:.2f}", res))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (title, res) in zip(axes, panels):
        ax.imshow(rgb(ld.draw_overlay(res)))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "bonus2_missing_boundary.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bonus 3: drivable area and cost map
# ---------------------------------------------------------------------------
def bonus3_drivable_area(img, reference):
    """
    Drivable area = all asphalt (both lanes), from the HSV road mask.
    Cost map = how "expensive" it is for a planner to put the vehicle there:
      0.0 on the lane centre, rising to 0.5 at the lane edges,
      0.7 on the other lane (drivable, but not where we should be),
      1.0 off the road (grass, sky: never drive there).
    A planner such as A* in PS 2 could use this as cell costs.
    """
    h, w = img.shape[:2]
    road = reference["road_mask"]
    drivable = np.full((h, w), 255, np.uint8)
    drivable[road > 0] = 0                      # black = drivable, like the main bitmap
    cv2.imwrite(str(OUT / "bonus3_drivable_area.png"), drivable)

    cost = np.ones((h, w), np.float32)          # default: off-road
    cost[road > 0] = 0.7                        # on asphalt
    lane = reference["bitmap"] == 0
    centre_x = np.full(h, np.nan)
    half_width = np.full(h, np.nan)
    ys = reference["centre"][:, 1]
    centre_x[ys] = reference["centre"][:, 0]
    xl = np.polyval(reference["left_fit"][0], ys)
    xr = np.polyval(reference["right_fit"][0], ys)
    half_width[ys] = (xr - xl) / 2
    cols = np.arange(w)[None, :]
    with np.errstate(invalid="ignore"):
        dist = np.abs(cols - centre_x[:, None]) / half_width[:, None]
    lane_cost = np.clip(0.5 * dist, 0, 0.5)
    cost[lane] = lane_cost[lane]
    np.save(OUT / "bonus3_cost_map.npy", cost)  # raw numbers, for use in code

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].imshow(drivable, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Drivable area (black = asphalt, both lanes)")
    im = axes[1].imshow(cost, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Cost map (0 = lane centre, 1 = off-road)")
    fig.colorbar(im, ax=axes[1], fraction=0.035)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "bonus3_cost_map.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    share = 100 * (road > 0).sum() / road.size
    print(f"\nBonus 3 - drivable area covers {share:.1f} % of the image; cost map saved")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img = ld.load_image(HERE / "input" / "road.jpg")
    reference = ld.detect_lane(img)
    bonus1_robustness(img, reference)
    bonus2_missing_boundary(img, reference)
    bonus3_drivable_area(img, reference)
    print(f"\nSaved bonus results to {OUT}")


if __name__ == "__main__":
    main()
