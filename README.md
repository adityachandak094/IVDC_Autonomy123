# IVDC Autonomy Induction 2026 — Aditya Chandak

Submission for the IVDC Autonomy 2026 induction: Computer Vision (PS 1), Motion Planning (PS 2) and Controllers / RL (PS 3).



## Repository structure

```
ivdc-autonomy-induction-2026/
├── README.md
├── requirements.txt
├── PS-1
│   ├── PS-1 PART-A.pdf          # Part A answers + references
│   ├── lane\_detection.py         # Part B code
│   ├── input/road.jpg            # provided test image
│   └── output/                   # intermediate + final images
├── PS-2
│   ├── PS 2-PART-A.pdf
│   ├── path\_planner.py
│   └── output/                   # plots of the search
└── PS-3
    ├── PS 3-PART-A.pdf


   Dependencies

* Python 3.10 or newer
* OpenCV (`opencv-python`), NumPy, Matplotlib

## Setup

```bash
git clone https://github.com/adityachandak094/IVDC_AUTONOMY.git
cd IVDC\_AUTONOMY
pip install -r requirements.txt
```

\---

## PS 1 — Lane Marking and Background Isolation

Full details, all intermediate images and the requirements checklist are in [`PS1\_Computer\_Vision/README.md`](PS1_Computer_Vision/README.md).

### Approach

**Blur → grayscale + HSV → road mask (low saturation) + lane-marking mask (top-hat + contour shape) → Canny → ROI → Hough segments → split left/right by slope → fit x = f(y) → fill lane polygon + centre line → bitmap.**

The ego lane runs from the left road edge, which has no painted line and is found where grey asphalt meets coloured grass, to the white line.

### How to run

```bash
python PS-1/lane\_detection.py        # main task
python PS-1/bonus\_experiments.py     # bonus tasks
```

### Input / Output

* **Input:** `PS-1/input/road.jpg` (the photo from the problem statement)
* **Output:** `PS-1/output/final\_bitmap.png` (lane **black**, surroundings **white**, values exactly 0/255), plus 9 intermediate images and the bonus results

### Results

!\[PS1 pipeline](PS-1/output/00\_pipeline\_overview.png)

### Bonus results

* **Lighting / clutter:** IoU 0.89–0.99 across six harder versions of the image (darker, brighter, low contrast, noise, shadows, clutter).
* **Missing boundary:** each side removed in turn and assumed from a constant real lane width → IoU 0.97 (left removed) and 0.90 (right removed).
* **Drivable area + cost map:** a mask of all asphalt, and a 0–1 cost map a planner could use.

### Limitations

The lane stops before the far bend because the curved top of the white line fails the slope rule. The settings are tuned to this one scene (grey road between coloured fields).

\---

## PS 2 — Grid-Based Path Planning Visualization

### Approach

* Map: a \[10×10 / 20×20] NumPy array, where `0` = free and `1` = obstacle.
* Algorithm(s) implemented: \[BFS / Dijkstra / A\* with Manhattan heuristic], written from scratch with no path-planning library.
* \[Explain in 2–3 lines how your search works: frontier, came\_from, path reconstruction.]

### How to run

```bash
cd PS-2
python path\_planner.py
```

### Input / Output

* **Input:** the grid, start and goal defined at the top of `path\_planner.py`
* **Output:** a plot showing free cells, obstacles, start, goal, explored cells and the final path

### Results

!\[Path planning result](PS-2/output/result.png)

|Algorithm|Cells explored|Path length|
|-|-|-|
|\[BFS]|\[ ]|\[ ]|
|\[A\*]|\[ ]|\[ ]|

### Observations and limitations

* \[e.g. A\* explored fewer cells than BFS because the heuristic points it toward the goal.]
* \[Any bonus attempted: weighted cells, re-planning after a new obstacle.]

\---

## PS 3 — Controllers / Reinforcement Learning

* **Part A:** answers are in `PS 3-PART-A.pdf`.

\---

## Use of AI and external resources

As required by the induction guidelines:

* I used Claude  to understand concepts / explain errors / review code. I wrote and tested the final code myself and can explain every part of it.
* 

## References

* OpenCV-Python Tutorials — https://docs.opencv.org/4.x/d6/d00/tutorial\_py\_root.html
* Red Blob Games, Introduction to A\* — https://www.redblobgames.com/pathfinding/a-star/introduction.html
* Hugging Face Deep RL Course, Unit 2 — https://huggingface.co/learn/deep-rl-course/unit2/introduction
* \[Add every other source you actually used.]

