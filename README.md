# IVDC Autonomy Induction 2026 — Aditya Chandak

Submission for the IVDC Autonomy 2026 induction: Computer Vision (PS 1), Motion Planning (PS 2) and Controllers / RL (PS 3).

## Repository structure

```
IVDC_Autonomy123/
├── README.md
├── requirements.txt
├── PS-1/
│   ├── PS-1 PART-A.pdf           # Part A answers + references
│   ├── lane_detection.py         # Part B code
│   ├── bonus_experiments.py
│   ├── input/road.jpg            # provided test image
│   └── output/                   # intermediate + final images
├── PS-2/
│   ├── PS 2-PART-A.pdf
│   ├── planners.py               # BFS, Dijkstra, A*
│   ├── path_planner.py           # map + visualization
│   ├── bonus_experiments.py
│   ├── test_planners.py
│   └── output/                   # plots and animation
└── PS-3/
    ├── PS 3-PART-A.pdf
    └── deepml_frozenlake_completed.png   # DeepML screenshot proof
```

## Dependencies

- Python 3.10 or newer
- OpenCV (`opencv-python`), NumPy, Matplotlib

## Setup

```bash
git clone https://github.com/adityachandak094/IVDC_Autonomy123.git
cd IVDC_Autonomy123
pip install -r requirements.txt
```

---

## PS 1 — Lane Marking and Background Isolation

Full details, all intermediate images and the requirements checklist are in [`PS-1/README.md`](PS-1/README.md).

### Approach
**Blur → grayscale + HSV → road mask (low saturation) + lane-marking mask (top-hat + contour shape) → Canny → ROI → Hough segments → split left/right by slope → fit x = f(y) → fill lane polygon + centre line → bitmap.**

The ego lane runs from the left road edge, which has no painted line and is found where grey asphalt meets coloured grass, to the white line.

### How to run
```bash
python PS-1/lane_detection.py        # main task
python PS-1/bonus_experiments.py     # bonus tasks
```

### Input / Output
- **Input:** `PS-1/input/road.jpg` (the photo from the problem statement)
- **Output:** `PS-1/output/final_bitmap.png` (lane **black**, surroundings **white**, values exactly 0/255), plus 9 intermediate images and the bonus results

### Results
![PS1 pipeline](PS-1/output/00_pipeline_overview.png)

### Bonus results
- **Lighting / clutter:** IoU 0.89–0.99 across six harder versions of the image (darker, brighter, low contrast, noise, shadows, clutter).
- **Missing boundary:** each side removed in turn and assumed from a constant real lane width → IoU 0.97 (left removed) and 0.90 (right removed).
- **Drivable area + cost map:** a mask of all asphalt, and a 0–1 cost map a planner could use.

### Limitations
The lane stops before the far bend because the curved top of the white line fails the slope rule. The settings are tuned to this one scene (grey road between coloured fields).

---

## PS 2 — Grid-Based Path Planning Visualization

Full details, the testing write-up and the requirements checklist are in [`PS-2/README.md`](PS-2/README.md).

### Approach
- **Map:** a 20 × 20 NumPy array (`0` = free, `1` = obstacle), 4-connected moves, start (1, 1), goal (18, 18).
- **Algorithms, written from scratch:** BFS (FIFO queue), Dijkstra (priority queue on cost so far) and A\* (priority queue on cost + Manhattan heuristic). Each keeps a frontier, records `came_from`, and rebuilds the path by walking back from the goal.

### How to run
```bash
python PS-2/path_planner.py         # main task + animation
python PS-2/bonus_experiments.py    # bonus tasks
python PS-2/test_planners.py        # 20 correctness checks
```

### Input / Output
- **Input:** the map, start and goal defined at the top of `PS-2/path_planner.py`
- **Output:** plots in `PS-2/output/` showing free cells, obstacles, start, goal, explored cells and the final path, plus an animation of A\* searching

### Results
![Path planning comparison](PS-2/output/00_comparison.png)

| Algorithm | Cells explored | Path length |
| --- | --- | --- |
| BFS | 339 | 46 |
| Dijkstra | 339 | 46 |
| A\* | 253 | 46 |

### Observations
- All three find the true shortest path (46 steps). A\* explores 25 % fewer cells, less than it could, because a dead end that looks close to the goal misleads its heuristic.
- **High-cost cells (bonus):** BFS goes straight through the mud (17 steps, cost 33). Dijkstra and A\* go round it (27 steps, cost 27).
- **Re-planning (bonus):** after a new obstacle blocks the plan, A\* re-plans from the vehicle's current position; the trip ends 4 steps longer.

---

## PS 3 — Controllers / Reinforcement Learning

- **Part A:** answers are in `PS-3/PS 3-PART-A.pdf`.
- **Part B:** the DeepML project "Q-Learning on Frozen-Lake from Scratch", with proof of completion in `PS-3/deepml_frozenlake_completed.png`.

---

## Use of AI and external resources

As required by the induction guidelines:

- The Part B code in `PS-1` and `PS-2` was written with the help of Claude (Anthropic), an AI assistant. Claude also helped prepare study notes for the Part A topics.
- No code was copied from other repositories. Learning resources used are listed under References.

## References

- OpenCV-Python Tutorials — https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- Red Blob Games, Introduction to A* — https://www.redblobgames.com/pathfinding/a-star/introduction.html
- Gabriel Gambetta, Pathfinding Demystified — https://gabrielgambetta.com/generic-search.html
- MathWorks, What Is PID Control? — https://www.mathworks.com/discovery/pid-control.html
- Hugging Face Deep RL Course, Unit 2 — https://huggingface.co/learn/deep-rl-course/unit2/introduction
