# Horn–Schunck Optical Flow (Serial + OpenMP, OpenCV)

## Overview

This project implements the **Horn–Schunck (HS) dense optical flow algorithm** in two variants:

* **Serial version** (single-threaded, Gauss–Seidel updates)
* **Parallel version** using **OpenMP** (Jacobi double buffering)

Given two consecutive grayscale frames, the algorithm estimates a **dense motion field** for each pixel, represented by horizontal (**u**) and vertical (**v**) flow components.
<img width="4094" height="2158" alt="image" src="https://github.com/user-attachments/assets/f487df7a-de73-4235-a9e1-2f95a04da3ec" />

---

## Algorithm (Intuition)

The Horn–Schunck method is based on two assumptions:

1. **Brightness constancy**
   A pixel maintains the same intensity between consecutive frames.

2. **Spatial smoothness**
   Neighboring pixels tend to have similar motion.

These assumptions are combined into an energy minimization problem consisting of:

* A **data term** (uses image derivatives (I_x, I_y, I_t))
* A **smoothness term** (controlled by regularization parameter ( \alpha ))

### Iterative Update Procedure

* Compute spatial derivatives (I_x, I_y) and temporal derivative (I_t)
* Iteratively:

  * Compute local averages of flow vectors (smoothing)
  * Update flow using derivative constraints
* Repeat for a fixed number of iterations

---

## Update Schemes

* **Serial (Gauss–Seidel)**
  Updates values in-place → faster convergence per iteration

* **Parallel (Jacobi)**
  Uses double buffering (`u_next`, `v_next`) → avoids race conditions and ensures deterministic behavior

---

## Requirements

* C++17
* CMake ≥ 3.16
* OpenCV (core, imgproc, highgui)
* OpenMP (for parallel execution)

---

## Build & Run

Ensure you have two grayscale images in the `data/` directory, for example:

```
data/rm1.jpg
data/rm2.jpg
```

Then run:

```bash
chmod +x run.sh
./run.sh
```

### Output

* Console prints execution time (serial vs parallel)
* Flow visualizations saved to:

  * `output/flow_serial.png`
  * `output/flow_parallel.png`
* Display windows showing both results (requires GUI)

---

## Headless Execution

If running on a server without GUI:

* Comment out `imshow()` and `waitKey()` in `main.cpp`
* OR use a virtual display (e.g., Xvfb)

---

## Parameter Tuning

### α (Regularization Parameter)

* Controls smoothness
* Higher → smoother flow
* Lower → more detail but noisier

**Typical range:** `0.1 – 1.5`
**Default:** `0.5`

---

### Iterations

* More iterations → better convergence but slower

**Typical range:** `50 – 500`
**Default:** `100`

---

## Matching Serial & Parallel Results

* Gauss–Seidel (serial) and Jacobi (parallel) produce slightly different outputs
* For identical results:
  → Use **Jacobi update in both implementations**

---

## Performance Tips

* Use all CPU cores:

  ```bash
  export OMP_NUM_THREADS=$(nproc)
  ```
* Build in **Release mode**
* For large images:

  * Consider 8-neighbor averaging
  * Or separable Gaussian smoothing

---

## Project Structure

```
.
├── main.cpp
├── include/
│   ├── opt_flow_serial.hpp
│   └── opt_flow_parallel.hpp
├── src/
│   ├── opt_flow_serial.cpp
│   └── opt_flow_parallel.cpp
├── data/
├── output/
└── run.sh
```

---

## Reference

Horn, B. K. P., & Schunck, B. G. (1981).
**“Determining Optical Flow.”**
*Artificial Intelligence, 17(1–3), 185–203.*

---

## Notes

* This implementation is designed for **educational and performance comparison purposes**
* Demonstrates trade-offs between **sequential and parallel numerical methods**
