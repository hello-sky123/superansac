# superansac — build and verification notes

## Python environment

Build and verify with **ransac_env** (Python 3.10). Do not use `/usr/bin/python3.8`.

```bash
PY=/home/zhangzhongping/anaconda3/envs/ransac_env/bin/python3.10
PB=$($PY -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake -B <builddir> -S . -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$PY -DPython_EXECUTABLE=$PY -DPython3_EXECUTABLE=$PY \
  -Dpybind11_DIR=$PB
cmake --build <builddir> --target pysuperansac -j$(nproc)

PYTHONPATH=<builddir>/python $PY script.py
```

Pass all three `*_EXECUTABLE` variables. CMake otherwise picks its own
interpreter, and the module then carries an ABI tag the running interpreter
cannot load.

**Why this matters.** An ABI-mismatched module does not raise an error. Python
falls back to whatever `pysuperansac` is installed in site-packages, so two
builds silently execute the same third binary — A/B comparisons come back
"almost identical" and read as *no regression* when nothing was measured at
all. This invalidated a full sweep once.

Guard against it: check the module path resolves into the build directory,

```bash
PYTHONPATH=<builddir>/python $PY -c "import pysuperansac as p; print(p.__file__)"
```

and include a **control** in every A/B comparison — some quantity the change
must move. If the control does not move, the measurement is void, whatever the
other numbers say.

## Measuring performance

Wall-clock time is unusable on this machine: `powersave` governor, turbo on
(2500–4451 MHz observed), and a standing load average of 8–12 on 16 threads.
Repeated best-of-7 medians contradicted each other by 75%.

Use instruction counts instead — they are load- and frequency-independent
(~0.1% run-to-run):

```bash
taskset -c 6 perf stat -x, -e instructions,cycles <cmd>
```

Two requirements for a meaningful number:

- **Pin the work.** Set `min_iterations == max_iterations` and `confidence = 1.0`,
  or confidence-based termination lets the two builds do different amounts of
  work and the comparison means nothing.
- **Subtract the interpreter baseline.** Import, numpy, and data construction
  cost ~409M instructions — about half of a typical measurement. Run the same
  script with the estimation call removed and subtract, or effects come out
  diluted roughly twofold.

## Verification practice

- Compare geometric error against ground truth, not scores. Five-point scores in
  particular are sensitive to heap alignment (see the note on
  `solver_essential_matrix_five_point_nister.h`) and are not a valid baseline
  across builds or processes.
- Use enough seeds. A 10-seed essential-matrix run showed a 35% "regression"
  that disappeared at 40 seeds.
- Watch for saturated benchmarks. Precision 1.0000 and recall 0.999 in both arms
  means the test cannot discriminate; raise noise or the outlier ratio.

## API conventions worth knowing

Return shapes differ between estimators:

| function | returns |
|---|---|
| `estimateHomography` / `estimateFundamentalMatrix` | `(M, inliers, score, iters)` |
| `estimateEssentialMatrix` | `(E, inliers, score, iters)` |
| `estimateRigidTransform` | `(T, inliers, score, iters)` |
| `estimateAbsolutePose` | `(R, t, inliers, score, iters)` — R and t separate |

- `estimateRigidTransform`'s `T` is row-major/transposed: the map is
  `Y = X @ T[:3,:3] + T[3,:3]`, not `R @ X + t`.
- `estimateAbsolutePose` takes a 5-element bounding box
  `(image width, image height, X, Y, Z)` and builds a 5-D neighborhood over all
  five data columns, so every column must fall inside it.
- Grid neighborhoods reject negative coordinates and coordinates beyond the
  bounding box. Offset synthetic 3D data to be positive.
