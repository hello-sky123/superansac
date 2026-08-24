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
- A change that touches the sampler cannot be byte-identical — every random draw
  moves. Compare aggregates over **two independent seed ranges per build** and
  treat the baseline-against-itself spread as the noise floor. That floor is wide:
  60 seeds apart, the same binary gave MSAC 0.3494 vs 0.3035 px (13%). One range
  showed MINPRAN 10.8% "worse" until two further ranges put it ahead.

## Run AddressSanitizer periodically

It found three real defects on the ordinary estimation path that no amount of
result-diffing would have surfaced — an out-of-range sampler index, `back()` on an
empty vector, and a `new[]`/`delete` mismatch. Each aborted the run before the
next was reachable, so expect to fix and re-run.

```bash
PY=/home/zhangzhongping/anaconda3/envs/ransac_env/bin/python3.10
cmake -B <dir> -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address" \
  -DPYTHON_EXECUTABLE=$PY -DPython_EXECUTABLE=$PY -DPython3_EXECUTABLE=$PY \
  -Dpybind11_DIR=$($PY -c "import pybind11;print(pybind11.get_cmake_dir())")
cmake --build <dir> --target pysuperansac -j$(nproc)

LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
  ASAN_OPTIONS=detect_leaks=0 PYTHONPATH=<dir>/python $PY script.py
```

`LD_PRELOAD` is required: the sanitizer arrives through a dlopen'd extension, and
without it ASan dies with `Sanitizer CHECK failed: ... (tsd_key_inited) != (0)`
before running anything. The build takes ~10 minutes, so start it in the
background.

Cover all five estimators and every scoring — the first sweep only exercised
homography, and the ACRANSAC bug sat behind two others that had to be fixed first.
Read the region arithmetic in the report rather than guessing at the cause: "8
bytes to the right of a 9616-byte region" pinned the sampler bug to row 300 of a
300-row matrix, and "16 bytes to the left" is the signature of `back()` on an
empty container.

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

## `inlier_threshold` does not mean the same thing to every scoring

Two scorings reinterpret it, which makes their inlier sets look wrong when
compared against the value you passed in. Both are deliberate; details and
measurements are in the class comments.

- **`ScoringType.Grid`** works at `1.5 * inlier_threshold`. Its score counts
  occupied grid cells rather than points, and the margin keeps cells whose points
  sit just outside the nominal threshold from contributing nothing. So it reports
  more inliers than the others and a larger geometric error, and up to a fifth of
  the inliers it returns lie beyond the threshold you asked for — 325 of 1519 past
  3.0 at `inlier_threshold = 3.0`, and none past its own 4.5.
- **`ScoringType.MINPRAN`** estimates the noise scale itself; the threshold only
  bounds the candidate set it searches. Give it a generous value, or it truncates
  the range it needs: at sigma 1.5 it goes 0.79 px / recall 0.57 at threshold 3.0,
  to 0.33 / 0.85 at 6.0, to 0.26 / 0.90 at 9.0.

When checking that reported inliers satisfy the threshold, compare against the
scoring's own effective value, not the configured one. Measuring against the
configured value is what made Grid look like it was leaking inliers.
