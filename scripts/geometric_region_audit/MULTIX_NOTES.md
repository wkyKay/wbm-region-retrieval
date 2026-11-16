# Multi-X line/circle region audit: implementation scope

This is a **Multi-X line+circle adaptation with model validation**, not a claim
that the published MixedWM38 benchmark or the authors' line/circle experiment
has been reproduced. The algorithm consumes only the coordinates of `raw == 2`.
Neither category labels nor the number of labels is passed to it.

## Primary sources

- Daniel Barath and Jiri Matas, *Multi-Class Model Fitting by Energy Minimization
  and Mode-Seeking*, ECCV 2018, pp. 221–236.
  <https://cmp.felk.cvut.cz/~matas/papers/barath-2018-multi_X-eccv.pdf>
- Author implementation <https://github.com/danini/multi-x>, downloaded at commit
  `60f71708a0e02a2a9192f59b6509556137b1fa61`.
- The downloaded author source is kept in `author_multi_x/`. Its `src/main.cpp`
  TODO explicitly lists missing 2D line application, median shift and adaptive
  parameter setting. It provides a line estimator, but **no circle estimator**.
  Therefore merely compiling its demo cannot reproduce line/circle fitting.

## Executed components

1. Generate random minimal-sample line and circle hypotheses jointly.
2. Separately cluster each model class in a spatial point representation, then
   replace nearby hypotheses with modes, refitting their union of supporting points.
3. Assign every observed point to one model or an outlier label using alpha
   expansion, a spatial Potts term and per-model label costs.
4. Refit each model to its assigned points and repeat. Mode replacement after the
   first iteration is rejected if its optimized energy increases (paper Eq. 5).
5. Apply **paper Algorithm 2 model validation**: 100 times draw a minimal sample
   from the assigned cluster, fit a model, and compute its mean distance to all
   points of that cluster. Retain the cluster only when the mean over all trials
   is below gamma. This is not a threshold on the fraction of successful trials.

The original C++ `GCoptimization.cpp`, `LinkedBlockList.cpp`, graph and energy
headers run directly through `local_adapter/gco_bridge.cpp`; this is not a greedy
RANSAC replacement or a nearest-model-only assignment. No original author file is
edited. The only portability shim, `local_adapter/ppl.h`, is empty because the
author source includes Microsoft's PPL header while its only uses are commented.

The Python outer loop ports the public framework. The circle estimator is local:
three-point algebraic fit and nonminimal nonlinear least-squares radial residual;
lines use orthogonal total least squares. All outputs are hard labels on the
original defective coordinates, with `0` meaning unassigned or rejected.
Output regions are sorted by area, with raster-order ties. A point receives at
most one region; intersecting models do not share their crossing pixels.

## Parameters and explicit departures

`multix.PARAMETERS` is the authoritative machine-readable configuration.
All 38 real samples use the same settings and the same seed convention.

| Component | Executed setting | Evidence / adaptation |
| --- | --- | --- |
| Outlier threshold gamma | 2 pixels | Published line/circle setting, p. 8 |
| Potts spatial weight | 0.3 | Published p. 8 |
| Hypothesis number | Total 2N, divided N line + N circle | Paper says 2N; distribution across classes is a local convention |
| Sampling | Uniform without replacement | Public framework supports uniform; paper uses NAPSAC. Uniform avoids tiny pixel-neighborhood degenerate circle samples |
| Spatial graph | Euclidean neighbors within 1.5 pixels, undirected once | Local wafer-grid adaptation |
| Mode seeking | Public-code pointwise-mean-distance mean shift; bandwidth 2 pixels | Paper uses median shift + Hausdorff + adaptive k-neighbor bandwidth. These are **not** claimed reproduced |
| Line representation | Nearest point to origin and point one image-side length along canonical tangent | Paper-style canonical point representation, fixed length is a local choice |
| Circle representation | Four ordered cardinal points | Local missing-estimator implementation |
| Label cost | m log(N) / 10; m=2 for line, 3 for circle; outlier cost 0 | Paper formula; hmax=10 is a fixed implementation setting, not the true class count and not a hard output cap |
| Data energy | d²/(2.25 gamma²) if <=1, otherwise 2; outlier unary 1 | Exact public `MultiX.h::data_energy`, different from a literal unnormalized reading of paper Eq. 2 |
| Refit | Total/nonlinear least squares; retain old model if truncated data cost increases | Public nonminimal-refit structure; local safeguard |
| Iterations | At most 10; stop on unchanged energy | Public framework default scale |
| Final model validation | 100 MSS fits, mean residual < gamma | Paper Algorithm 2, missing from the public implementation's basic `is_valid` check |
| Degenerate CV draw | Resample at most 100 times, then reject model | Local rule for collinear triples / discrete pixels; all resampling counts logged |
| Circle radius | 0.5 to twice image diagonal | Local numerical bound; avoids arbitrarily large circles approximating lines |
| Minimum support | MSS only: line 2, circle 3 | F's area >=5 and top4 are applied separately by the comparison runner |

After CV rejection, pixels from a rejected model remain unassigned. They are not
silently reassigned to a surviving model. No raster filling, interpolation,
erosion, dilation, class-conditioned selection, or connected-component merge is
used. A nonconvex spatial region is not constrained to be connected.

This geometric model family is deliberately limited: dense Loc/Near-full/Random
patterns may be broken into lines and circles, or rejected. Such output is a
limitation to inspect, not hidden by changing the sample or per-class parameters.

## Build and validation

The wrapper auto-builds the native library if absent, using:

```sh
clang++ -std=c++14 -O2 -shared -fPIC -include cfloat \
  -Ilocal_adapter -Iinclude local_adapter/gco_bridge.cpp \
  include/GCoptimization.cpp include/LinkedBlockList.cpp \
  -o local_adapter/libmultix_gco.dylib
```

This was checked on the current macOS host. C++14 avoids upstream legacy
`register` warnings becoming C++17 errors. No OpenCV installation is required by
the thin graph-cut wrapper. It uses the original float energy type.

Implementation smoke checks (seed 2026, not data-driven model selection):

- Two one-pixel perpendicular lines: returns two line regions, 79/79 original
  defective pixels assigned; crossing neighborhood follows exclusive Potts labels.
- One-pixel circle crossed by a horizontal line: returns one circle and one line,
  126/126 defective pixels assigned. This is an easy synthetic structural check,
  not proof of correctness on arbitrary real defect mixtures.
- One fixed real audit sample, ID 19295, 810 defect points: roughly 1.4 seconds;
  a repeat gives identical masks. No parameter tuning to that sample was performed.
- Independent checks confirm output pixels are a subset of the original defects,
  labels are contiguous, and each accepted mode move after the first iteration
  has no larger energy than a freshly optimized current-model baseline.
- The graph-cut bridge was independently checked on 20 random small problems:
  reported energy agrees with a direct sum of unary, undirected Potts and active
  label costs (maximum floating-point discrepancy approximately 2.75e-7).

The original repository includes copyright and redistribution notices in its
source headers, including separate notices for GCoptimization. Preserve them.
