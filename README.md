# Brain transport: regional calibration and diffusion-field mismatch

Code for the article *Coarse regional calibration can conceal diffusion-field mismatch in a human-derived brain
transport model* (submitted to *Journal of the Royal Society Interface*).

The model is tracer diffusion with clearance in one human-derived brain geometry,

    dc/dt = div(s_D D(x) grad c) - r c,   with a Robin exchange with the subarachnoid space on the brain surface,

calibrated by fitting two scalars, the diffusion scale `s_D` and the clearance rate `r`, to three regional tissue
concentration curves. The repository contains:

* `dumux_miniapp/`: the official DuMux brain-diffusion solver used for the forward simulations, with the
  modifications made for this study (GPL-3.0-or-later; see `dumux_miniapp/NOTICE.md`).
* `src/twin.py`: an independent Python finite-element re-implementation of that solver (P1 stiffness, lumped mass,
  backward Euler, Robin exchange, uniform clearance), with the regional observation operators, two-scalar fitting with
  exact forward sensitivities, the regional/hidden residual split, uniform red mesh refinement, a positivity-preserving
  variant and the coefficient fields.
* `src/`: the analyses built on it, listed below.

## What each script reproduces

| Script | Result in the article |
| --- | --- |
| `step0_validate.py` | Re-implementation check against 33 archived official trajectories (§2.8, ESM S12) |
| `stepA_numerics.py`, `stepA_level2.py` | Tensor T2 case on the three mesh levels and with the positivity-preserving variant (§3.7, Table S10) |
| `convergence_case.py` | T1, T3, T3wide and E5 on the three mesh levels (§3.7) |
| `convergence_summary.py` | Richardson extrapolation and observed orders (Table 2, Figure 6, Table S9) |
| `stepC1_effective.py`, `homog.py` | Fitted `s_D` against volume averages, homogenisation and tracer weighting (§3.8, Figure 7, Table S11) |
| `run_case.py` | Concentration convention (`B_*`, Table S13), diffusion-field scale series (`C2_*`, Table S11), time-window and inflow transfer (`C3a_*`, `C3b_*`, Table S12) and the widened `s_D` bound (`D1_tensorT3`) |
| `analyze_C2.py` | Predictions for the `C2_*` scale series (Table S11) |
| `collect_C3.py` | Time-window and inflow-transfer tables from the `C3a_*` and `C3b_*` cases (Figure 7, Table S12) |
| `experiments.py` | Shared truth, fit and score routine |

## Requirements

Python 3.10 or later with the packages in `requirements.txt` (`pyamg` is needed only for the refined meshes).
Use single-threaded BLAS; multi-threaded BLAS oversubscribes the machine:

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

Building `dumux_miniapp/` needs DUNE and DuMux; see `dumux_miniapp/README.md`.

## Input data

The geometry, diffusion data and boundary concentrations derive from the public Gonzo human dataset
(Zenodo, doi:10.5281/zenodo.14266867). The scripts read the derived input files from a data folder with this layout
(the default location is `data/` next to `src/`; set `BT_DATA` to use another folder):

    04_DATA/geometry/fem_input.npz                         mesh, boundary and tissue labels
    04_DATA/plotting_assets/representative_fields.npz      diffusion tensors and labels in solver cell order
    04_DATA/observation_operators/operators.npz            regional observation operators
    04_DATA/diffusion_fields/E3_coefficient_mesh.vtu       heterogeneous diffusion multipliers
    04_DATA/diffusion_fields/E4_coefficient_mesh.vtu
    04_DATA/trajectories/DataTensor_T2.npz                 archived official trajectory
    03_RESULTS/phase8e_exact_checks.csv                    archived official fits (Step 0)
    03_RESULTS/phase8h_operator_atlas.csv                  archived error atlas (stepC1_effective.py)

`step0_validate.py`, `stepC1_effective.py` and `analyze_C2.py` also read archived official trajectories
(`04_DATA/trajectories/*.npz`, `03_RESULTS/phase8c_library_manifest.csv`) from `data_extension/`, or from the folder
named by `BT_DATA_EXTENSION`.

`results/` already contains the two inputs that every other script needs: `boundary_params_recovered.npy`, the
per-vertex boundary-input parameters used by the official solver (recovered by `step0_validate.py`), and
`official_clock_hours.npy`, the official 75-state time clock.

## Running

From `src/`:

    python step0_validate.py
    python stepA_numerics.py 0 standard          # level 0 (working mesh); also: 1, and mmatrix
    python stepA_level2.py standard <s_D0> <r0>  # level 2, one Gauss-Newton step from the level-1 fit
    python convergence_case.py T1 0 fit          # cases T1 T2 T3 T3wide E5; levels 0 and 1: fit, level 2: gn
    python convergence_summary.py
    python stepC1_effective.py
    python run_case.py C2_ell8                   # any case name listed above, optionally followed by a level
    python analyze_C2.py
    python collect_C3.py

Results are written to `results/`. On the working mesh (about 0.10 million nodes) a case takes 4–12 minutes on one
core. The refined levels (0.74 and 5.61 million nodes) ran as single jobs on a Slurm cluster: a level-1 fit on one core
and a level-2 Gauss–Newton step on 32 cores with 4 GB per core, each within 36 hours.

## Implementation notes

* **Cell order.** `fem_input.npz` and `representative_fields.npz` store the same tetrahedra in different orders.
  Cell-wise coefficients and tissue labels belong to the order in `representative_fields.npz`, which the solver uses.
* **Boundary input.** The official solver fits `B = a tau exp(-b tau)` to the five public checkpoints at each boundary
  vertex. At about 100 of 47,621 vertices its fit stops at a different `(a, b)` from a converged fit, so the
  re-implementation uses the exact parameters recovered from an archived official run.
* **Refined meshes** use red refinement (each tetrahedron into 8 along the shortest interior diagonal), inherited labels
  and coefficients, and the piecewise-linear interpolant of the coarse boundary input, so every level sees the same
  continuous data.

## Licence

The code in `src/` is released under the MIT licence (`LICENSE`). `dumux_miniapp/` is a modified copy of the DuMux
brain-diffusion miniapp (https://github.com/timokoch/dumux-braindiffusion-miniapp, commit 1b1ca8c) and remains under
GPL-3.0-or-later (`dumux_miniapp/LICENSE.md`).
