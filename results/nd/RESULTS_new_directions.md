# New-directions results (first full pass, 2026-10-30)

Tables: `q1_summary.md`, `q2_summary.md`, `q4_summary.md`, `q4_profile.csv`, `fingerprint.md`.
Figures: `figures/` (png + pdf). Code: `src/q1_error_budget.py`, `q2_flow_detect.py`, `q4_real_data.py`,
`q4_profile.py`, `fingerprint.py`, `nd_figures.py`; Rockfish batch in `cluster/submit.sh`.

## Q1 — Can model and measurement errors fake "enhanced diffusion"? (52 cases)

Synthetic truth: DTI tensor field, alpha = 1, r = 0.018/h. Analyst fits (alpha, r) to tissue curves.
Figure: `figures/q1_apparent_enhancement`.

| error (one at a time) | alpha, tissue curves | alpha, native field |
|---|---|---|
| none | 1.000 | 1.000 |
| node noise 0.02 mmol/L | 1.006 | 1.001 |
| CSF input linear between sessions | 1.47 | 1.30 |
| CSF input 15% low / 15% high | 1.38 / 0.74 | 1.09 / – |
| CSF partial volume, 1 mm / 2 mm | 1.84 / 3.25 | 1.00 / – |
| clearance fixed at 0 | 0.54 | – |
| literature (tortuosity) field | 0.81 | 0.79 |
| combined input + partial-volume errors | **4.43** | 1.41 |
| truth alpha 3.5 (Vinje), combined errors | 6.91 | 1.59 |
| tissue flow 1 / 2 / 4 um/min (drainage) | 1.10 / 1.21 / 1.49 | 1.11 / 1.25 / 1.57 |
| tissue flow 4 um/min (basal-dorsal / pial-ventricle) | 1.17 / 0.99 | 1.18 / 1.16 |

Message: realistic input and partial-volume errors produce apparent enhancement as large as the
published alpha = 3.5; tissue flow at physiological speeds produces much less. Field objectives
reduce but do not remove the bias, and can bias it the other way.

## Q2 — Can the data tell flow from diffusion? (160 fits)

Figure: `figures/q2_flow_speed_recovery`.

- Exact boundary input and observations: the flow speed is recovered exactly for every pattern
  and speed (0.5–4 um/min) from every observation type; flow at 0.5 um/min is detected with
  probability 0.89 from tissue curves and ~1 from depth, blocks or field.
- With the combined realistic errors:
  - the drainage pattern absorbs the errors: its fitted speed runs to the 20 um/min bound for
    every truth, including the truth with no flow (a certain false positive);
  - tissue curves miss real pial-ventricle and basal-dorsal flow (fitted speed 0);
  - the native field recovers those two patterns' speeds roughly (0.96, 3.90; 0.82, 3.46 for
    true 1, 4) but also finds small spurious speeds (0.12–0.17) when there is no flow.
- Rejecting the diffusion-only model is not evidence of flow: with realistic errors it is
  rejected even without flow.

## Q4 — The real Gonzo data (54 fits + 60-point alpha profile)

Figures: `figures/q4_alpha_by_choice`, `figures/q4_alpha_profile`, `figures/fingerprint`.

- Fitted alpha ranges 0.38–9.0 across field x CSF-input x objective x session window. Only-24 h fits
  give alpha 4.5–9 with r 0.28–0.38/h (the Vinje-like corner); all-session tissue curves give 0.4–0.9.
- No choice fits: misfit is >1000x the iid-noise expectation for regional and depth observations and
  ~2.6–3.4x for the native field.
- Profile (alpha fixed, r refitted; chi^2 rise scaled by the misfit per observation):
  - tissue curves: alpha is not identified (linear input: rise < 3.84 for every alpha 0.3–8);
  - depth profiles: linear input, flat for alpha >= 1.4 (alpha and r rise together);
  - native field: sharp minimum, alpha ≈ 1.0–1.15 (curve input) or ≈ 1.7–2.0 (linear input).
    The scaling assumes independent nodal residuals; spatially correlated residuals would widen it.
- Flow models on the real data: the drainage pattern is rejected (speed 0) in every space; a
  pial-to-ventricle flow of 2.6–3.4 um/min improves the field misfit by only 0.2–0.3%; with tissue
  curves, flow replaces diffusion (alpha driven to its lower bound, speed 8–20 um/min).
- Fingerprint: the best diffusion fit predicts too much tracer at the surface and too little at
  4 mm and deeper after 24 h. This pattern is most similar to an over-estimated CSF input (cosine +0.73),
  opposite to partial volume or an under-estimated input (−0.6 to −0.7), and moderately similar to
  flow (+0.4). Known error signatures explain ~71% of the squared misfit.

## Q5 — CSF-side unknowns as nuisance parameters (66 fits; q5_summary.md)

- Synthetic (truth alpha 1, combined errors): freeing input scale + partial volume takes alpha from
  7.50 / 1.72 / 1.35 (curves / depth / field) to 1.23 / 1.26 / 1.20; misfit drops several hundred-fold.
  Nuisances over-compensate (scale 1.27–1.40 vs 1.18, f0 0.58–0.67 vs 0.5) because the linear-input
  time course is not representable; residual alpha bias +17–26%. Truth alpha 3.5: 3.99–4.20.
- Real: partial volume alone never used (f0 = 0). Input scale 0.63–0.77 in every space and for both
  inputs. Tissue-curve misfit down 6–12x, field chi2/n 2.77 -> 2.36 (curve), 2.58 -> 2.23 (linear).
  alpha 1.06–1.99 (curve input), 1.46–2.46 (linear input), vs 0.47–1.75 without nuisances.
  Exchange multiplier alone is confounded with alpha (kx 0.09–0.35, alpha up to 8.4).

## Robustness (mesh_check.md, q6_summary.md, geometry_compare.md)

- Mesh: coarse analyst on refined no-flow truth gives alpha 0.92 (-8% mesh bias). Flow signal error
  43–99%, but flow-induced enhancement (alpha / alpha_noflow on the same mesh) changes by <= 4 points.
- Correlated noise (half of variance, ell 2.5/5/10 mm): tissue-curve noncentrality for drainage 1 um/min
  49 -> 9.6 / 1.9 / 0.48 (power 1.00 -> 0.87 / 0.29 / 0.11). Depth profiles 11–49 (power >= 0.91);
  8 mm blocks 60–610. Real residual: normalised covariance 0.4–0.6 at 1–2 mm, 0.05–0.15 at 8–10 mm.
- Geometry (Vinje meshes 14 and 4): Q1 budget reproduced (linear input 1.47 everywhere; partial volume
  1.84–2.11; combined 4.43/4.69; drainage 4 um/min 1.49–1.64). Q5 synthetic nuisance fit 1.17–1.23 on all
  three. Field objective under combined errors depends on weighting: per-vertex whitening 0.86–0.90 on
  Vinje vs 1.35 on Gonzo.

## Second round (2026-10-31): A published pipeline, B uncertainty, D perfusion, E groups

"field" below = per-node whitened native field (every node weight 1); the publication's objective is the
volume-weighted L2(Omega) norm (third-round S6w runs check this).

### A — published pipeline (q7_summary.md; figure q7_published_pipeline)

- Synthetic, step by step from the Q1 combined-error field fit (truth alpha 1 / 3.5):
  S1 1.35 / 1.16; + scans <= 48 h 1.38 / 1.11; + regional isotropic D 1.41 / 1.13; + alpha cortex only 1.69 / 1.37;
  + Dirichlet from observed boundary nodes 0.97 / 1.86; + bounds [1, 10] 1.00 / 1.86 (= published pipeline).
- Pipeline structure alone (exact data): 1.32 / 2.91. Noise but no partial volume: 1.26 / 2.59. Tissue
  curves instead of field: 1.00 / 2.19. -> The published pipeline does not inflate alpha; it compresses a
  large alpha, mainly through CSF partial volume in the prescribed boundary values.
- Note: the field objective with the combined errors is shallow along the alpha-r ridge (truth 3.5: cost
  0.4827 at alpha 1.16, 0.4848 at 1.59, 0.5239 at the truth), so Q1's 1.59 (volume-weighted field) and q7's
  1.16 (whitened field) for the same nominal case are both within 0.5% of the minimum.
- Gonzo (linear input): S1 1.70, S2 1.91, S3 2.18, S4 1.39, S5 = S6 = 1.336 (r 0.010 1/h), tissue curves 1.34.
  Curve input: S1–S4 0.93–1.34. Dirichlet fits the field far better (chi2/n 0.97 vs 2.6–3.0 with Robin input).
  -> The published pipeline on this subject gives 1.34, not 3.5.

### B — uncertainty (q7u_summary.md)

- Cluster-robust SE of alpha (field, cubic blocks pooled over sessions): design effect kappa 5–10 (8 mm),
  9–22 (16 mm), 14–52 (32 mm); SE 0.02–0.05 (8 mm) to 0.045–0.106 (32 mm). E.g. amp_pv: curve 1.414 +- 0.045,
  linear 1.857 +- 0.058; base: curve 1.151 +- 0.051, linear 1.700 +- 0.106 (32 mm).
- Profiles with (r, amp, f0) refitted, threshold 3.84 kappa_max: field curve 1.4 (grid 1.2–1.7 excluded),
  linear 1.7–2.0; alpha 3.5 lies 5,800–11,900 scaled units above the minimum (threshold 55–64).
  Depth profiles: kappa < 1 with 14 row clusters (unreliable); iid profile 1.0–1.4 (curve), 2.0 (linear).
- -> Sampling uncertainty is ~+-0.05; spread across analysis choices (0.9–2.2) dominates.

### D — perfusion (q8_summary.md; figure q8_perfusion)

- Exact data, diffusion-only fit to perfusion truths (alpha 1, r 0.018): Q15/1 mm -> alpha 1.40–1.50;
  Q15/3 mm -> 2.7–3.3; Q45/1 mm -> 2.1–2.5; Q45/3 mm -> 6.2–10 (field chi2/n 0.01–0.14, i.e. residual below
  the noise level per node). Perfusion model with true ell_p recovers Q exactly.
- Combined errors: no-perfusion truth fitted with Q = 74–86 mL/h (depth, field; 324 tissue curves) and the
  perfusion model fits it better than the true diffusion model (chi2/n 4.00 vs 4.21). Q15 truths -> 49–126,
  Q45 -> 133–201; "both" fits drive alpha to 0.1–0.3. -> Q not identified under input errors.
- Gonzo, alpha 1, Q free: Q 0–7 mL/h with curve input, 4–21 mL/h with linear input (field/depth), below
  45 +- 20 mL/h (Wahlin 2026). With input scale free the perfusion model (alpha 1) fits as well as
  enhanced diffusion (linear field: chi2 897k vs 902k for alpha 1.54); alpha with Q free 0.16–1.6.

### E — group comparisons (q9_summary.md; figure q9_groups)

- Exact analysts: all differences exact.
- CSF kinetics only (no transport difference): tissue curves with errors give spurious Delta alpha -22% /
  +33% (CSF slower / faster, slow clearance) and -28% (fast clearance); native field -2 to -9% and Delta r
  -13% (fast clearance); published pipeline <= 5%.
- True clearance -33%: published pipeline recovers -15% (r 0.018) / -27% (r 0.12) (r biased up:
  0.052 vs 0.018, 0.178 vs 0.12); native field with errors: r at 0 for slow clearance (the difference
  appears as +8% alpha instead), -19% with +40% spurious alpha for fast clearance; tissue curves -29%
  with +16% alpha.
- -> Errors do not cancel between groups when CSF kinetics differ; the Dirichlet pipeline is robust to
  CSF kinetics but attenuates clearance differences.

### Third round (q7 list3 / list3b)

- Pipeline calibration, truth alpha in cortex only (r 0.018): returned 1.00 / 1.57 / 2.22 for true 1 / 2 / 3.5
  (S6, combined errors); S6w (volume-weighted L2) 1.00 / 1.56 / 2.20; no partial volume 1.26 / 2.44 / 4.08;
  exact observations 1.32 / 2.57 / 4.35 (S6w exact 1.30 / - / 4.43). Vinje truth (3.5 everywhere) S6w 1.64.
- Gonzo: S6 1.336 +- 0.022 / 0.034 / 0.044 (8 / 16 / 32 mm blocks; kappa 11 / 27 / 46); S6w 1.452.
  -> true cortical alpha 1.6–1.8 (with boundary PV as modelled) or 1.1–1.2 (no PV); returned 3.5 needs true >= 2.8.
- Perfusion truths through the pipeline: S6 1.25 (Q15/1 mm), 1.70 (Q15/3), 1.73 (Q45/1), 2.84 (Q45/3);
  exact 1.86, 3.22, 2.90, 7.79.
- Dirichlet + perfusion (alpha 1; Q, r): no-perfusion truth Q 19 (chi2 175,323 -> 170,940); Q15/1 -> 45; Q45/1 -> 89;
  Q45/3 -> 62. D_both drives alpha to its bound 1 (except Q45/3: 1.43).
- Gonzo Dirichlet + perfusion: Q 12.5 (1 mm), 5.5 (3 mm); chi2 285,767 / 279,643 vs 291,229 for S6 (alpha 1.34);
  D_both -> alpha 1 (bound). Improvement 5,500–11,600 vs 4,400 from errors alone in synthetic.

## Open checks

1. Vinje flow-model fits (q2 flow) and remaining Vinje Q1/Q5 cases running.
2. Optional: experimental-design section; time-resolved CSF-input nuisance model.
3. Third round done (see above). Manuscript v2.0 rewritten 2026-10-31 around these results.
