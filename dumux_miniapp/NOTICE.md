# Modified copy of the DuMux brain-diffusion miniapp

This directory is a modified copy of https://github.com/timokoch/dumux-braindiffusion-miniapp at commit
1b1ca8cda6ec7be79daafcb8e27b4fd5289ee051, distributed under GPL-3.0-or-later (`LICENSE.md`, `LICENSES/`).
Only `app/main.cc` was changed; `CHANGES_FROM_UPSTREAM.diff` is the complete diff. The changes, made in 2026 for the
study in the parent repository, are:

* `Problem.DiffusionType` is resolved into the problem's diffusion type (`DataTensor`, `DataIsotropic` or
  `LiteratureIsotropic`) and the resolved settings are printed at start-up.
* Optional tissue-specific clearance (`Problem.TransferCoeffToBloodGM`, `Problem.TransferCoeffToBloodWM`) and
  diffusion scales (`Problem.GMDiffusionScale`, `Problem.WMDiffusionScale`), and an optional cell-wise diffusion
  multiplier read from the grid (`Problem.UseHeterogeneityField`, cell array `phase8h_scale`). All default to the
  original behaviour.
* VTK output at every time step when `vtkAtCheckPoints` is false (dense output), instead of no output.
