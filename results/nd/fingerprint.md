| signature | alpha_fit | cos vs curve | cos vs linear |
|---|---|---|---|
| A1_linear_bc | 1.47 | -0.41 | -0.23 |
| A2_amp0.85 | 1.38 | -0.74 | -0.68 |
| A2_amp1.15 | 0.74 | +0.73 | +0.68 |
| A3_pv0.5 | 1.84 | -0.61 | -0.56 |
| A3_pv0.5_ell2 | 3.25 | -0.63 | -0.57 |
| A6_fix_r0 | 0.54 | +0.16 | +0.08 |
| B1_literature_field | 0.81 | +0.02 | +0.07 |
| E_flow_pial_inflow_drainage_2 | 1.21 | +0.42 | +0.46 |
| E_flow_pial_inflow_drainage_4 | 1.49 | +0.38 | +0.41 |
| E_flow_pial_to_ventricle_flux_2 | 0.96 | +0.41 | +0.37 |
| C1_combo | 4.43 | -0.64 | -0.56 |

NNLS decomposition of the real misfit (tensor__curve__roi__all__diff): explained 71.2% of the squared misfit
| signature | weight |
|---|---|
| A2_amp1.15 | 0.1897 |
| A3_pv0.5 | 0.0976 |
| A2_amp0.85 | 0.0351 |
| E_flow_pial_to_ventricle_flux_2 | 0.0311 |
| B1_literature_field | 0.0233 |
| E_flow_pial_inflow_drainage_4 | 0.0119 |
| A1_linear_bc | 0.0073 |

NNLS decomposition of the real misfit (tensor__linear__roi__all__diff): explained 73.0% of the squared misfit
| signature | weight |
|---|---|
| A2_amp1.15 | 0.1702 |
| A3_pv0.5 | 0.1033 |
| A1_linear_bc | 0.0289 |
| E_flow_pial_to_ventricle_flux_2 | 0.0264 |
| B1_literature_field | 0.0192 |
| E_flow_pial_inflow_drainage_4 | 0.0126 |
| A2_amp0.85 | 0.0071 |

Real misfit (observed - fitted depth means, mmol/L) for the curve-input fit:

| row | 4.4 h | 23.9 h | 47.9 h | 69.9 h |
|---|---|---|---|---|
| GM 0-1e-06 mm | -0.0223 | -0.0209 | -0.0100 | -0.0123 |
| GM 1e-06-2 mm | -0.0052 | -0.0051 | -0.0042 | -0.0074 |
| GM 2-4 mm | -0.0052 | +0.0060 | +0.0015 | -0.0038 |
| GM 4-6 mm | -0.0047 | +0.0330 | +0.0148 | +0.0081 |
| WM 0-1e-06 mm | -0.0229 | -0.0146 | +0.0064 | +0.0008 |
| WM 1e-06-2 mm | -0.0021 | +0.0009 | +0.0024 | -0.0025 |
| WM 2-4 mm | -0.0041 | +0.0101 | +0.0097 | +0.0031 |
| WM 4-6 mm | -0.0035 | +0.0145 | +0.0283 | +0.0225 |
| WM 6-inf mm | -0.0028 | +0.0038 | +0.0217 | +0.0212 |
| subcortical 0-1e-06 mm | -0.0025 | -0.0092 | +0.0106 | +0.0056 |
| subcortical 1e-06-2 mm | +0.0269 | -0.0020 | +0.0078 | +0.0103 |
| subcortical 2-4 mm | +0.0104 | +0.0031 | +0.0030 | +0.0062 |
| subcortical 4-6 mm | +0.0067 | +0.0203 | +0.0124 | +0.0096 |
| subcortical 6-inf mm | -0.0003 | +0.0082 | +0.0169 | +0.0121 |
