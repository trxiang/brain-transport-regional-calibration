
### A0: Perfect analyst (true field, true CSF input, 4 sessions, tissue curves)

| case       |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:-----------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| A0_perfect |            1 |           1 |                          0 |   0.018 |            0 |               0 | False          |

### A1: CSF input linearly interpolated between sessions

| case         |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:-------------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| A1_linear_bc |            1 |      1.4728 |                    47.2768 |  0.0241 |       2.0069 |          5.1925 | False          |

### A2: CSF input amplitude bias

| case       |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:-----------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| A2_amp0.7  |            1 |      2.0342 |                   103.416  |  0      |      18.05   |         26.4605 | True           |
| A2_amp0.85 |            1 |      1.3847 |                    38.4725 |  0.0099 |       8.5703 |         12.9329 | False          |
| A2_amp1.15 |            1 |      0.736  |                   -26.4021 |  0.0241 |       7.9097 |         12.7031 | False          |
| A2_amp1.3  |            1 |      0.5456 |                   -45.4399 |  0.0287 |      15.3223 |         25.3603 | False          |

### A3: CSF partial volume in tissue values

| case          |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:--------------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| A3_pv0.25     |            1 |      1.3817 |                    38.1719 |  0.0164 |       8.7854 |          9.4902 | False          |
| A3_pv0.5      |            1 |      1.8432 |                    84.3195 |  0.0143 |      16.9435 |         18.1821 | False          |
| A3_pv0.5_ell2 |            1 |      3.2451 |                   224.506  |  0.0144 |      28.5147 |         30.9362 | False          |

### A4: Node noise sigma 0.02 mmol/L

| case     |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:---------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| A4_noise |            1 |      1.0059 |                     0.5915 |   0.018 |       0.1128 |          0.1328 | False          |

### A5: Sessions used

| case     |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:---------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| A5_dense |            1 |           1 |                          0 |   0.018 |            0 |               0 | False          |
| A5_win24 |            1 |           1 |                         -0 |   0.018 |            0 |               0 | False          |

### A6: Clearance fixed at zero

| case            |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:----------------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| A6_fix_r0       |            1 |      0.5404 |                   -45.9631 |       0 |      11.7367 |         15.0763 | False          |
| A6_fix_r0_win24 |            1 |      0.75   |                   -25.0015 |       0 |      14.2256 |         14.975  | False          |

### A7: Full-field objective

| case               |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:-------------------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| A7_field_amp0.85   |            1 |      1.0923 |                     9.2283 |  0.0085 |       9.5289 |         11.9226 | False          |
| A7_field_linear_bc |            1 |      1.3016 |                    30.1646 |  0.0199 |       2.6323 |          4.7102 | False          |
| A7_field_noise     |            1 |      1.0013 |                     0.1291 |  0.018  |       0.0604 |          0.0615 | False          |
| A7_field_perfect   |            1 |      1      |                     0      |  0.018  |       0      |          0      | False          |
| A7_field_pv0.5     |            1 |      1.0035 |                     0.3452 |  0.0119 |       5.4076 |          5.3395 | False          |

### B1: Literature (tortuosity) diffusion field

| case                         |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:-----------------------------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| B1_literature_field          |            1 |      0.8095 |                   -19.0501 |  0.0194 |       0.6621 |          4.6593 | False          |
| B1_literature_field_obsfield |            1 |      0.7947 |                   -20.532  |  0.0188 |       0.7357 |          4.6431 | False          |

### B2: DTI isotropic (no anisotropy) field

| case             |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:-----------------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| B2_dataiso_field |            1 |      1.0154 |                     1.5396 |  0.0184 |       0.1837 |          2.6581 | False          |

### C1: Combined realistic errors

| case                  |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:----------------------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| C1_combo              |            1 |      4.4322 |                   343.217  |  0.0064 |      18.8313 |         27.9082 | False          |
| C1_combo_field        |            1 |      1.4113 |                    41.1306 |  0.0038 |      11.6037 |         15.4492 | False          |
| C1_combo_literature   |            1 |      3.4931 |                   249.31   |  0.0082 |      18.8242 |         27.9916 | False          |
| C1_combo_win24_fix_r0 |            1 |      4.4544 |                   345.435  |  0      |      22.3721 |         30.9177 | False          |

### D: Other truths (no clearance; genuine alpha 3.5)

| case                |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:--------------------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| D_r0_combo          |          1   |      4.8841 |                   388.407  |  0      |      15.6092 |         27.0509 | True           |
| D_r0_perfect        |          1   |      1      |                     0.0001 |  0      |       0      |          0      | False          |
| D_vinje_combo       |          3.5 |      6.9122 |                    97.4925 |  0.0623 |      26.8016 |         29.2726 | False          |
| D_vinje_combo_field |          3.5 |      1.5938 |                   -54.4619 |  0.0393 |      15.403  |         17.3558 | False          |
| D_vinje_perfect     |          3.5 |      3.5    |                     0      |  0.186  |       0      |          0      | False          |

### E: Truth with tissue flow, analyst without flow

| case                                  |   alpha_true |   alpha_fit |   apparent_enhancement_pct |   r_fit |   roi_L2_pct |   native_L2_pct | active_bound   |
|:--------------------------------------|-------------:|------------:|---------------------------:|--------:|-------------:|----------------:|:---------------|
| E_flow_basal_to_dorsal_1              |            1 |      1.0286 |                     2.865  |  0.0185 |       1.1812 |          4.5578 | False          |
| E_flow_basal_to_dorsal_1_field        |            1 |      1.0198 |                     1.9765 |  0.0181 |       1.2024 |          4.551  | False          |
| E_flow_basal_to_dorsal_2              |            1 |      1.0651 |                     6.5126 |  0.0186 |       2.7231 |          8.886  | False          |
| E_flow_basal_to_dorsal_2_combo        |            1 |      4.8224 |                   382.235  |  0.0055 |      18.745  |         29.1135 | False          |
| E_flow_basal_to_dorsal_2_field        |            1 |      1.0598 |                     5.9827 |  0.0183 |       2.7324 |          8.883  | False          |
| E_flow_basal_to_dorsal_4              |            1 |      1.1721 |                    17.2083 |  0.0185 |       5.9025 |         16.1435 | False          |
| E_flow_basal_to_dorsal_4_field        |            1 |      1.1841 |                    18.4097 |  0.0189 |       5.9046 |         16.1422 | False          |
| E_flow_pial_inflow_drainage_1         |            1 |      1.1001 |                    10.0058 |  0.0155 |       0.6086 |          2.7131 | False          |
| E_flow_pial_inflow_drainage_1_field   |            1 |      1.1149 |                    11.4911 |  0.0158 |       0.6396 |          2.7006 | False          |
| E_flow_pial_inflow_drainage_2         |            1 |      1.2142 |                    21.4246 |  0.0133 |       1.2282 |          5.0847 | False          |
| E_flow_pial_inflow_drainage_2_combo   |            1 |      5.4493 |                   444.927  |  0      |      16.8762 |         26.7139 | True           |
| E_flow_pial_inflow_drainage_2_field   |            1 |      1.2482 |                    24.8241 |  0.0139 |       1.2947 |          5.0555 | False          |
| E_flow_pial_inflow_drainage_4         |            1 |      1.4913 |                    49.1288 |  0.0103 |       2.319  |          8.7633 | False          |
| E_flow_pial_inflow_drainage_4_field   |            1 |      1.5673 |                    56.727  |  0.0111 |       2.4353 |          8.708  | False          |
| E_flow_pial_to_ventricle_flux_1       |            1 |      0.9714 |                    -2.8578 |  0.0152 |       0.6386 |          4.0212 | False          |
| E_flow_pial_to_ventricle_flux_1_field |            1 |      1.0153 |                     1.531  |  0.0163 |       0.912  |          3.9276 | False          |
| E_flow_pial_to_ventricle_flux_2       |            1 |      0.9603 |                    -3.9735 |  0.0124 |       1.3927 |          7.6782 | False          |
| E_flow_pial_to_ventricle_flux_2_combo |            1 |      3.9519 |                   295.188  |  0.0005 |      17.9429 |         26.5479 | False          |
| E_flow_pial_to_ventricle_flux_2_field |            1 |      1.0494 |                     4.9365 |  0.0146 |       1.9566 |          7.4609 | False          |
| E_flow_pial_to_ventricle_flux_4       |            1 |      0.9941 |                    -0.5852 |  0.008  |       3.4247 |         13.5621 | False          |
| E_flow_pial_to_ventricle_flux_4_field |            1 |      1.1601 |                    16.0138 |  0.0119 |       4.326  |         13.1246 | False          |
